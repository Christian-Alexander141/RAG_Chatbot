import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rag_chatpipline import ad_auth, config

USER_DN = "CN=Max Muster,OU=Mitarbeiter,DC=firma,DC=local"
OTHER_DN = "CN=Erika Beispiel,OU=Mitarbeiter,DC=firma,DC=local"
ADMIN_DN = "CN=Anna Admin,OU=IT,DC=firma,DC=local"
GROUP_DN = "CN=Chatbot-Nutzer,OU=Gruppen,DC=firma,DC=local"
ADMINS_DN = "CN=Chatbot-Admins,OU=Gruppen,DC=firma,DC=local"
ACTIVE = {"userAccountControl": "512", "accountExpires": "9223372036854775807", "msDS-User-Account-Control-Computed": "0"}


class FakeAD:
    """Ein kleines AD: Konten, Gruppen, Mitgliedschaften und Passwörter, in jedem Test frisch."""

    def __init__(self):
        self.users = {
            USER_DN: {"sAMAccountName": "MMuster", "userPrincipalName": "m.muster@firma.de",
                      "displayName": "Max Muster", "mail": "m.muster@firma.de", **ACTIVE},
            OTHER_DN: {"sAMAccountName": "ebeispiel", "userPrincipalName": "ebeispiel@firma.local",
                       "displayName": "Erika Beispiel", "mail": "", **ACTIVE},
            ADMIN_DN: {"sAMAccountName": "aadmin", "userPrincipalName": "aadmin@firma.local",
                       "displayName": "Anna Admin", "mail": "", **ACTIVE},
        }
        self.groups = {"Chatbot-Nutzer": GROUP_DN, "Chatbot-Admins": ADMINS_DN}
        # verschachtelte Mitgliedschaften schon aufgelöst, wie es das AD mit IN_CHAIN tut
        self.members = {USER_DN: {GROUP_DN}, ADMIN_DN: {ADMINS_DN}}
        self.passwords = {"mmuster@firma.local": "geheim", "m.muster@firma.de": "geheim",
                          "ebeispiel@firma.local": "passwort", "aadmin@firma.local": "admin",
                          "svc-chatbot@firma.local": "dienst"}
        self.binds = []
        self.searches = 0
        self.drop_connection = False  # nächste Suche bricht ab, als hätte das AD die Verbindung getrennt
        self.unreachable = False

    def bind(self, name, password):
        if self.unreachable:
            raise ad_auth.ADError("Kein Domänencontroller hat die Anmeldung angenommen: dc01.firma.local: timeout")
        self.binds.append(name)
        return FakeConnection(self) if self.passwords.get(name.lower()) == password else None

    def service_binds(self):
        return [b for b in self.binds if b.startswith("svc-")]


class FakeConnection:
    """Antwortet auf die Suchen, die ad_auth stellt, wie ein Domänencontroller."""

    def __init__(self, ad):
        self.ad = ad
        self.response = []
        self.result = None

    def search(self, base, search_filter, search_scope="SUBTREE", attributes=None, size_limit=0):
        self.ad.searches += 1
        if self.ad.drop_connection:
            self.ad.drop_connection = False
            raise ConnectionResetError("Verbindung vom AD getrennt")
        users = self.ad.users
        hits = []
        if "(objectClass=group)" in search_filter:
            hits = [(dn, {}) for name, dn in self.ad.groups.items() if f"(cn={ad_auth.escape(name)})" in search_filter]
        elif search_filter.startswith("(memberOf:"):
            hits = [(base, {}) for group in self.ad.members.get(base, ()) if ad_auth.escape(group) in search_filter]
        elif search_filter == "(objectClass=*)" and search_scope == "BASE":
            # berechnete Attribute gibt es nur bei einer Suche auf das Objekt selbst
            computed = "msDS-User-Account-Control-Computed"
            hits = [(base, {computed: users[base][computed]})] if base in users else []
        elif base.endswith("DC=firma,DC=local"):
            hits = [(dn, {k: v for k, v in attrs.items() if not k.startswith("msDS-")})
                    for dn, attrs in users.items()
                    if f"(samaccountname={attrs['sAMAccountName'].lower()})" in search_filter.lower()
                    or f"(userPrincipalName={attrs['userPrincipalName']})" in search_filter]
        # Das AD hängt an Suchen ab der Domänenwurzel Verweise an, die keine Einträge sind.
        self.response = [{"type": "searchResEntry", "dn": dn, "attributes": {k: [v] for k, v in a.items()}}
                         for dn, a in hits] + [{"type": "searchResRef", "uri": ["ldap://DomainDnsZones.firma.local"]}]
        self.result = {"result": 0, "description": "success"}
        return bool(hits)

    def unbind(self):
        pass


@pytest.fixture
def ad(monkeypatch):
    for name, value in {"AD_ENABLED": True, "AD_SERVER": "ldaps://dc01.firma.local", "AD_DOMAIN": "firma.local",
                        "AD_BASE_DN": "", "AD_ALLOWED_GROUPS": [], "AD_ADMIN_GROUPS": [], "AD_CA_FILE": "",
                        "AD_BIND_USER": "svc-chatbot", "AD_BIND_PASSWORD": "dienst", "AD_BIND_PASSWORD_FILE": "",
                        "AD_CHECK_SECONDS": 0}.items():
        monkeypatch.setattr(config, name, value)
    ad_auth.reset()
    fake = FakeAD()
    monkeypatch.setattr(ad_auth, "_bind", fake.bind)
    yield fake
    ad_auth.reset()


# === Anmeldung ===

@pytest.mark.parametrize("login", ["mmuster", "MMuster", "FIRMA\\mmuster", "mmuster@firma.local", "m.muster@firma.de"])
def test_login_forms_give_the_same_user(ad, login):
    user = ad_auth.authenticate(f"  {login} ", "geheim")
    assert user == ad_auth.ADUser(username="mmuster", display_name="Max Muster", email="m.muster@firma.de")


def test_short_name_and_domain_prefix_bind_with_the_ad_domain(ad):
    ad_auth.authenticate("FIRMA\\mmuster", "geheim")
    ad_auth.authenticate("mmuster", "geheim")
    assert ad.binds == ["mmuster@firma.local", "mmuster@firma.local"]


def test_wrong_and_empty_passwords_are_rejected(ad):
    assert ad_auth.authenticate("mmuster", "falsch") is None
    # Ein leeres Passwort wäre eine anonyme Anmeldung, die das AD annimmt: es darf gar nicht erst hingehen.
    assert ad_auth.authenticate("mmuster", "") is None
    assert ad_auth.authenticate("", "geheim") is None
    assert ad.binds == ["mmuster@firma.local"]


def test_filter_characters_in_names_never_reach_the_ad(ad):
    for name in ("*", "mm*", "mmuster)(sAMAccountName=*", "a=b", "x\x00y"):
        assert ad_auth.authenticate(name, "geheim") is None
    assert ad.binds == []
    assert ad_auth.escape("a*b(c)\\d") == "a\\2ab\\28c\\29\\5cd"


def test_too_many_failures_lock_the_name_without_asking_the_ad(ad):
    for login in ("mmuster", "FIRMA\\mmuster", "mmuster@firma.local", "MMUSTER", "mmuster"):
        assert ad_auth.authenticate(login, "falsch") is None
    asked = len(ad.binds)
    assert ad_auth.authenticate("mmuster", "geheim") is None  # gesperrt, obwohl das Passwort stimmt
    assert len(ad.binds) == asked
    assert ad_auth.authenticate("ebeispiel", "passwort").username == "ebeispiel"  # andere Nutzer betrifft das nicht
    ad_auth.reset()
    assert ad_auth.authenticate("mmuster", "geheim").username == "mmuster"


def test_success_resets_the_failure_count(ad):
    for _ in range(ad_auth.MAX_FAILURES - 1):
        ad_auth.authenticate("mmuster", "falsch")
    assert ad_auth.authenticate("mmuster", "geheim") is not None
    assert ad_auth.authenticate("mmuster", "falsch") is None
    assert ad_auth.authenticate("mmuster", "geheim") is not None


@pytest.mark.parametrize("group", ["Chatbot-Nutzer", GROUP_DN])
def test_allowed_groups_by_name_or_dn(ad, monkeypatch, group):
    monkeypatch.setattr(config, "AD_ALLOWED_GROUPS", [group])
    assert ad_auth.authenticate("mmuster", "geheim").username == "mmuster"
    assert ad_auth.authenticate("ebeispiel", "passwort") is None


def test_admins_are_marked_and_always_allowed(ad, monkeypatch):
    monkeypatch.setattr(config, "AD_ALLOWED_GROUPS", ["Chatbot-Nutzer"])
    monkeypatch.setattr(config, "AD_ADMIN_GROUPS", ["Chatbot-Admins"])
    admin = ad_auth.authenticate("aadmin", "admin")  # nicht in Chatbot-Nutzer, aber Admin
    assert admin.admin and admin.username == "aadmin"
    assert ad_auth.authenticate("mmuster", "geheim").admin is False
    assert ad_auth.authenticate("ebeispiel", "passwort") is None


def test_unknown_group_lets_nobody_in_and_is_logged(ad, monkeypatch, caplog):
    monkeypatch.setattr(config, "AD_ALLOWED_GROUPS", ["Gibt-es-nicht"])
    with caplog.at_level(logging.WARNING):
        assert ad_auth.authenticate("mmuster", "geheim") is None
    assert "Gibt-es-nicht" in caplog.text


def test_user_outside_the_base_dn_is_a_configuration_error(ad, monkeypatch):
    monkeypatch.setattr(config, "AD_BASE_DN", "OU=Woanders,DC=andere,DC=domaene")
    with pytest.raises(ad_auth.ADError, match="AD_BASE_DN"):
        ad_auth.authenticate("mmuster", "geheim")


def test_base_dn_from_domain_or_setting(ad, monkeypatch):
    assert ad_auth.base_dn() == "DC=firma,DC=local"
    monkeypatch.setattr(config, "AD_BASE_DN", "OU=Mitarbeiter,DC=firma,DC=local")
    assert ad_auth.base_dn() == "OU=Mitarbeiter,DC=firma,DC=local"


def test_check_config(ad, monkeypatch, tmp_path):
    pytest.importorskip("ldap3")
    ad_auth.check_config()
    monkeypatch.setattr(config, "AD_SERVER", "ldaps://dc01.firma.local, ldap://dc02.firma.local")
    ad_auth.check_config()
    for changes, message in [({"AD_SERVER": ""}, "AD_SERVER fehlt"), ({"AD_SERVER": "dc01.firma.local"}, "ldaps://"),
                             ({"AD_DOMAIN": ""}, "AD_DOMAIN fehlt"),
                             ({"AD_CA_FILE": str(tmp_path / "fehlt.pem")}, "AD_CA_FILE"),
                             ({"AD_BIND_USER": ""}, "AD_BIND_USER fehlt"),
                             ({"AD_BIND_PASSWORD": ""}, "AD_BIND_PASSWORD"),
                             ({"AD_BIND_PASSWORD_FILE": str(tmp_path / "fehlt.txt")}, "AD_BIND_PASSWORD_FILE")]:
        with monkeypatch.context() as m:
            for name, value in changes.items():
                m.setattr(config, name, value)
            with pytest.raises(ad_auth.ADError, match=message):
                ad_auth.check_config()


# === Laufende Prüfung nach der Anmeldung ===

def test_active_account_has_access_and_admins_are_recognised(ad, monkeypatch):
    monkeypatch.setattr(config, "AD_ADMIN_GROUPS", ["Chatbot-Admins"])
    assert ad_auth.check_access("mmuster") == ad_auth.Access(True, admin=False)
    assert ad_auth.check_access("aadmin") == ad_auth.Access(True, admin=True)
    assert ad.service_binds() == ["svc-chatbot@firma.local"]  # eine Verbindung des Dienstkontos für alle Abfragen


def account_expires_in(seconds):
    return str(int((time.time() + seconds + ad_auth._FILETIME_TO_UNIX) * 10**7))


@pytest.mark.parametrize("change, reason", [
    ({"userAccountControl": "514"}, "Konto deaktiviert"),  # 512 + ACCOUNTDISABLE
    ({"msDS-User-Account-Control-Computed": "16"}, "Konto gesperrt"),  # LOCKOUT, z. B. zu viele falsche Passwörter
    ({"accountExpires": account_expires_in(-60)}, "Konto abgelaufen"),
])
def test_locked_disabled_or_expired_accounts_lose_access_at_once(ad, change, reason):
    assert ad_auth.check_access("mmuster").allowed
    ad.users[USER_DN].update(change)
    assert ad_auth.check_access("mmuster") == ad_auth.Access(False, reason=reason)
    ad.users[USER_DN].update(ACTIVE)  # wieder entsperrt
    assert ad_auth.check_access("mmuster").allowed


def test_account_that_expires_later_keeps_access(ad):
    ad.users[USER_DN]["accountExpires"] = account_expires_in(3600)
    assert ad_auth.check_access("mmuster").allowed


def test_deleted_account_and_removed_group_membership_lose_access(ad, monkeypatch):
    monkeypatch.setattr(config, "AD_ALLOWED_GROUPS", ["Chatbot-Nutzer"])
    assert ad_auth.check_access("mmuster").allowed
    ad.members[USER_DN].clear()
    assert ad_auth.check_access("mmuster").reason == "in keiner Gruppe aus AD_ALLOWED_GROUPS"
    del ad.users[USER_DN]
    assert ad_auth.check_access("mmuster").reason == "Konto nicht im AD gefunden"


def test_lost_admin_rights_are_noticed(ad, monkeypatch):
    monkeypatch.setattr(config, "AD_ADMIN_GROUPS", ["Chatbot-Admins"])
    assert ad_auth.check_access("aadmin").admin
    ad.members[ADMIN_DN].clear()
    assert ad_auth.check_access("aadmin") == ad_auth.Access(True, admin=False)


def test_result_is_reused_for_ad_check_seconds(ad, monkeypatch):
    monkeypatch.setattr(config, "AD_CHECK_SECONDS", 10)
    assert ad_auth.check_access("mmuster").allowed
    searches = ad.searches
    ad.users[USER_DN]["userAccountControl"] = "514"
    assert ad_auth.check_access("mmuster").allowed and ad.searches == searches  # noch das Ergebnis von eben
    monkeypatch.setattr(config, "AD_CHECK_SECONDS", 0)
    assert not ad_auth.check_access("mmuster").allowed


def test_login_counts_as_fresh_check(ad, monkeypatch):
    monkeypatch.setattr(config, "AD_CHECK_SECONDS", 10)
    ad_auth.authenticate("mmuster", "geheim")
    assert ad_auth.check_access("mmuster").allowed
    assert ad.service_binds() == []


def test_dropped_service_connection_is_rebuilt_once(ad):
    assert ad_auth.check_access("mmuster").allowed
    ad.drop_connection = True
    assert ad_auth.check_access("mmuster").allowed
    assert len(ad.service_binds()) == 2


def test_unreachable_ad_means_no_access(ad, caplog):
    ad.unreachable = True
    with caplog.at_level(logging.ERROR):
        assert ad_auth.check_access("mmuster") == ad_auth.Access(False, reason=ad_auth.AD_UNREACHABLE)
    assert "nicht prüfbar" in caplog.text
    ad.unreachable = False
    assert ad_auth.check_access("mmuster").allowed


def test_rejected_service_account_is_not_retried_at_once(ad, monkeypatch):
    """Sonst würde der Chatbot das Dienstkonto mit jeder Anfrage weiter ins Sperrlimit des AD treiben."""
    monkeypatch.setattr(config, "AD_BIND_PASSWORD", "veraltet")
    for _ in range(5):
        assert ad_auth.check_access("mmuster").reason == ad_auth.AD_UNREACHABLE
    assert len(ad.service_binds()) == 1
    monkeypatch.setattr(ad_auth, "_service_retry_at", 0.0)  # Pause vorbei, Passwort korrigiert
    monkeypatch.setattr(config, "AD_BIND_PASSWORD", "dienst")
    assert ad_auth.check_access("mmuster").allowed


def test_service_password_from_file(ad, monkeypatch, tmp_path):
    secret = tmp_path / "dienstkonto.txt"
    secret.write_text("dienst\n")
    monkeypatch.setattr(config, "AD_BIND_PASSWORD", "")
    monkeypatch.setattr(config, "AD_BIND_PASSWORD_FILE", str(secret))
    assert ad_auth.check_access("mmuster").allowed


# === Verbindung zum Domänencontroller (ldap3) ===

class RecordingConnection:
    """Ersetzt ldap3.Connection: merkt sich, was ad_auth mit der Verbindung macht."""

    calls: list = []
    unreachable: set = set()
    passwords = {"mmuster@firma.local": "geheim"}

    def __init__(self, server, user, password, **kwargs):
        self.server, self.user, self.password, self.kwargs = server, user, password, kwargs
        self.result = None

    def open(self):
        from ldap3.core.exceptions import LDAPSocketOpenError

        self.calls.append(("open", self.server.host))
        if self.server.host in self.unreachable:
            raise LDAPSocketOpenError("socket connection error")

    def start_tls(self):
        self.calls.append(("start_tls", self.server.host))
        return True

    def bind(self):
        self.calls.append(("bind", self.server.host))
        if self.passwords.get(self.user) == self.password:
            self.result = {"result": 0}
            return True
        self.result = {"result": 49, "description": "invalidCredentials",
                       "message": "80090308: LdapErr: DSID-0C09044E, comment: AcceptSecurityContext error, data 52e"}
        return False

    def unbind(self):
        self.calls.append(("unbind", self.server.host))


@pytest.fixture
def recording(monkeypatch):
    ldap3 = pytest.importorskip("ldap3")
    monkeypatch.setattr(ldap3, "Connection", RecordingConnection)
    monkeypatch.setattr(config, "AD_DOMAIN", "firma.local")
    monkeypatch.setattr(config, "AD_CA_FILE", "")
    RecordingConnection.calls = []
    RecordingConnection.unreachable = set()
    return RecordingConnection


def test_bind_encrypts_before_sending_the_password(recording, monkeypatch):
    monkeypatch.setattr(config, "AD_SERVER", "ldap://dc01.firma.local")
    connection = ad_auth._bind("mmuster@firma.local", "geheim")
    assert connection.kwargs["auto_referrals"] is False
    assert recording.calls == [("open", "dc01.firma.local"), ("start_tls", "dc01.firma.local"),
                               ("bind", "dc01.firma.local")]

    recording.calls = []
    monkeypatch.setattr(config, "AD_SERVER", "ldaps://dc01.firma.local")  # schon verschlüsselt, kein StartTLS
    assert ad_auth._bind("mmuster@firma.local", "geheim").server.ssl
    assert [c[0] for c in recording.calls] == ["open", "bind"]


def test_bind_wrong_password_and_failover(recording, monkeypatch, caplog):
    monkeypatch.setattr(config, "AD_SERVER", "ldaps://dc01.firma.local,ldaps://dc02.firma.local")
    recording.unreachable = {"dc01.firma.local"}
    with caplog.at_level(logging.INFO):
        assert ad_auth._bind("mmuster@firma.local", "Sommer2026!") is None
    assert "falsches Passwort" in caplog.text and "Sommer2026!" not in caplog.text  # Grund ja, Passwort nie
    assert ("bind", "dc02.firma.local") in recording.calls  # erster Server nicht erreichbar, zweiter gefragt

    recording.unreachable = {"dc01.firma.local", "dc02.firma.local"}
    with pytest.raises(ad_auth.ADError, match="dc02.firma.local"):
        ad_auth._bind("mmuster@firma.local", "geheim")


# === API und Oberfläche ===

def test_api_requires_login_from_an_active_account(workspace, monkeypatch):
    from rag_chatpipline import api_server  # zuerst: legt fest, wo Chainlit seine Konfiguration sucht

    from chainlit.auth import create_jwt
    from chainlit.user import User
    from fastapi.testclient import TestClient

    client = TestClient(api_server.app)
    assert client.get("/files/fehlt.pdf").status_code == 404  # ohne AD: offen wie bisher

    monkeypatch.setattr(config, "AD_ENABLED", True)
    access = {"mmuster": ad_auth.Access(True)}
    monkeypatch.setattr(ad_auth, "check_access", lambda name: access[name])
    for path in ("/files/fehlt.pdf", "/documents", "/images/x.png", "/videos/x.mp4", "/docs"):
        assert client.get(path).status_code == 401, path
    assert client.post("/chat", json={"message": "Hallo"}).status_code == 401
    assert client.get("/health").status_code == 200
    assert client.get("/files/fehlt.pdf", headers={"Authorization": "Bearer kaputt"}).status_code == 401

    token = create_jwt(User(identifier="mmuster"))
    assert client.get("/files/fehlt.pdf", headers={"Authorization": f"Bearer {token}"}).status_code == 404
    client.cookies.set("access_token", token)
    assert client.get("/files/fehlt.pdf").status_code == 404

    access["mmuster"] = ad_auth.Access(False, reason="Konto gesperrt")  # im AD gesperrt, das Cookie gilt noch
    reply = client.get("/files/fehlt.pdf")
    assert reply.status_code == 401
    assert "access_token=;" in reply.headers["set-cookie"] and "Max-Age=0" in reply.headers["set-cookie"]


# Läuft in einem eigenen Prozess: Welche Anmeldung die Oberfläche nutzt, legt Chainlit beim Import fest.
UI_LOGIN_SCRIPT = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from rag_chatpipline import config
from pathlib import Path
config.MEMORY_DIR = Path(sys.argv[2])  # Chat-Datenbank, Geheimnis und Vektordatenbank nur im Testordner
config.QDRANT_PATH = config.MEMORY_DIR / "qdrant"  # wird beim Import fest berechnet, daher eigens umlenken
from rag_chatpipline import ad_auth
ad_auth.authenticate = lambda u, p: ad_auth.ADUser("mmuster", "Max Muster") if (u, p) == ("mmuster", "geheim") else None
state = {"access": ad_auth.Access(True)}
ad_auth.check_access = lambda name: state["access"]
from fastapi.testclient import TestClient
from rag_chatpipline import api_server

client = TestClient(api_server.app)
result = {"config": client.get("/ui/auth/config").json(), "before": client.get("/files/fehlt.pdf").status_code,
          "wrong": client.post("/ui/login", data={"username": "mmuster", "password": "falsch"}).status_code}
login = client.post("/ui/login", data={"username": "mmuster", "password": "geheim"})
result.update(login=login.status_code, after=client.get("/files/fehlt.pdf").status_code,
              user=client.get("/ui/user").json().get("identifier"))
state["access"] = ad_auth.Access(False, reason="Konto gesperrt")
locked = client.get("/ui/user")
result.update(locked_user=locked.status_code, locked_cookie=locked.headers.get("set-cookie", ""),
              locked_files=client.get("/files/fehlt.pdf").status_code)
# Der Browser hat das alte Cookie noch (z. B. Antwort nicht angekommen), das Konto wird entsperrt, und die Prüfung
# meldet für einen Moment noch "gesperrt": Die neue Anmeldung darf nicht gleich wieder gelöscht werden.
client.cookies.set("access_token", login.cookies["access_token"])
relogin = client.post("/ui/login", data={"username": "mmuster", "password": "geheim"})
result.update(relogin_cookies=relogin.headers.get_list("set-cookie"))
print(json.dumps(result))
"""


def test_ui_login_page_and_locked_accounts(tmp_path):
    pytest.importorskip("ldap3")
    root = Path(__file__).resolve().parents[1]
    env = {"AD_ENABLED": "true", "AD_SERVER": "ldaps://dc01.firma.local", "AD_DOMAIN": "firma.local",
           "AD_BIND_USER": "svc-chatbot", "AD_BIND_PASSWORD": "dienst",
           "WORKSPACE_DIR": str(tmp_path / "workspace"), "PATH": "", "HOME": str(tmp_path)}
    run = subprocess.run([sys.executable, "-c", UI_LOGIN_SCRIPT, str(root / "src"), str(tmp_path / "memory")],
                         capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120)
    assert run.returncode == 0, run.stderr[-3000:]
    result = json.loads(run.stdout.strip().splitlines()[-1])
    assert result["config"]["passwordAuth"] is True and result["config"]["headerAuth"] is False
    assert result["before"] == 401 and result["wrong"] == 401
    assert result["login"] == 200 and result["after"] == 404 and result["user"] == "mmuster"
    # Konto im AD gesperrt: Die Oberfläche gilt sofort als abgemeldet, das Cookie wird gelöscht.
    assert result["locked_user"] == 401 and "Max-Age=0" in result["locked_cookie"] and result["locked_files"] == 401
    assert [c for c in result["relogin_cookies"] if c.startswith("access_token=")] != []
    assert not any(c.startswith("access_token=;") for c in result["relogin_cookies"])


def test_only_admins_delete_or_replace_documents(workspace, vector_store, monkeypatch, tmp_path):
    from rag_chatpipline import api_server

    from chainlit.auth import create_jwt
    from chainlit.user import User
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api_server.knowledge_base, "_vector_store", vector_store)
    client = TestClient(api_server.app)

    def upload(name, text, token=None):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return client.post("/documents", files={"files": (name, text.encode(), "text/plain")}, headers=headers)

    def delete(name, token=None):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return client.delete(f"/documents/{name}", headers=headers)

    # ohne AD: jeder darf alles, wie bisher
    assert upload("a.txt", "Erste Fassung.").status_code == 200
    assert upload("a.txt", "Zweite Fassung.").status_code == 200
    assert delete("a.txt").status_code == 204

    monkeypatch.setattr(config, "AD_ENABLED", True)
    roles = {"mmuster": ad_auth.Access(True, admin=False), "aadmin": ad_auth.Access(True, admin=True)}
    monkeypatch.setattr(ad_auth, "check_access", lambda name: roles[name])
    user, admin = create_jwt(User(identifier="mmuster")), create_jwt(User(identifier="aadmin"))

    assert upload("richtlinie.txt", "Original.", user).status_code == 200  # hochladen darf jeder
    replaced = upload("richtlinie.txt", "Überschrieben.", user)
    assert replaced.status_code == 409 and "nur Admins" in replaced.json()["detail"]
    assert delete("richtlinie.txt", user).status_code == 403
    assert (config.DOCUMENTS_DIR / "richtlinie.txt").read_text() == "Original."

    assert upload("richtlinie.txt", "Neue Fassung.", admin).status_code == 200
    assert (config.DOCUMENTS_DIR / "richtlinie.txt").read_text() == "Neue Fassung."
    assert delete("richtlinie.txt", admin).status_code == 204
    assert not (config.DOCUMENTS_DIR / "richtlinie.txt").exists()
