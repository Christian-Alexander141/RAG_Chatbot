"""Anmeldung mit Microsoft Active Directory (optional, Standard: aus).

Eingeschaltet mit AD_ENABLED=true. Die Oberfläche zeigt dann eine Anmeldeseite, Benutzername und Passwort werden per
LDAP am Domänencontroller geprüft (Bind als der Nutzer selbst). Der Server des Chatbots muss nicht Mitglied der Domäne
sein und kann unter Linux oder Windows laufen. Ohne AD (z. B. in einer reinen Linux-Umgebung) bleibt AD_ENABLED aus
und alles funktioniert wie ohne Anmeldung.

- Passwörter gehen nur verschlüsselt zum Domänencontroller: ldaps:// oder ldap:// mit StartTLS, das Zertifikat wird
  geprüft (AD_CA_FILE für die Zertifizierungsstelle der Firma).
- Nach der Anmeldung fragt check_access() über ein Dienstkonto laufend nach, ob das Konto noch Zugang hat (nicht
  gesperrt, deaktiviert, abgelaufen oder gelöscht, noch in einer erlaubten Gruppe) und ob es Admin ist. Ein Ergebnis
  gilt AD_CHECK_SECONDS Sekunden. Ist das AD nicht erreichbar, gibt es keinen Zugang.
- AD_ALLOWED_GROUPS beschränkt den Zugang auf Mitglieder bestimmter Gruppen, auch über verschachtelte Gruppen.
  Mitglieder von AD_ADMIN_GROUPS sind Admins und haben immer Zugang.
- Nach MAX_FAILURES Fehlversuchen wird ein Benutzername eine Weile abgewiesen, ohne das AD zu fragen. So lässt sich
  ein Konto nicht über den Chatbot durch Ausprobieren im AD sperren.
"""
from __future__ import annotations

import getpass
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import config

logger = logging.getLogger(__name__)

MAX_FAILURES = 5
LOCK_SECONDS = 15 * 60
# Lehnt das AD das Dienstkonto ab, erst nach dieser Pause erneut versuchen, sonst sperrt das AD es selbst.
SERVICE_RETRY_SECONDS = 60
# LDAP_MATCHING_RULE_IN_CHAIN: Mitgliedschaft auch über verschachtelte Gruppen
_IN_CHAIN = "1.2.840.113556.1.4.1941"
_INVALID_CREDENTIALS = 49
_SEARCH_OK = (0, 4, 32)  # Erfolg, Größenlimit erreicht, Objekt nicht vorhanden
_UF_ACCOUNTDISABLE = 0x2  # in userAccountControl
_UF_LOCKOUT = 0x10  # in msDS-User-Account-Control-Computed
_NEVER_EXPIRES = (0, 0x7FFFFFFFFFFFFFFF)  # accountExpires
_FILETIME_TO_UNIX = 11644473600  # Sekunden von 1601 (Zeitrechnung des AD) bis 1970
# Zeichen, die in AD-Anmeldenamen nicht vorkommen (sAMAccountName), dazu Steuerzeichen
_FORBIDDEN = re.compile(r'[\x00-\x1f"/\[\]:;|=,+*?<>()]')
# Unterstatus von AD bei Fehler 49 (steht in der Meldung als "data 52e" usw.), nur für das Log
_AD_REASONS = {
    "525": "Benutzer nicht gefunden",
    "52e": "falsches Passwort",
    "530": "Anmeldung zu dieser Zeit nicht erlaubt",
    "531": "Anmeldung an diesem Rechner nicht erlaubt",
    "532": "Passwort abgelaufen",
    "533": "Konto deaktiviert",
    "701": "Konto abgelaufen",
    "773": "Passwort muss geändert werden",
    "775": "Konto im AD gesperrt",
}
AD_UNREACHABLE = "AD nicht erreichbar"


class ADError(RuntimeError):
    """AD nicht erreichbar oder falsch eingerichtet (nicht: falsches Passwort)."""


@dataclass
class ADUser:
    username: str  # sAMAccountName in Kleinbuchstaben: Kennung für Chats und Einstellungen
    display_name: str
    email: str = ""
    admin: bool = False


@dataclass(frozen=True)
class Access:
    allowed: bool
    admin: bool = False
    reason: str = ""  # warum kein Zugang


_lock = threading.Lock()
_failures: dict[str, list[float]] = {}
_access_cache: dict[str, tuple[float, Access]] = {}
# Verbindung des Dienstkontos: wird wiederverwendet, eine Abfrage zur Zeit
_service_lock = threading.Lock()
_service: Optional[Any] = None
_service_retry_at = 0.0


def enabled() -> bool:
    return config.AD_ENABLED


def admin_rights(admin: Optional[bool]) -> bool:
    """Embedding-Modell wählen, Dokumente löschen oder ersetzen: mit AD nur Admins, ohne AD jeder (wie bisher)."""
    return not enabled() or bool(admin)


REPLACE_DENIED = "Ersetzen und Löschen dürfen nur Admins, bitte die Datei umbenennen."


def base_dn() -> str:
    return config.AD_BASE_DN or ",".join(f"DC={part}" for part in config.AD_DOMAIN.split(".") if part)


def server_urls() -> list[str]:
    return [url.strip() for url in config.AD_SERVER.split(",") if url.strip()]


def check_config() -> None:
    """Bricht mit einer verständlichen Meldung ab, wenn die AD-Anmeldung nicht funktionieren kann."""
    try:
        import ldap3  # noqa: F401
    except ImportError as exc:
        raise ADError("Das Paket ldap3 fehlt. Installation mit ./install.sh --ad (Windows: install.bat --ad)") from exc
    if not server_urls():
        raise ADError("AD_SERVER fehlt, z. B. AD_SERVER=ldaps://dc01.firma.local")
    for url in server_urls():
        if not url.lower().startswith(("ldaps://", "ldap://")):
            raise ADError(f"AD_SERVER '{url}' muss mit ldaps:// oder ldap:// beginnen (ldap:// nutzt StartTLS)")
    if not config.AD_DOMAIN:
        raise ADError("AD_DOMAIN fehlt, z. B. AD_DOMAIN=firma.local")
    if config.AD_CA_FILE and not Path(config.AD_CA_FILE).is_file():
        raise ADError(f"AD_CA_FILE nicht gefunden: {config.AD_CA_FILE}")
    if not config.AD_BIND_USER:
        raise ADError("AD_BIND_USER fehlt: Dienstkonto, mit dem der Chatbot prüft, ob Konten noch Zugang haben")
    if config.AD_BIND_PASSWORD_FILE and not Path(config.AD_BIND_PASSWORD_FILE).is_file():
        raise ADError(f"AD_BIND_PASSWORD_FILE nicht gefunden: {config.AD_BIND_PASSWORD_FILE}")
    if not config.AD_BIND_PASSWORD and not config.AD_BIND_PASSWORD_FILE:
        raise ADError("Passwort des Dienstkontos fehlt: AD_BIND_PASSWORD oder AD_BIND_PASSWORD_FILE setzen")


def escape(value: str) -> str:
    """Maskiert einen Wert für einen LDAP-Suchfilter (RFC 4515)."""
    return "".join(f"\\{ord(c):02x}" if c in '\\*()\x00' else c for c in value)


def _short_name(username: str) -> str:
    """FIRMA\\mmuster, mmuster@firma.local und mmuster ergeben alle "mmuster" (für die Fehlversuch-Sperre)."""
    return username.split("\\")[-1].split("@")[0].lower()


def _user_filter(match: str) -> str:
    return f"(&(objectCategory=person)(objectClass=user){match})"


def login_names(username: str) -> tuple[str, str]:
    """Name für den LDAP-Bind und Suchfilter, der den Nutzer danach findet.

    Erlaubt sind "mmuster", "FIRMA\\mmuster" und "mmuster@firma.local" (UPN, auch mit anderem UPN-Suffix).
    """
    name = username.split("\\")[-1]
    if "@" in name:
        local, _, suffix = name.partition("@")
        match = f"(userPrincipalName={escape(name)})"
        if suffix.lower() == config.AD_DOMAIN.lower():  # AD erlaubt name@domäne auch, wenn der UPN anders lautet
            match = f"(|{match}(sAMAccountName={escape(local)}))"
        bind = name
    else:
        match = f"(sAMAccountName={escape(name)})"
        bind = f"{name}@{config.AD_DOMAIN}"
    return bind, _user_filter(match)


def _locked(key: str) -> bool:
    with _lock:
        recent = [t for t in _failures.get(key, []) if time.monotonic() - t < LOCK_SECONDS]
        _failures[key] = recent
        return len(recent) >= MAX_FAILURES


def _record(key: str, success: bool) -> None:
    with _lock:
        if success:
            _failures.pop(key, None)
        else:
            _failures.setdefault(key, []).append(time.monotonic())


def reset() -> None:
    """Vergisst Fehlversuche, geprüfte Konten und die Verbindung des Dienstkontos (für Tests)."""
    global _service_retry_at
    with _lock:
        _failures.clear()
        _access_cache.clear()
    with _service_lock:
        _drop_service()
        _service_retry_at = 0.0


def _bind(bind_name: str, password: str) -> Optional[Any]:
    """Meldet sich am ersten erreichbaren Domänencontroller an.

    Gibt die Verbindung zurück, None bei falschen Anmeldedaten, und wirft ADError, wenn kein Server erreichbar ist.
    """
    import ssl

    from ldap3 import NONE, SIMPLE, Connection, Server, Tls
    from ldap3.core.exceptions import LDAPException

    tls = Tls(validate=ssl.CERT_REQUIRED, ca_certs_file=config.AD_CA_FILE or None)
    errors = []
    for url in server_urls():
        server = Server(url, tls=tls, get_info=NONE, connect_timeout=config.AD_TIMEOUT)
        # auto_referrals aus: Suchen ab der Domänenwurzel liefern Verweise auf DomainDnsZones usw., denen ldap3 sonst
        # mit eigenen Verbindungen folgt (langsam, und das Passwort ginge an weitere Server).
        connection = Connection(server, user=bind_name, password=password, authentication=SIMPLE, read_only=True,
                                receive_timeout=config.AD_TIMEOUT, auto_referrals=False)
        try:
            connection.open()
            # Ohne ldaps:// erst verschlüsseln, das Passwort geht nie im Klartext über das Netz.
            if not server.ssl and not connection.start_tls():
                raise ADError(f"StartTLS abgelehnt: {connection.result}")
            if connection.bind():
                return connection
        except (LDAPException, OSError, ADError) as exc:
            errors.append(f"{url}: {exc}")
            connection.unbind()
            continue
        result = connection.result or {}
        connection.unbind()
        if result.get("result") == _INVALID_CREDENTIALS:
            code = re.search(r"data ([0-9a-f]{3})", str(result.get("message", "")))
            reason = _AD_REASONS.get(code.group(1), "") if code else ""
            logger.info("AD-Anmeldung abgelehnt für %s%s", bind_name, f": {reason}" if reason else "")
            return None
        errors.append(f"{url}: {result.get('description')} {result.get('message', '')}".strip())
    raise ADError("Kein Domänencontroller hat die Anmeldung angenommen: " + "; ".join(errors))


def _search(connection: Any, base: str, search_filter: str, attributes: list[str], scope: str = "SUBTREE",
            size_limit: int = 0) -> list[dict]:
    """Einträge der Suche, ohne Verweise. ADError, wenn das AD einen Fehler meldet (statt "nichts gefunden")."""
    connection.search(base, search_filter, search_scope=scope, attributes=attributes, size_limit=size_limit)
    result = connection.result or {}
    if result.get("result", 0) not in _SEARCH_OK:
        raise ADError(f"Suche im AD fehlgeschlagen: {result.get('description')} {result.get('message', '')}".strip())
    return [r for r in (connection.response or []) if r.get("type") == "searchResEntry"]


def _first(attributes: dict, key: str) -> str:
    value = attributes.get(key)
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "")


def _number(attributes: dict, key: str) -> int:
    try:
        return int(_first(attributes, key) or 0)
    except ValueError:
        return 0


def _group_dn(connection: Any, group: str) -> Optional[str]:
    """Gruppenname (cn oder sAMAccountName) zu DN. Ein DN (enthält "=") bleibt, wie er ist."""
    if "=" in group:
        return group
    found = _search(connection, base_dn(),
                    f"(&(objectClass=group)(|(cn={escape(group)})(sAMAccountName={escape(group)})))", [], size_limit=1)
    if not found:
        logger.warning("AD-Gruppe '%s' nicht gefunden (unter %s)", group, base_dn())
        return None
    return found[0]["dn"]


def _member_of_any(connection: Any, user_dn: str, groups: list[str]) -> bool:
    for group in groups:
        dn = _group_dn(connection, group)
        if dn and _search(connection, user_dn, f"(memberOf:{_IN_CHAIN}:={escape(dn)})", [], scope="BASE"):
            return True
    return False


def _groups_access(connection: Any, user_dn: str) -> Access:
    """Admin? Und ohne Admin-Rechte: in einer erlaubten Gruppe?"""
    admin = _member_of_any(connection, user_dn, config.AD_ADMIN_GROUPS)
    if config.AD_ALLOWED_GROUPS and not admin and not _member_of_any(connection, user_dn, config.AD_ALLOWED_GROUPS):
        return Access(False, reason="in keiner Gruppe aus AD_ALLOWED_GROUPS")
    return Access(True, admin=admin)


def _remember(username: str, access: Access) -> None:
    with _lock:
        previous = _access_cache.get(username)
        _access_cache[username] = (time.monotonic(), access)
    if not access.allowed and (previous is None or previous[1] != access):
        logger.warning("Kein Zugang für %s: %s", username, access.reason)


def authenticate(username: str, password: str) -> Optional[ADUser]:
    """Prüft Benutzername und Passwort am AD.

    None, wenn die Anmeldung abgelehnt wird (falsches Passwort, gesperrtes Konto, nicht in einer erlaubten Gruppe,
    zu viele Fehlversuche). ADError, wenn das AD nicht erreichbar oder falsch eingerichtet ist.
    """
    username = (username or "").strip()
    # Ein leeres Passwort wäre für LDAP eine anonyme Anmeldung, die "gelingt": nie ans AD schicken.
    if not username or not password or len(username) > 256 or _FORBIDDEN.search(username.split("\\")[-1]):
        return None
    key = _short_name(username)
    if _locked(key):
        logger.warning("AD-Anmeldung für %s abgewiesen: %d Fehlversuche in %d Minuten", key, MAX_FAILURES,
                       LOCK_SECONDS // 60)
        return None

    bind_name, user_filter = login_names(username)
    connection = _bind(bind_name, password)
    if connection is None:
        _record(key, success=False)
        return None
    try:
        found = _search(connection, base_dn(), user_filter, ["sAMAccountName", "displayName", "mail"], size_limit=2)
        if len(found) != 1:
            raise ADError(f"Anmeldung von {bind_name} gelang, aber unter {base_dn()} wurde der Nutzer "
                          f"{'nicht' if not found else 'mehrfach'} gefunden. AD_BASE_DN prüfen.")
        entry = found[0]
        access = _groups_access(connection, entry["dn"])
    finally:
        connection.unbind()
    if not access.allowed:
        logger.info("AD-Anmeldung abgelehnt für %s: %s", bind_name, access.reason)
        _record(key, success=False)
        return None

    _record(key, success=True)
    attributes = entry.get("attributes") or {}
    account = (_first(attributes, "sAMAccountName") or key).lower()
    _remember(account, access)  # gerade geprüft, die erste Prüfung danach braucht das AD nicht
    return ADUser(username=account, display_name=_first(attributes, "displayName") or account,
                  email=_first(attributes, "mail"), admin=access.admin)


# === Laufende Prüfung nach der Anmeldung (Dienstkonto) ===

def _service_password() -> str:
    if config.AD_BIND_PASSWORD_FILE:
        return Path(config.AD_BIND_PASSWORD_FILE).read_text(encoding="utf-8").strip()
    return config.AD_BIND_PASSWORD


def _drop_service() -> None:
    global _service
    if _service is not None:
        try:
            _service.unbind()
        except Exception:
            pass
        _service = None


def _service_connection() -> Any:
    global _service, _service_retry_at
    if _service is not None:
        return _service
    if time.monotonic() < _service_retry_at:
        raise ADError("Das AD hat das Dienstkonto AD_BIND_USER eben abgelehnt, neuer Versuch in Kürze")
    name = config.AD_BIND_USER
    if not any(c in name for c in "@\\="):
        name = f"{name}@{config.AD_DOMAIN}"
    connection = _bind(name, _service_password())
    if connection is None:
        _service_retry_at = time.monotonic() + SERVICE_RETRY_SECONDS
        raise ADError("Das AD hat das Dienstkonto AD_BIND_USER abgelehnt (Name oder Passwort falsch, Konto gesperrt?)")
    _service = connection
    return connection


def _account_access(connection: Any, username: str) -> Access:
    found = _search(connection, base_dn(), _user_filter(f"(sAMAccountName={escape(username)})"),
                    ["userAccountControl", "accountExpires"], size_limit=2)
    if len(found) != 1:
        return Access(False, reason="Konto nicht im AD gefunden")
    dn, attributes = found[0]["dn"], found[0].get("attributes") or {}
    if _number(attributes, "userAccountControl") & _UF_ACCOUNTDISABLE:
        return Access(False, reason="Konto deaktiviert")
    expires = _number(attributes, "accountExpires")
    if expires not in _NEVER_EXPIRES and expires / 10**7 - _FILETIME_TO_UNIX < time.time():
        return Access(False, reason="Konto abgelaufen")
    # Gesperrt (zu viele falsche Passwörter oder von Hand): Nur dieses berechnete Attribut berücksichtigt das Ende
    # der Sperrdauer, lockoutTime bleibt danach gesetzt. Das AD liefert es nur bei einer Suche auf das Objekt selbst.
    computed = _search(connection, dn, "(objectClass=*)", ["msDS-User-Account-Control-Computed"], scope="BASE")
    if computed and _number(computed[0].get("attributes") or {}, "msDS-User-Account-Control-Computed") & _UF_LOCKOUT:
        return Access(False, reason="Konto gesperrt")
    return _groups_access(connection, dn)


def _query_access(username: str) -> Access:
    with _service_lock:
        for attempt in (1, 2):
            try:
                return _account_access(_service_connection(), username)
            except Exception as exc:  # jeder Fehler: neu verbinden, beim zweiten Mal kein Zugang (check_access)
                _drop_service()  # z. B. vom AD nach langem Leerlauf getrennt
                if attempt == 2:
                    if isinstance(exc, ADError):
                        raise
                    raise ADError(f"Abfrage im AD fehlgeschlagen: {exc}") from exc
    raise AssertionError("nicht erreichbar")


def check_access(username: str) -> Access:
    """Hat dieses Konto (noch) Zugang, und ist es Admin? Das Ergebnis gilt AD_CHECK_SECONDS Sekunden.

    Fragt das AD über das Dienstkonto. Ist es nicht erreichbar, gibt es keinen Zugang (lieber zu streng als zu offen).
    """
    with _lock:
        cached = _access_cache.get(username)
    if cached and time.monotonic() - cached[0] < config.AD_CHECK_SECONDS:
        return cached[1]
    try:
        access = _query_access(username)
    except ADError as exc:
        logger.error("Zugang für %s nicht prüfbar: %s", username, exc)
        access = Access(False, reason=AD_UNREACHABLE)
    _remember(username, access)
    return access


def cli(username: Optional[str]) -> int:
    """python main.py --mode ad: Einstellungen, eine Anmeldung und das Dienstkonto prüfen, ohne den Chatbot zu starten."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not enabled():
        print("Hinweis: AD_ENABLED ist aus. Die Oberfläche fragt erst nach einer Anmeldung, wenn AD_ENABLED=true ist.")
    try:
        check_config()
    except ADError as exc:
        print(f"Fehler: {exc}")
        return 2
    print(f"Domänencontroller: {', '.join(server_urls())}\nSuchbereich: {base_dn()}\nDienstkonto: {config.AD_BIND_USER}")
    for label, groups in (("Erlaubte Gruppen", config.AD_ALLOWED_GROUPS), ("Admin-Gruppen", config.AD_ADMIN_GROUPS)):
        if groups:
            print(f"{label}: {'; '.join(groups)}")
    username = username or input("Benutzername: ")
    try:
        user = authenticate(username, getpass.getpass("Passwort: "))
        if user is None:
            print("Anmeldung abgelehnt (Gründe stehen oben, falls das AD sie nennt).")
            return 1
        print(f"Anmeldung erfolgreich: {user.display_name} (Kennung im Chatbot: {user.username}"
              f"{', ' + user.email if user.email else ''}{', Admin' if user.admin else ''})")
        access = _query_access(user.username)
    except ADError as exc:
        print(f"Fehler: {exc}")
        return 2
    print("Prüfung über das Dienstkonto: " + (("Zugang" + (", Admin" if access.admin else "")) if access.allowed
                                              else f"kein Zugang ({access.reason})"))
    return 0 if access.allowed else 1
