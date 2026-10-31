// Fügt im Benutzermenü (Avatar oben rechts) den Punkt "Einstellungen" ein.
// Chainlit bietet dort keine eigenen Einträge an. Der Punkt öffnet den Einstellungsdialog des Chats,
// den Chainlit sonst nur über das Zahnrad neben dem Eingabefeld zeigt.
(function () {
  const ITEM_ID = "user-menu-settings";
  const GEAR =
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="ml-auto">' +
    '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73' +
    "l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38" +
    "a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18" +
    "a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08" +
    "a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08" +
    'a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>';

  function openSettings() {
    // Menü schließen, dann den Dialog über das (von Chainlit erzeugte) Zahnrad öffnen.
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    setTimeout(() => {
      const button =
        document.getElementById("chat-settings-open-modal") ||
        document.getElementById("chat-settings-header-button");
      if (button) button.click();
      else alert("Die Einstellungen sind verfügbar, sobald ein Chat geöffnet ist.");
    }, 50);
  }

  function addItem() {
    const trigger = document.getElementById("user-nav-button");
    if (!trigger || trigger.getAttribute("aria-expanded") !== "true") return;
    const menu = document.getElementById(trigger.getAttribute("aria-controls"));
    if (!menu || menu.querySelector("#" + ITEM_ID)) return;
    const items = menu.querySelectorAll('[role="menuitem"]');
    const logout = items[items.length - 1];
    if (!logout) return;

    const item = logout.cloneNode(false); // gleiches Aussehen wie "Abmelden", ohne dessen Klick-Logik
    item.id = ITEM_ID;
    item.innerHTML = "Einstellungen" + GEAR;
    item.addEventListener("click", openSettings);
    item.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && openSettings());
    item.addEventListener("pointerenter", () => item.setAttribute("data-highlighted", ""));
    item.addEventListener("pointerleave", () => item.removeAttribute("data-highlighted"));
    logout.before(item);
  }

  new MutationObserver(addItem).observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["aria-expanded"],
  });
})();
