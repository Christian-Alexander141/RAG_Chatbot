import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Trash2, Upload } from "lucide-react"

// Wissensquellen verwalten. Spricht direkt mit den Dokument-Routen der API (/documents, gleiche Origin).
export default function DocumentsPanel() {
  const [docs, setDocs] = useState([])
  const [status, setStatus] = useState("")
  const [busy, setBusy] = useState(false)
  const input = useRef(null)

  async function load() {
    try {
      const res = await fetch("/documents")
      if (!res.ok) throw new Error((await res.json()).detail)
      setDocs(await res.json())
    } catch (err) {
      setStatus("Wissensbasis nicht verfügbar: " + err.message)
    }
  }

  useEffect(() => { load() }, [])

  async function upload(event) {
    const files = Array.from(event.target.files)
    if (!files.length) return
    const body = new FormData()
    files.forEach((f) => body.append("files", f))
    setBusy(true)
    setStatus("Indexiere …")
    try {
      const res = await fetch("/documents", { method: "POST", body })
      setStatus(res.ok ? `${files.length} Datei(en) hinzugefügt.` : (await res.json()).detail)
    } catch (err) {
      setStatus("Upload fehlgeschlagen: " + err)
    }
    setBusy(false)
    input.current.value = ""
    load()
  }

  async function remove(name) {
    await fetch("/documents/" + encodeURIComponent(name), { method: "DELETE" })
    load()
  }

  return (
    <div className="flex flex-col gap-3 p-1 text-sm">
      <input ref={input} type="file" multiple hidden onChange={upload}
        accept=".pdf,.docx,.txt,.md,.mdx,.json,.csv,.html,.xml,.yml,.yaml,.log" />
      <Button disabled={busy} onClick={() => input.current.click()}>
        <Upload className="mr-2 h-4 w-4" /> Dokumente hochladen
      </Button>
      {status && <div className="text-muted-foreground">{status}</div>}
      <ul className="flex flex-col gap-2">
        {docs.length === 0 && <li className="text-muted-foreground">Noch keine Dokumente.</li>}
        {docs.map((doc) => (
          <li key={doc.name} className="flex items-center justify-between gap-2">
            <a className="truncate hover:underline" href={"/files/" + encodeURIComponent(doc.name)} target="_blank"
              rel="noreferrer" title={doc.name + " öffnen"}>
              {doc.name} <span className="text-muted-foreground">({doc.chunks})</span>
            </a>
            <Button variant="ghost" size="icon" onClick={() => remove(doc.name)} title="Entfernen">
              <Trash2 className="h-4 w-4" />
            </Button>
          </li>
        ))}
      </ul>
    </div>
  )
}
