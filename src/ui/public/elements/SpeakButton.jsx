import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Volume2, Square } from "lucide-react"

// Liest die Antwort mit der Sprachausgabe des Browsers vor (Web Speech API, läuft lokal im Browser).
export default function SpeakButton() {
  const [speaking, setSpeaking] = useState(false)
  const synth = window.speechSynthesis

  useEffect(() => () => synth && synth.cancel(), [])

  function toggle() {
    if (!synth) return
    if (synth.speaking) {
      synth.cancel()
      setSpeaking(false)
      return
    }
    const utterance = new SpeechSynthesisUtterance(props.text)
    utterance.lang = "de-DE"
    utterance.onend = utterance.onerror = () => setSpeaking(false)
    setSpeaking(true)
    synth.speak(utterance)
  }

  if (!synth) return null
  return (
    <Button variant="outline" size="sm" onClick={toggle}>
      {speaking ? <Square className="mr-2 h-4 w-4" /> : <Volume2 className="mr-2 h-4 w-4" />}
      {speaking ? "Stopp" : "Vorlesen"}
    </Button>
  )
}
