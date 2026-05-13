# ASR contract — SFSpeechRecognizer, continuous mode

This is the contract `mobile-app/Sema/Speech/ContinuousSpeechRecognizer.swift` implements against.

## Mode

`SFSpeechRecognizer` with on-device recognition forced (`requiresOnDeviceRecognition = true`). Apple Speech is free, on-device-capable on iOS 13+, and supports the languages we need (English + Swahili — Swahili availability is device/region-dependent; the client falls back to English if `SFSpeechRecognizer(locale:)` returns `nil` for Swahili).

## Request rotation

`SFSpeechAudioBufferRecognitionRequest` has an Apple-imposed per-request limit (~1 minute, undocumented exactly). The continuous wrapper rotates requests:

1. Open `request_n` with `shouldReportPartialResults = true` and `taskHint = .dictation`.
2. Forward audio buffers from the shared `AVAudioEngine` tap.
3. At `t = 50 s` after `request_n.startTime`, open `request_{n+1}` and start forwarding **the same buffers** to both for a 1 s overlap window.
4. At `t = 51 s`, finish `request_n` (`endAudio()`), keep `request_{n+1}`.
5. Stitch transcripts: take `request_n`'s final transcript through to the overlap midpoint; from `request_{n+1}` use the suffix after that midpoint. Resolve duplicates by longest-common-suffix matching on the overlap.

## TTS coordination

`AVSpeechSynthesizer` and `SFSpeechRecognizer` cannot run simultaneously without echo (the speaker output is picked up by the mic).

| Event | Action |
|---|---|
| `synthesizer:willSpeak` | Pause the recognizer (`audioEngine.pause()`); buffer any text the user signs in the meantime via Path A. |
| `synthesizer:didFinish` | Resume after a 200 ms tail to let the audio tail decay; flush any buffered audio. |
| `synthesizer:didCancel` | Same as didFinish. |
| User dismisses TTS | Same. |

## Output

A throttled stream of `Transcript` events:

```swift
struct Transcript {
    let text: String          // best-effort full sentence so far
    let isFinal: Bool         // false for partials, true on segment boundary
    let confidence: Float?    // SFTranscription.averageConfidence, if available
    let language: Locale
}
```

The Gemma translator in `Sema/ML/GemmaTranslator.swift` subscribes and triggers a translation on `isFinal = true` OR when `text.count - lastTranslatedCount > 80` to keep latency tolerable on long utterances.

## Permissions and audio session

| Item | Value |
|---|---|
| `NSMicrophoneUsageDescription` | "Sema converts your speech into sign-language animation." |
| `NSSpeechRecognitionUsageDescription` | "Sema recognizes what you say so it can translate it to KSL signs." |
| `AVAudioSession.Category` | `.playAndRecord` (see `../../mobile-app/README.md` for full config) |

## What this contract does NOT cover

- Wake-word detection. The recognizer is active whenever the app is foregrounded; there is no "Hey Sema".
- Speaker separation. Single-speaker input only.
- Non-speech audio events (music, noise). Out of scope for the demo.
