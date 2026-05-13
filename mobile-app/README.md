# mobile-app/

**The Android/Flutter client.** Orchestrates everything that runs on-device: camera, microphone, MediaPipe Holistic, the LiteRT-exported gloss tagger from `../recognition/`, the LiteRT-exported Gemma 4 E4B from `../gemma-glossing/`, the stickman renderer specified in `../generation/`, the system TTS, and the UI that switches between Path A and Path B.

No training happens here. This folder consumes the artefacts produced by the other three.

## What this folder produces

- An installable Android app (`.apk` / `.aab`) that runs the full Sema demo on-device.

## Intended layout

```
mobile-app/
├── README.md
├── settings.gradle.kts
├── build.gradle.kts
├── gradle.properties
├── app/
│   ├── build.gradle.kts
│   ├── proguard-rules.pro
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── assets/
│       │   ├── models/
│       │   │   ├── gloss_tagger.tflite     # from ../recognition/export/
│       │   │   └── gemma4_e4b_int4.tflite  # from ../gemma-glossing/export/
│       │   └── pose_library/               # from ../generation/pose_library/
│       ├── java/com/signvrse/sema/
│       │   ├── MainActivity.kt
│       │   ├── ui/                         # Compose screens, mode switcher
│       │   ├── camera/                     # CameraX capture + preview
│       │   ├── mic/                        # AudioRecord + VAD
│       │   ├── mediapipe/                  # Holistic landmarker wrapper
│       │   ├── asr/                        # Whisper-tiny or system STT
│       │   ├── models/
│       │   │   ├── GlossTagger.kt          # LiteRT runner for recognition
│       │   │   └── GemmaTranslator.kt      # LiteRT runner for translation
│       │   ├── tts/                        # Android TextToSpeech wrapper
│       │   ├── render/                     # Canvas/OpenGL stickman drawer
│       │   └── pipeline/
│       │       ├── PathA.kt                # camera → gloss → text → TTS
│       │       └── PathB.kt                # mic → text → gloss → avatar
│       └── res/                            # layouts, strings, drawables
├── benchmark/                              # latency / battery harness
└── docs/
    ├── permissions.md                      # camera, mic, storage
    └── device_support.md                   # min spec, tested devices
```

## How the pieces connect

```
PathA.kt
  └── camera/         frames
      └── mediapipe/  keypoints
          └── models/GlossTagger.kt   gloss tokens
              └── models/GemmaTranslator.kt   EN/SW text
                  └── tts/                    audio out

PathB.kt
  └── mic/            audio
      └── asr/        text
          └── models/GemmaTranslator.kt   gloss tokens
              └── render/                 stickman animation
```

## Artefacts consumed from sibling folders

| Asset | Comes from | Goes to |
|---|---|---|
| `gloss_tagger.tflite` | `../recognition/export/` | `app/src/main/assets/models/` |
| `gemma4_e4b_int4.tflite` | `../gemma-glossing/export/` | `app/src/main/assets/models/` |
| `pose_library/` (indexed clips) | `../generation/pose_library/` | `app/src/main/assets/pose_library/` |
| Renderer contract | `../generation/renderer/android_contract.md` | `render/` implementation |
| ASR contract | `../generation/asr/contract.md` | `asr/` implementation |

## Out of scope

- Server-side anything. Sema is on-device by design.
- iOS. Android only for the demo (see the iOS-demo reference repo in the root README for a future port).
