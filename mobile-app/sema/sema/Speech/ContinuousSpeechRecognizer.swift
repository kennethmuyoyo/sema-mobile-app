import AVFoundation
import Foundation
import Observation
import Speech

/// SFSpeechRecognizer wrapped for continuous live transcription.
///
/// v0 implementation: a single `SFSpeechAudioBufferRecognitionRequest`,
/// auto-restarted when it ends (either by the 1-minute cap or after a long
/// silence). The full 50-second rotation-with-overlap strategy from
/// `generation/asr/contract.md` is a follow-up — for now this gets us a
/// usable transcript stream.
///
/// Reads `TTSGate.shared.isSpeaking` and pauses audio routing while true to
/// avoid the avatar's speech echoing back into the mic.
@MainActor
@Observable
final class ContinuousSpeechRecognizer: NSObject {

    struct TranscriptEvent: Sendable, Equatable {
        let text: String
        let isFinal: Bool
        let language: String
    }

    /// Live transcript text. `isFinal` events also surface through
    /// `transcriptHandler` below.
    var liveText: String = ""
    private(set) var isRunning = false
    private(set) var errorMessage: String? = nil

    /// Receives every transcript event. Set by `PathBCoordinator` on each
    /// `start()`, cleared on `stop()`. Replaced the previous `AsyncStream`
    /// approach because that stream's single-consumer contract meant any
    /// post-pause re-iteration would silently miss events (the symptom:
    /// listening worked the first time but went silent after a
    /// watching→listening toggle even though the mic tap was healthy).
    var transcriptHandler: (@MainActor (TranscriptEvent) -> Void)?

    private let recogniser: SFSpeechRecognizer?
    /// Rebuilt on every `start()` because reusing the same AVAudioEngine
    /// across audio-session category swaps (`.playAndRecord` ⇄ `.playback`,
    /// which is what watching/listening mode toggles do) leaves the engine's
    /// internal graph in a state where `start()` succeeds but the input node
    /// produces no buffers. A fresh engine sidesteps that entirely.
    @ObservationIgnored private var audioEngine = AVAudioEngine()
    @ObservationIgnored private let audioFeed = RecognitionAudioFeed()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var ttsObserver: Task<Void, Never>? = nil
    private var pausedForTTS = false
    /// Prefer on-device per ASR contract; falls back to server if assets fail.
    private var preferOnDeviceRecognition = true
    /// Bumped when a task is torn down so stale callbacks are ignored.
    private var recognitionEpoch = 0
    @ObservationIgnored private var lastRotationTime: Date = .distantPast

    init(localeIdentifier: String = "en-US") {
        self.recogniser = SFSpeechRecognizer(locale: Locale(identifier: localeIdentifier))
        super.init()
    }

    static var speechAuthorizationStatus: SFSpeechRecognizerAuthorizationStatus {
        SFSpeechRecognizer.authorizationStatus()
    }

    @discardableResult
    func requestAuthorisation() async -> Bool {
        switch Self.speechAuthorizationStatus {
        case .authorized:
            return true
        case .denied, .restricted:
            return false
        case .notDetermined:
            return await withCheckedContinuation { continuation in
                SFSpeechRecognizer.requestAuthorization { status in
                    continuation.resume(returning: status == .authorized)
                }
            }
        @unknown default:
            return false
        }
    }

    func start() throws {
        guard !isRunning else { return }
        guard let recogniser, recogniser.isAvailable else {
            throw RecogniserError.unavailable
        }
        switch Self.speechAuthorizationStatus {
        case .authorized:
            break
        case .denied, .restricted:
            throw RecogniserError.speechPermissionDenied
        case .notDetermined:
            throw RecogniserError.speechPermissionUndetermined
        @unknown default:
            throw RecogniserError.unavailable
        }
        preferOnDeviceRecognition = true
        recognitionEpoch = 0
        try AudioSession.shared.configure()
        try beginCaptureAndRecognition()
        isRunning = true
        errorMessage = nil
        observeTTSGate()
    }

    func stop() {
        ttsObserver?.cancel()
        ttsObserver = nil
        finishRecognitionTask()
        tearDownMicrophoneTap()
        // NB: we no longer deactivate the audio session here. The
        // orchestrator owns the AudioSession lifecycle now — flipping it on
        // every stop forced CoreAudio to renegotiate with its server on
        // every mode toggle and produced `IPCAUClient: can't connect to
        // server` / empty-buffer errors when listening came back up after
        // watching. The session stays in `.playAndRecord` across
        // pause/start cycles; only the orchestrator's rollbackLiveSession
        // deactivates it (when the user actually stops the conversation).
        isRunning = false
    }

    /// Reinstalls the mic tap after the front camera session starts (shared
    /// `AVAudioSession` can invalidate the previous tap format).
    func reinstallAfterSharedSessionChange() throws {
        guard isRunning else { return }
        finishRecognitionTask()
        tearDownMicrophoneTap()
        try AudioSession.shared.configure()
        try beginCaptureAndRecognition()
        print("[ASR] microphone tap reinstalled after camera session change")
    }

    // MARK: - Internals

    private func beginCaptureAndRecognition() throws {
        try startRecognitionTask()
        try installMicrophoneTap()
    }

    private func installMicrophoneTap() throws {
        // Force a clean engine. The lazy-singleton approach worked while the
        // audio session never changed category, but the mode toggle now
        // bounces between .playAndRecord and .playback — and the underlying
        // AVAudioEngine sometimes "starts" successfully on the second
        // listening session but never delivers buffers (mic tap fires zero
        // callbacks). Rebuilding the engine is cheap and unambiguous.
        if audioEngine.isRunning { audioEngine.stop() }
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.reset()
        audioEngine = AVAudioEngine()
        let input = audioEngine.inputNode

        audioEngine.prepare()
        if !audioEngine.isRunning {
            try audioEngine.start()
        }

        let hardwareFormat = input.outputFormat(forBus: 0)
        guard hardwareFormat.sampleRate > 0, hardwareFormat.channelCount > 0 else {
            throw RecogniserError.invalidTapFormat
        }
        print(
            "[ASR] tap format: \(hardwareFormat.sampleRate) Hz, "
                + "\(hardwareFormat.channelCount) ch"
        )

        let feed = audioFeed
        input.installTap(onBus: 0, bufferSize: 1024, format: hardwareFormat) { buffer, _ in
            feed.append(buffer)
        }
    }

    private func tearDownMicrophoneTap() {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.reset()
    }

    private func startRecognitionTask() throws {
        guard let recogniser else { return }

        let epoch = recognitionEpoch
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        if #available(iOS 13.0, *) {
            let useOnDevice = preferOnDeviceRecognition && recogniser.supportsOnDeviceRecognition
            request.requiresOnDeviceRecognition = useOnDevice
            print("[ASR] started recognition task (onDevice=\(useOnDevice), epoch=\(epoch))")
        }
        request.taskHint = .dictation
        self.request = request
        audioFeed.bind(request)

        task = recogniser.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }
            Task { @MainActor in
                guard epoch == self.recognitionEpoch else { return }

                if let result {
                    let text = result.bestTranscription.formattedString
                    guard !text.isEmpty else { return }
                    self.errorMessage = nil
                    self.liveText = text
                    self.transcriptHandler?(TranscriptEvent(
                        text: text,
                        isFinal: result.isFinal,
                        language: recogniser.locale.identifier
                    ))
                    if result.isFinal {
                        print("[ASR] final transcript: \"\(text)\"")
                        self.rotateRecognitionRequest()
                    }
                    return
                }

                if let error {
                    self.handleRecognitionError(error)
                }
            }
        }
    }

    private func handleRecognitionError(_ error: Error) {
        guard isRunning else { return }
        let nsError = error as NSError

        // Cancel / no-speech / end-of-request — never spin-restart (caused tight loops).
        if Self.isBenignRecognitionError(nsError) {
            return
        }

        if preferOnDeviceRecognition, Self.isOnDeviceRecognitionFailure(nsError) {
            preferOnDeviceRecognition = false
            print("[ASR] on-device recognition unavailable; retrying with server")
            rotateRecognitionRequest()
            return
        }

        print("[ASR] recognition error (\(nsError.domain) \(nsError.code)): \(error)")
        errorMessage = error.localizedDescription
    }

    private static func isBenignRecognitionError(_ error: NSError) -> Bool {
        if error.domain == "kAFAssistantErrorDomain" {
            switch error.code {
            case 1110, 1101, 1107: // cancelled, no speech, not available yet
                return true
            default:
                break
            }
        }
        if error.domain == NSURLErrorDomain, error.code == NSURLErrorCancelled {
            return true
        }
        return false
    }

    private static func isOnDeviceRecognitionFailure(_ error: NSError) -> Bool {
        if error.domain == "kAFAssistantErrorDomain", error.code == 216 { return true }
        if error.localizedDescription.localizedCaseInsensitiveContains("on-device") {
            return true
        }
        return false
    }

    private func finishRecognitionTask() {
        recognitionEpoch += 1
        request?.endAudio()
        task?.cancel()
        audioFeed.bind(nil)
        request = nil
        task = nil
    }

    private func rotateRecognitionRequest() {
        guard isRunning, !pausedForTTS else { return }
        let now = Date()
        guard now.timeIntervalSince(lastRotationTime) >= 0.35 else { return }
        lastRotationTime = now
        finishRecognitionTask()
        do {
            try startRecognitionTask()
        } catch {
            errorMessage = "Speech recognition restart failed: \(error)"
            print("[ASR] rotate failed: \(error)")
        }
    }

    private func observeTTSGate() {
        ttsObserver?.cancel()
        ttsObserver = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let speaking = TTSGate.shared.isSpeaking
                if speaking && !self.pausedForTTS {
                    self.pausedForTTS = true
                    self.audioEngine.pause()
                } else if !speaking && self.pausedForTTS {
                    self.pausedForTTS = false
                    try? self.audioEngine.start()
                }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
    }
}

/// Thread-safe bridge from the realtime audio tap to the active recognition request.
private final class RecognitionAudioFeed: @unchecked Sendable {
    private let lock = NSLock()
    private var request: SFSpeechAudioBufferRecognitionRequest?

    func bind(_ request: SFSpeechAudioBufferRecognitionRequest?) {
        lock.lock()
        defer { lock.unlock() }
        self.request = request
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        let request = self.request
        lock.unlock()
        request?.append(buffer)
    }
}

enum RecogniserError: Error, CustomStringConvertible {
    case unavailable
    case speechPermissionDenied
    case speechPermissionUndetermined
    case invalidTapFormat

    var description: String {
        switch self {
        case .unavailable:
            return "Speech recognition is unavailable for this device or language."
        case .speechPermissionDenied:
            return "Speech recognition is not allowed. Enable it in Settings."
        case .speechPermissionUndetermined:
            return "Speech recognition permission is required before listening."
        case .invalidTapFormat:
            return "Microphone audio format is invalid after the camera started. Try stopping and starting again."
        }
    }
}
