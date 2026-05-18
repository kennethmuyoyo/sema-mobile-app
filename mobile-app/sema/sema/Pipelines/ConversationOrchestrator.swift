import Foundation
import Observation
import UIKit

/// Which on-device pose libraries the avatar plays from. Compile-time switch
/// — edit this and rebuild to toggle. There's no UI for it yet.
///
/// List order is priority. The PoseDatabase merges these in order and the
/// first hit wins on token collisions, so a curated demo clip always
/// shadows a same-named alignment-derived clip.
///
///   - `"PoseLibrary"`     : curated demo set (~31 clips, hand-picked for
///                            the Hospital + Bank scenarios).
///   - `"PoseLibraryFull"` : 406-clip v11-derived library built by
///                            `generation/pose_library/build_full_library.py`.
///                            Boundaries are recogniser-aligned, so
///                            quality varies (sort by `alignment_quality`
///                            in `index_full.json` for the clean takes).
///
/// Both folders ship in the app bundle; see `docs/bundle_inventory.md`.
private let kActivePoseLibrarySubdirs = ["PoseLibrary", "PoseLibraryFull"]

/// Runs Path A (sign → speech) and Path B (speech → sign) on one live session
/// with a headless front camera. Default behavior is simultaneous: both paths
/// run together, turn-taking implicit (ASR `isFinal`, no-hand timeout).
///
/// The `SessionMode` infrastructure (`.listening`/`.watching` half-duplex) is
/// kept available for fallback if the camera↔mic CoreMediaIO race recurs —
/// callers can invoke `switchMode()` / the `enterListeningMode` /
/// `enterWatchingMode` methods to force one sensor off. Not gated by default.
@MainActor
@Observable
final class ConversationOrchestrator {
    /// Half-duplex sub-state, available via `switchMode()` if a caller needs
    /// to force exclusive use of one sensor. The default startup path
    /// (`startSession`) leaves this at `.listening` after bringing both
    /// paths up — it's a soft status hint, not an enforced gate.
    ///
    /// `.finalizing` is the brief window after Stop is tapped while outputs
    /// (TTS / avatar playback) are still wrapping up. The mic + camera are
    /// already off but the session is still "live" from the user's POV — the
    /// caption card stays visible, isLive stays true, and the previous
    /// listening / watching content stays on screen until the outputs end.
    enum SessionMode: String, Equatable {
        case idle = "Idle"
        case listening = "Listening"   // mic primary
        case watching = "Watching"     // camera primary
        case finalizing = "Finalizing" // Stop tapped, outputs flushing
    }

    let pathA: PathACoordinator
    let pathB: PathBCoordinator
    let camera: CameraSessionController

    var bridge: SignToSpeechBridge

    private(set) var mode: SessionMode = .idle
    /// Remembers the most recent active capture mode (`.listening` or
    /// `.watching`) so the caption card can keep rendering the right track
    /// while `mode == .finalizing` and outputs are still wrapping up.
    private(set) var lastActiveMode: SessionMode = .listening
    /// Derived from `mode` — kept as a separate accessor so call sites that
    /// only care "is the session up" don't have to know about half-duplex.
    var sessionActive: Bool { mode != .idle }
    private(set) var sessionError: String?
    private(set) var mediaPermissions = MediaPermissionSnapshot()

    var areCaptionsVisible = true

    @ObservationIgnored private var ttsObserver: Task<Void, Never>?
    @ObservationIgnored private var wasSpeaking = false
    @ObservationIgnored private var startSessionTask: Task<Void, Never>?
    @ObservationIgnored private var memoryWarningObserver: NSObjectProtocol?
    @ObservationIgnored private var gemmaWarmupTask: Task<Void, Never>?
    @ObservationIgnored private var finalizeTask: Task<Void, Never>?
    private(set) var isStartingSession = false

    /// How long Stop lingers in `.finalizing` so TTS/avatar can wrap up.
    /// Most utterances finish well inside this; we still hard-rollback at
    /// the end so the app never gets stuck.
    private let finalizeLinger: Duration = .seconds(7)

    init(
        pathA: PathACoordinator,
        pathB: PathBCoordinator,
        camera: CameraSessionController
    ) {
        self.pathA = pathA
        self.pathB = pathB
        self.camera = camera
        self.bridge = SignToSpeechBridge(translator: pathB.translator)
        wireCameraCallbacks()
    }

    /// Convenience no-arg init — constructs default dependencies inside the body
    /// (not as default parameter expressions) to satisfy Swift 6 strict concurrency.
    init() {
        let pathA = PathACoordinator()
        // Translator mode for both paths:
        //   .onDevice — bundled `gemma-4-e2b-ksl-Q4_K_M.gguf` via
        //               LlamaGemmaEngine, drives both KSL→EN (PathA → TTS)
        //               and EN→KSL (PathB → avatar) through the few-shot
        //               prompts in KSLPrompts. Costs ~3.2 GB Metal at load;
        //               the memory-warning observer (see `bootstrap`) will
        //               close the engine if iOS signals pressure.
        //   .stub     — phrasebook lookup, zero GPU cost. Falls back to
        //               per-word uppercase glosses for inputs outside the
        //               Hospital + Bank demo phrases.
        //   .server   — HTTPS endpoint per `gemma-glossing/README.md` —
        //               for cloud inference once an endpoint is configured.
        // SignToSpeechBridge and PathBCoordinator both share `pathB.translator`,
        // so flipping the mode here is enough.
        let pathB = PathBCoordinator(
            translatorMode: .onDevice,
            poseLibrarySubdirs: kActivePoseLibrarySubdirs
        )
        let camera = CameraSessionController()
        self.pathA = pathA
        self.pathB = pathB
        self.camera = camera
        self.bridge = SignToSpeechBridge(translator: pathB.translator)
        wireCameraCallbacks()
        if TestEnvironment.isPreview {
            pathA.prepareForPreview()
            pathB.prepareForPreview()
        }
    }

    /// Orchestrator with both paths already `.ready` for SwiftUI previews.
    static func forPreview() -> ConversationOrchestrator {
        let orchestrator = ConversationOrchestrator()
        orchestrator.pathA.prepareForPreview()
        orchestrator.pathB.prepareForPreview()
        return orchestrator
    }

    var heardText: String { pathB.heardCaption }

    var spokenReply: String { bridge.spokenSentence }

    var signTranscript: String { pathA.transcript }

    var isPreparing: Bool {
        pathA.state == .warmingUp || pathB.state == .warmingUp
    }

    var isFailed: Bool {
        pathA.state == .failed || pathB.state == .failed
    }

    var canStart: Bool {
        pathsAreReady && !isFailed
    }

    private var pathsAreReady: Bool {
        (pathA.state == .ready || pathA.state == .running)
            && (pathB.state == .ready || pathB.state == .running)
    }

    var isLive: Bool { sessionActive }
    /// During `.finalizing` these fall back to whichever was the last
    /// active capture mode so the caption card keeps showing that track's
    /// content while outputs flush.
    var isListening: Bool {
        if mode == .listening { return true }
        if mode == .finalizing { return lastActiveMode == .listening }
        return false
    }
    var isWatching: Bool {
        if mode == .watching { return true }
        if mode == .finalizing { return lastActiveMode == .watching }
        return false
    }

    var shouldShowCaptions: Bool {
        areCaptionsVisible && isLive
    }

    var statusLabel: String {
        if isFailed { return "Error" }
        if isPreparing { return "Preparing…" }
        if isStartingSession { return "Starting…" }
        switch mode {
        case .listening:
            return "Listening · tap camera to sign"
        case .watching:
            return "Watching · tap camera to listen"
        case .finalizing:
            return "Finishing up…"
        case .idle:
            return "Ready"
        }
    }

    var combinedError: String? {
        if let sessionError { return sessionError }
        if let msg = pathA.errorMessage { return msg }
        if let msg = pathB.errorMessage { return msg }
        return nil
    }

    func bootstrap() {
        guard !TestEnvironment.skipsPipelineStartup else { return }
        sessionError = nil
        camera.frameDelegate = pathA
        pathA.bootstrap()
        pathB.bootstrap()
        Task { await refreshMediaPermissions() }
        // NB: Gemma warmup is *not* fired here. Doing so collides with the
        // camera + MediaPipe + CoreML Metal initialization burst — on
        // memory-tight devices the Gemma context's `sched_reserve` finishes
        // right as AVCaptureSession tries to bring up the front camera, and
        // the next access to model hparams crashes with EXC_BAD_ACCESS.
        // Warmup now happens in `startSession` once camera + ASR are stable.
        // Under memory pressure free the 3 GB Gemma allocation; the next
        // translate call will lazily reload it. Better a slow first reply
        // than a jetsam kill mid-session.
        if memoryWarningObserver == nil {
            memoryWarningObserver = NotificationCenter.default.addObserver(
                forName: UIApplication.didReceiveMemoryWarningNotification,
                object: nil,
                queue: .main
            ) { _ in
                print("[Orchestrator] memory warning — closing Gemma engine")
                Task { await GemmaTranslator.closeSharedOnDeviceEngine() }
            }
        }
    }

    /// Refreshes permission flags (requests mic + speech when still undetermined).
    func refreshMediaPermissions() async {
        var snapshot = MediaPermissionSnapshot.capture()

        if snapshot.microphone == .unknown {
            snapshot.microphone = await AudioSession.shared.requestRecordPermission()
                ? .granted : .denied
        }
        if snapshot.speechRecognition == .unknown {
            snapshot.speechRecognition = await pathB.recogniser.requestAuthorisation()
                ? .granted : .denied
        }
        snapshot.camera = MediaPermissionSnapshot.capture().camera
        mediaPermissions = snapshot
    }

    /// Starts a live session in **listening** mode (mic on, camera off).
    /// Half-duplex by construction — the camera↔mic CoreMediaIO race used to
    /// make the gloss tagger and ASR fight each other; with only one sensor
    /// active at a time both paths actually produce data. Switch to watching
    /// (camera + gloss tagger) by tapping the avatar, which calls `toggleMode()`.
    func startSession() async {
        guard pathsAreReady, !isFailed else { return }
        sessionError = nil
        await refreshMediaPermissions()

        guard mediaPermissions.canListen else {
            sessionError = Self.listenPermissionMessage(for: mediaPermissions)
            return
        }

        await enterListeningMode()

        if mode == .listening {
            startTTSObservation()
            scheduleGemmaWarmupIfNeeded()
        }
        await refreshMediaPermissions()
    }

    /// Kicks off Gemma warmup with a short delay, but only when listening
    /// (camera is off and the 3.2 GB Metal allocation has the whole GPU).
    /// The simultaneous bring-up with capture was crashing llama context
    /// construction at `llama_hparams::n_embd_inp`. Subsequent calls are
    /// cheap — `ensureLoaded()` short-circuits once the model is resident.
    private func scheduleGemmaWarmupIfNeeded() {
        guard gemmaWarmupTask == nil else { return }
        let translator = pathB.translator
        gemmaWarmupTask = Task.detached(priority: .utility) { [weak self] in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            guard !Task.isCancelled, self != nil else { return }
            try? await translator.warmupOnDevice()
        }
    }

    /// Flip between listening (mic) and watching (camera). No-op when idle
    /// or finalizing (outputs are still flushing — don't interrupt them).
    func switchMode() async {
        switch mode {
        case .listening:
            await enterWatchingMode()
        case .watching:
            await enterListeningMode()
        case .idle, .finalizing:
            return
        }
        await refreshMediaPermissions()
    }

    /// Async entry point for buttons that can `await`. `switchMode()` is the
    /// one above; this is the fire-and-forget wrapper used by SwiftUI buttons.
    func toggleMode() {
        guard mode == .listening || mode == .watching else { return }
        Task { await switchMode() }
    }

    private func enterListeningMode() async {
        // Tear down camera capture if it was running for .watching.
        if camera.status == .running {
            pathA.pause()
            camera.stop()
        }
        // The audio session has stayed in `.playAndRecord` since the last
        // listening session (we no longer swap it for watching mode), so
        // this `configure()` is idempotent on the second-and-later entries
        // but still useful on first start.
        do {
            try AudioSession.shared.configure()
        } catch {
            sessionError = Self.friendlyAudioMessage(error)
            mode = .idle
            return
        }
        pathB.start()
        if pathB.state == .failed {
            sessionError = pathB.errorMessage ?? "Speech recognition could not start."
            // Don't deactivate the session here either — same IPC-issue
            // surface as rollback. The user can retry from `.idle`.
            mode = .idle
            return
        }
        sessionError = nil
        mode = .listening
        lastActiveMode = .listening
        print("[Orchestrator] mode → listening (mic on, camera off)")
    }

    private func enterWatchingMode() async {
        // Tear down ASR's mic engine but keep the audio session in
        // `.playAndRecord`. TTS playback works fine in that category, and
        // swapping to `.playback` here (then back to `.playAndRecord` when
        // the user toggled to listening) was triggering CoreAudio IPC
        // failures (`IPCAUClient: can't connect to server -66748`) and
        // empty mic buffers on the return trip. Since `AVCaptureSession`
        // has `automaticallyConfiguresApplicationAudioSession = false`, it
        // doesn't claim the mic when we're not feeding one — leaving the
        // session category alone is the safer move.
        pathB.pause()
        // Wipe MediaPipe's per-joint carry-forward buffers so the first
        // post-switch frames don't fill missing joints from the previous
        // watching session's last-detected positions.
        await pathA.resetLandmarkerForNewSession()

        let cameraStatus = await camera.activate()
        mediaPermissions.camera = Self.permissionStatus(from: cameraStatus)

        switch cameraStatus {
        case .running:
            pathA.start()
            sessionError = nil
            mode = .watching
            lastActiveMode = .watching
            print("[Orchestrator] mode → watching (camera on, mic off, TTS playback enabled)")
        case .denied:
            sessionError = "Camera access is off — enable it in Settings → Sema to sign."
            // Fall back to listening so the user isn't stuck in a broken mode.
            await enterListeningMode()
        case .failed(let message):
            sessionError = "Front camera unavailable (\(message))."
            await enterListeningMode()
        case .starting, .requestingPermission, .idle:
            sessionError = "Front camera is still starting. Try again in a moment."
            await enterListeningMode()
        }
    }

    /// Fire-and-forget entry for buttons; prefer `startSession()` when you can await.
    func start() {
        guard !isStartingSession, canStart else { return }
        // Interrupt any in-progress finalize linger so the new session
        // doesn't get rolled back mid-bring-up.
        finalizeTask?.cancel()
        finalizeTask = nil
        if mode == .finalizing {
            // Old outputs were still playing; tear them down so the new
            // session starts on a clean slate.
            rollbackLiveSession()
        }
        isStartingSession = true
        startSessionTask = Task {
            await startSession()
            isStartingSession = false
            startSessionTask = nil
        }
    }

    /// Stop = "period". Commits whatever the user was in the middle of
    /// (recognized gloss burst → TTS, or heard transcript → avatar) and
    /// transitions into `.finalizing` so the caption card + outputs stay on
    /// screen while TTS/avatar wrap up. Inputs (mic, camera) are released
    /// immediately. After `finalizeLinger` we hard-rollback to `.idle`.
    /// Calling `pause()` again or `start()` interrupts the linger.
    func pause() {
        startSessionTask?.cancel()
        startSessionTask = nil
        isStartingSession = false

        // If we're already in .finalizing and the user taps Stop a second
        // time, treat it as "yes, really stop now" and rollback immediately.
        if mode == .finalizing {
            finalizeTask?.cancel()
            finalizeTask = nil
            rollbackLiveSession()
            Task { await refreshMediaPermissions() }
            return
        }
        // Same if we were already idle — defensive no-op rollback.
        if mode == .idle {
            rollbackLiveSession()
            Task { await refreshMediaPermissions() }
            return
        }

        // Active session: commit pending state THEN ease inputs down.
        finalizePendingOutputs()

        // Release sensors immediately (mic / camera). Outputs (TTS, avatar
        // playback) keep running because we don't touch the bridge or
        // pause signing playback here.
        pathB.stopInputOnly()
        pathA.pause()
        camera.stop()
        ttsObserver?.cancel()
        ttsObserver = nil
        wasSpeaking = false
        gemmaWarmupTask?.cancel()
        gemmaWarmupTask = nil

        mode = .finalizing
        print("[Orchestrator] mode → finalizing (inputs off, outputs running)")

        // Schedule the hard rollback after outputs have had time to finish.
        finalizeTask?.cancel()
        finalizeTask = Task { [weak self] in
            try? await Task.sleep(for: self?.finalizeLinger ?? .seconds(7))
            guard !Task.isCancelled, let self else { return }
            if self.mode == .finalizing {
                self.rollbackLiveSession()
                await self.refreshMediaPermissions()
            }
        }
        Task { await refreshMediaPermissions() }
    }

    /// Commit the in-flight recognition / generation state so the user
    /// gets to see (and hear) the sentence they were mid-way through.
    ///   • Watching → commit emitted gloss tokens as a finalized phrase,
    ///     which the screen view's onChange handler then routes through
    ///     the bridge to translate + TTS-speak.
    ///   • Listening → force one more translate pass on the last partial
    ///     ASR transcript, queue the resulting glosses to the stitcher so
    ///     the avatar finishes signing the user's spoken sentence.
    private func finalizePendingOutputs() {
        if !pathA.emittedTokens.isEmpty {
            let phrase = pathA.emittedTokens.map(\.label).joined(separator: " ")
            print("[Orchestrator] finalize recognition: \"\(phrase)\"")
            pathA.finalizedGlossPhrase = phrase
        }
        pathB.finalizePendingTranscript()
    }

    /// Called when Path A finalizes a signing burst (`noHandTimeout`).
    func handleFinalizedGloss(_ phrase: String) {
        bridge.handle(finalizedGloss: phrase)
    }

    func toggleCaptions() {
        areCaptionsVisible.toggle()
    }

    private func wireCameraCallbacks() {
        // Fires when camera permission is granted asynchronously and the
        // capture session reaches `.running`. With half-duplex, only start
        // PathA when we're actually in watching mode — otherwise the camera
        // shouldn't even have been activated.
        camera.onBecameRunning = { [weak self] in
            guard let self else { return }
            if self.mode == .watching, self.pathA.state == .ready {
                self.pathA.start()
            }
            Task { await self.refreshMediaPermissions() }
        }
    }

    private func rollbackLiveSession() {
        mode = .idle
        ttsObserver?.cancel()
        ttsObserver = nil
        wasSpeaking = false
        // Cancel any not-yet-fired warmup so it doesn't fire after the user
        // stopped. Already-running loads complete on the actor — they're
        // cheap to keep since the model stays resident for the next session.
        gemmaWarmupTask?.cancel()
        gemmaWarmupTask = nil
        pathB.pause()
        pathB.pauseSigningPlayback()
        pathA.pause()
        bridge.cancel()
        camera.stop()
        // Intentionally NOT calling AudioSession.shared.deactivate() here.
        // Stop → Start cycles would otherwise hit the same
        // deactivate/reactivate that broke CoreAudio's IPCAUClient
        // (-66748) and produced empty mic buffers on the new session.
        // The session stays in `.playAndRecord` for the app's lifetime;
        // iOS releases it automatically when we background.
    }

    /// Pauses in-flight PathB signing when TTS starts, so the avatar isn't
    /// mid-clip while the system is also speaking. Manual mode toggle stays
    /// the only way to switch between listening and watching — the auto
    /// turn-taking version was unintuitive in testing.
    private func startTTSObservation() {
        ttsObserver?.cancel()
        ttsObserver = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let speaking = TTSGate.shared.isSpeaking
                if speaking, !self.wasSpeaking {
                    self.pathB.pauseSigningPlayback()
                }
                self.wasSpeaking = speaking
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
    }

    private static func permissionStatus(from status: CameraSessionController.Status) -> MediaPermissionSnapshot.Status {
        switch status {
        case .running:
            return .granted
        case .denied:
            return .denied
        case .failed:
            return .unavailable
        case .idle, .requestingPermission, .starting:
            return .unknown
        }
    }

    private static func listenPermissionMessage(for permissions: MediaPermissionSnapshot) -> String {
        switch (permissions.microphone, permissions.speechRecognition) {
        case (.denied, _), (.restricted, _):
            return "Microphone access is required. Turn it on in Settings → Sema."
        case (_, .denied), (_, .restricted):
            return "Speech recognition is required. Turn it on in Settings → Sema."
        case (.unknown, _), (_, .unknown):
            return "Microphone and speech recognition permissions are required."
        default:
            return "Microphone and speech recognition must be allowed to start."
        }
    }

    private static func friendlyAudioMessage(_ error: Error) -> String {
        if let audioError = error as? AudioSessionError,
           case .microphonePermissionDenied = audioError {
            return "Microphone access is required. Turn it on in Settings → Sema."
        }
        return "Audio: \(error.localizedDescription)"
    }
}
