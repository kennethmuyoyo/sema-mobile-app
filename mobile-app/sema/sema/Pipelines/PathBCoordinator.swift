import Foundation
import Observation

/// Wires `ContinuousSpeechRecognizer → GemmaTranslator → Stitcher →
/// AvatarStreamPlayer`. Per `mobile-app/docs/path_b_avatar.md`, this is the
/// hearing-user-speaks → Deaf-user-sees-avatar path. The player holds the
/// latest `PoseFrame`, which `SimpleAvatar3DView` consumes each tick.
@MainActor
@Observable
final class PathBCoordinator {

    enum State: String {
        case idle = "Idle"
        case warmingUp = "Preparing…"
        case ready = "Ready"
        case running = "Listening"
        case failed = "Error"
    }

    var state: State = .idle
    var errorMessage: String? = nil
    var spokenText: String = ""
    var glossStream: [String] = []
    let player: AvatarStreamPlayer

    /// True while the avatar is mid-clip (vs holding idle). Proxies the
    /// player's own flag for callers that don't want to reach through.
    var isAvatarSigning: Bool { player.isAvatarSigning }

    let recogniser: ContinuousSpeechRecognizer

    /// Caption line for heard speech (partials + finals).
    var heardCaption: String {
        spokenText.isEmpty ? recogniser.liveText : spokenText
    }
    @ObservationIgnored let translator: GemmaTranslator
    @ObservationIgnored private(set) var stitcher: Stitcher?
    @ObservationIgnored private(set) var database: PoseDatabase?

    @ObservationIgnored private var micReinstallTask: Task<Void, Never>?
    @ObservationIgnored private var playerStreamTask: Task<Void, Never>?
    @ObservationIgnored private var missingStreamTask: Task<Void, Never>?
    @ObservationIgnored private var activePlaybackTask: Task<Void, Never>?
    @ObservationIgnored private var lastTranslated: String = ""
    /// Timestamp of the last translate call triggered by a NON-final ASR
    /// transcript. Used to throttle Gemma invocations while the user is
    /// mid-sentence so we don't fire 5 translations on 5 partials.
    @ObservationIgnored private var lastNonFinalTranslateAt: Date = .distantPast
    /// Current non-final transcript content and when it was first observed.
    /// When the same text persists for ≥1.0 s (i.e. ASR hasn't appended any
    /// more), we treat it as effectively final even though
    /// `event.isFinal == false`. Drives the "settled non-final" branch of
    /// the translate gate in `handle()`.
    @ObservationIgnored private var lastSeenTranscript: String = ""
    @ObservationIgnored private var firstSeenTranscriptAt: Date = .distantPast
    @ObservationIgnored private var suppressASRPlaybackUntil: Date = .distantPast
    @ObservationIgnored private let glossSplitChars = CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: ",;|"))
    @ObservationIgnored private var fingerspellTokenByChar: [Character: String] = [:]
    @ObservationIgnored private var poseGlossTokens: [String] = []
    @ObservationIgnored private var poseGlossByNormalized: [String: String] = [:]

    /// Which `Sema/Resources/` folders the PoseDatabase reads. List order is
    /// priority — first-hit-wins on token collisions so the curated demo
    /// library shadows the alignment-derived full library when both have
    /// the same gloss.
    ///   - `"PoseLibrary"`     : curated demo set (~31 clips, hand-picked).
    ///   - `"PoseLibraryFull"` : 406-clip v11-derived library built by
    ///                            `generation/pose_library/build_full_library.py`.
    ///                            Boundaries come from the v11 recogniser's
    ///                            sliding-window alignment; quality varies
    ///                            (sort by `alignment_quality` in the index
    ///                            to find clean takes).
    let poseLibrarySubdirs: [String]

    init(translatorMode: GemmaTranslator.Mode = .stub,
         localeIdentifier: String = "en-US",
         poseLibrarySubdirs: [String] = ["PoseLibrary", "PoseLibraryFull"]) {
        self.translator = GemmaTranslator(mode: translatorMode)
        self.recogniser = ContinuousSpeechRecognizer(localeIdentifier: localeIdentifier)
        self.player = AvatarStreamPlayer()
        self.poseLibrarySubdirs = poseLibrarySubdirs
    }

    /// Marks the path ready without loading PoseDatabase or stitcher (previews).
    func prepareForPreview() {
        state = .ready
        errorMessage = nil
    }

    func bootstrap() {
        guard !TestEnvironment.skipsPipelineStartup else {
            prepareForPreview()
            return
        }
        guard state == .idle || state == .failed else { return }
        state = .warmingUp
        errorMessage = nil
        Task { [weak self] in
            guard let self else { return }
            do {
                let database = try PoseDatabase(bundleSubdirs: self.poseLibrarySubdirs)
                let stitcher = Stitcher(database: database)
                self.database = database
                self.stitcher = stitcher
                let tokens = await database.allTokens()
                self.poseGlossTokens = tokens
                var normMap: [String: String] = [:]
                for t in tokens {
                    let k = self.normalizeForLookup(t)
                    if !k.isEmpty, normMap[k] == nil {
                        normMap[k] = t
                    }
                }
                self.poseGlossByNormalized = normMap
                let fsCoverage = await self.fingerspellCoverageSummary(db: database)
                print("[PathB] fingerspell coverage: \(fsCoverage)")

                if let idle = await self.loadIdleFrame(db: database) {
                    self.player.setIdleBase(idle)
                    self.player.startIdleLoop()
                }

                let frames = await stitcher.frames
                self.playerStreamTask = self.player.attach(frames)
                self.missingStreamTask = Task { [weak self] in
                    guard let self else { return }
                    for await miss in stitcher.missing {
                        await MainActor.run {
                            let msg = "[PathB] stitcher missing: \(miss)"
                            print(msg)
                            self.errorMessage = msg
                        }
                    }
                }
                self.state = .ready
            } catch {
                self.state = .failed
                self.errorMessage = "\(error)"
            }
        }
    }

    func start() {
        guard state == .ready || state == .running else { return }
        do {
            try recogniser.start()
            state = .running
            startObservingTranscripts()
        } catch {
            state = .failed
            errorMessage = "\(error)"
        }
    }

    /// Debounced reinstall after the camera session reconfigures shared audio.
    func scheduleMicrophoneReinstallAfterCamera() {
        guard state == .running else { return }
        micReinstallTask?.cancel()
        micReinstallTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 300_000_000)
            guard let self, !Task.isCancelled else { return }
            self.reinstallMicrophoneAfterCamera()
            self.micReinstallTask = nil
        }
    }

    /// Reinstall the mic tap after the camera session starts (keeps ASR alive).
    func reinstallMicrophoneAfterCamera() {
        guard state == .running else { return }
        do {
            try recogniser.reinstallAfterSharedSessionChange()
            errorMessage = nil
        } catch {
            errorMessage = "Microphone: \(error.localizedDescription)"
            print("[PathB] mic reinstall failed: \(error)")
        }
    }

    func pause() {
        guard state == .running else { return }
        recogniser.stop()
        recogniser.transcriptHandler = nil
        pauseSigningPlayback()
        state = .ready
    }

    /// Soft-stop: kill the mic + ASR observer but leave the stitcher and
    /// avatar player running so any in-flight gloss playback can finish on
    /// screen. Used by the orchestrator when Stop is tapped as a period —
    /// the user expects the avatar to complete the current sentence rather
    /// than freeze mid-sign.
    func stopInputOnly() {
        guard state == .running else { return }
        recogniser.stop()
        recogniser.transcriptHandler = nil
        state = .ready
    }

    /// Force a translate + enqueue pass on the latest ASR transcript so
    /// whatever the mic just heard gets signed by the avatar before the
    /// session ends. No-op if nothing new came in since the last fire.
    func finalizePendingTranscript() {
        guard !spokenText.isEmpty, spokenText != lastTranslated else { return }
        let pending = spokenText
        Task { [weak self] in
            guard let self else { return }
            await self.handle(.init(text: pending, isFinal: true, language: "en-US"))
        }
    }

    /// Stops in-flight avatar clip playback (e.g. while TTS speaks Path A output).
    func pauseSigningPlayback() {
        activePlaybackTask?.cancel()
        activePlaybackTask = nil
        if let stitcher {
            Task { await stitcher.reset() }
        }
    }

    func teardown() {
        recogniser.transcriptHandler = nil
        playerStreamTask?.cancel()
        missingStreamTask?.cancel()
        activePlaybackTask?.cancel()
        player.stopIdleLoop()
        recogniser.stop()
        if let stitcher {
            Task { await stitcher.finish() }
        }
    }

    /// Resolve a relaxed base pose for the idle animation. Scans a short list
    /// of likely-quiet candidates and picks the one whose first frame has the
    /// wrists furthest below the shoulders (i.e. arms hanging down rather
    /// than raised or mid-sign).
    private func loadIdleFrame(db: PoseDatabase) async -> PoseFrame? {
        let preferred = ["REST", "IDLE", "STAND", "READY", "OK", "ME", "I", "YOU", "GOOD"]
        var available: [String] = []
        for t in preferred {
            if await db.contains(t) { available.append(t) }
        }
        let candidates = available.isEmpty
            ? Array(await db.allTokens().sorted().prefix(30))
            : available

        let lwIdx = Landmark45.index(of: "left_wrist")
        let rwIdx = Landmark45.index(of: "right_wrist")
        let lsIdx = Landmark45.index(of: "left_shoulder")
        let rsIdx = Landmark45.index(of: "right_shoulder")

        var best: (token: String, frame: PoseFrame, score: Float)?
        for token in candidates {
            do {
                guard let clip = try await db.lookup(token), clip.frameCount > 0 else { continue }
                let f = clip.frame(at: 0)
                // MediaPipe coords: Y is DOWN. Larger Y == lower in space.
                // Score = how far below shoulders the wrists sit; bigger == better idle.
                let score = (f[lwIdx * 3 + 1] - f[lsIdx * 3 + 1])
                          + (f[rwIdx * 3 + 1] - f[rsIdx * 3 + 1])
                if best == nil || score > best!.score {
                    let frame = PoseFrame(
                        values: f,
                        rigRotations: clip.rigRotationFrame(at: 0),
                        timestamp: 0
                    )
                    best = (token, frame, score)
                }
            } catch {
                print("[PathB] idle scan failed for '\(token)': \(error)")
            }
        }
        if let best {
            print("[PathB] idle base from '\(best.token)' (wrist-below-shoulder score=\(best.score))")
            return best.frame
        }
        return nil
    }

    /// DEBUG: bypass ASR + Gemma and inject a gloss sequence directly into
    /// the stitcher. Use to verify the pose library and renderer end-to-end
    /// without dealing with mic permissions or hearing the phrasebook.
    ///
    /// Tapping a second button cancels the first one's in-flight clip
    /// playback so the new motion starts immediately instead of queueing.
    func playGlossSequence(_ glosses: [String]) {
        guard let stitcher = stitcher else { return }
        // Short debounce so a button tap can't be immediately overridden by
        // a transient ASR partial result. 30s was way too long — it silently
        // dropped every spoken word in the half-minute after every test tap.
        // TTS playback feedback is already gated separately via TTSGate.
        suppressASRPlaybackUntil = Date().addingTimeInterval(1.5)
        glossStream = glosses
        spokenText = glosses.joined(separator: " ")
        errorMessage = nil
        print("[PathB] debug playback requested: \(glosses.joined(separator: " "))")
        startPlayback(glosses: glosses, stitcher: stitcher)
    }

    private func startPlayback(glosses: [String], stitcher: Stitcher) {
        activePlaybackTask?.cancel()
        activePlaybackTask = Task { [weak self] in
            print("[PathB] startPlayback: \(glosses.joined(separator: " "))")
            await stitcher.reset()
            for g in glosses {
                if Task.isCancelled { break }
                await stitcher.append(g)
            }
            await MainActor.run { self?.activePlaybackTask = nil }
        }
    }

    // MARK: - Internals

    private func startObservingTranscripts() {
        // Direct callback instead of iterating an AsyncStream. The earlier
        // `for await event in recogniser.events` worked the first time but
        // silently dropped events on every subsequent listening session
        // because AsyncStream's single-consumer iterator becomes invalid
        // after cancellation — the new `for await` got a fresh iterator
        // that wasn't wired to the live continuation.
        recogniser.transcriptHandler = { [weak self] event in
            guard let self else { return }
            Task { [weak self] in
                await self?.handle(event)
            }
        }
    }

    private func handle(_ event: ContinuousSpeechRecognizer.TranscriptEvent) async {
        if Date() < suppressASRPlaybackUntil {
            print("[PathB] suppressed (post-button debounce): \"\(event.text)\"")
            return
        }
        print("[PathB] ASR transcript: \"\(event.text)\" final=\(event.isFinal)")
        spokenText = event.text
        recogniser.liveText = event.text

        // Translate gate — end-of-utterance only. The previous "mid-sentence
        // incremental" path fired translate on every partial that grew enough,
        // producing overlapping playbacks (the same prefix re-signed each time
        // the transcript advanced — `WHAT IN` was queued four times in a row
        // for "What is happening today"). Rules now:
        //   - `final=true`                                 → translate
        //   - non-final but text held ≥1.0 s unchanged AND
        //     differs from the last translated string      → translate
        //   - otherwise                                    → skip
        // Avatar lags by ~1 s but signs each utterance exactly once.
        let now = Date()
        let words = event.text.split(whereSeparator: { $0.isWhitespace }).count
        let textChars = event.text.count
        let grewSinceLast = event.text != lastTranslated

        if event.text != lastSeenTranscript {
            lastSeenTranscript = event.text
            firstSeenTranscriptAt = now
        }
        let settledFor = now.timeIntervalSince(firstSeenTranscriptAt)
        let isSettled = !event.isFinal
            && textChars > 0
            && grewSinceLast
            && settledFor >= 1.0
        let shouldTranslate = event.isFinal || isSettled
        guard shouldTranslate else {
            print("[PathB] gate skip: words=\(words) chars=\(textChars) " +
                  "final=\(event.isFinal) grew=\(grewSinceLast) " +
                  "settledFor=\(String(format: "%.2f", settledFor))s")
            return
        }
        if !event.isFinal {
            print("[PathB] gate fire (settled non-final): settledFor=\(String(format: "%.2f", settledFor))s words=\(words)")
        }
        lastTranslated = event.text
        if !event.isFinal { lastNonFinalTranslateAt = now }

        let glosses: [String]
        do {
            let t0 = Date()
            glosses = try await translator.translate(event.text, task: .englishToKSL)
            let dt = Int(Date().timeIntervalSince(t0) * 1000)
            print("[PathB] translate (\(words) words → \(glosses.count) glosses, " +
                  "\(dt) ms): \(glosses)")
        } catch {
            print("[PathB] translator FAILED: \(error)")
            errorMessage = "translator: \(error)"
            return
        }

        let (playable, missed) = await resolvePlayableGlossesWithMisses(glosses)
        print("[PathB] playable: \(glosses.count) → \(playable.count) (missed \(missed.count)): playable=\(playable) missed=\(missed)")
        if !missed.isEmpty {
            // Surface partial misses as a non-fatal hint so the user knows
            // why parts of their sentence didn't animate. Clears on the
            // next fully-playable utterance.
            // let label = missed.count == 1 ? "Not in library" : "Not in library"
            // errorMessage = "\(label): \(missed.joined(separator: ", "))"
        } else {
            errorMessage = nil
        }
        guard !playable.isEmpty else {
            errorMessage = "No playable glosses in pose library for \"\(event.text)\" → \(glosses.joined(separator: " "))"
            return
        }

        glossStream = playable
        guard let stitcher = stitcher else { return }
        // ASR-driven path: same cancellation discipline. A new transcript
        // supersedes the previous one's playback rather than queueing.
        startPlayback(glosses: playable, stitcher: stitcher)
    }

    /// Like `resolvePlayableGlosses` but also returns the list of input
    /// tokens that couldn't be resolved (no library entry, no fuzzy mapping,
    /// no fingerspell available). UI uses the miss list to tell the user
    /// what didn't animate so they're not staring at a partially-frozen
    /// avatar wondering why.
    private func resolvePlayableGlossesWithMisses(
        _ glosses: [String]
    ) async -> (playable: [String], missed: [String]) {
        var playable: [String] = []
        var missed: [String] = []
        guard let db = database else { return (glosses, []) }

        var normalized: [String] = []
        normalized.reserveCapacity(glosses.count * 2)
        for raw in glosses {
            for tok in normalizeGlossToken(raw) {
                normalized.append(tok)
            }
        }
        playable.reserveCapacity(normalized.count)
        for tok in normalized {
            if await db.contains(tok) {
                if playable.last != tok {
                    playable.append(tok)
                }
            } else if let mapped = await mapUnknownGlossToDataset(tok, db: db) {
                print("[PathB] gloss miss '\(tok)' -> mapped '\(mapped)'")
                if playable.last != mapped { playable.append(mapped) }
            } else {
                let fs = await spellTokenIfPossible(tok, db: db)
                if !fs.isEmpty {
                    print("[PathB] gloss miss '\(tok)' -> fingerspell \(fs.joined(separator: " "))")
                    for s in fs where playable.last != s {
                        playable.append(s)
                    }
                } else {
                    print("[PathB] gloss miss '\(tok)' (no fallback available)")
                    missed.append(tok)
                }
            }
        }
        return (playable, missed)
    }

    /// Resolve translator output into tokens guaranteed to exist in the
    /// bundled PoseLibrary. This completes the speech->avatar contract:
    /// only playable glosses are passed into the stitcher.
    private func resolvePlayableGlosses(_ glosses: [String]) async -> [String] {
        guard let db = database else { return glosses }

        var normalized: [String] = []
        normalized.reserveCapacity(glosses.count * 2)
        for raw in glosses {
            for tok in normalizeGlossToken(raw) {
                normalized.append(tok)
            }
        }

        var playable: [String] = []
        playable.reserveCapacity(normalized.count)
        for tok in normalized {
            if await db.contains(tok) {
                // Drop immediate duplicates from ASR/Gemma repetition.
                if playable.last != tok {
                    playable.append(tok)
                }
            } else {
                if let mapped = await mapUnknownGlossToDataset(tok, db: db) {
                    print("[PathB] gloss miss '\(tok)' -> mapped '\(mapped)'")
                    if playable.last != mapped {
                        playable.append(mapped)
                    }
                    continue
                }
                let fs = await spellTokenIfPossible(tok, db: db)
                if !fs.isEmpty {
                    print("[PathB] gloss miss '\(tok)' -> fingerspell \(fs.joined(separator: " "))")
                    for s in fs {
                        if playable.last != s {
                            playable.append(s)
                        }
                    }
                } else {
                    print("[PathB] gloss miss '\(tok)' (no fingerspell clips available)")
                }
            }
        }
        if !playable.isEmpty {
            print("[PathB] playable glosses: \(playable.joined(separator: " "))")
        }
        return playable
    }

    /// Try to map unknown Gemma glosses to the closest dataset gloss token.
    private func mapUnknownGlossToDataset(_ token: String, db: PoseDatabase) async -> String? {
        if await db.contains(token) { return token }
        if let exactNorm = poseGlossByNormalized[normalizeForLookup(token)] {
            return exactNorm
        }
        let clean = normalizeForLookup(token)
        guard !clean.isEmpty, !poseGlossTokens.isEmpty else { return nil }

        // Cheap candidate pruning for edit-distance scan.
        let first = clean.first
        let candidates = poseGlossTokens.filter { t in
            let n = normalizeForLookup(t)
            if n.isEmpty { return false }
            if let f = first, n.first != f { return false }
            return abs(n.count - clean.count) <= 2
        }
        var best: (tok: String, dist: Int)? = nil
        for c in candidates {
            let d = editDistance(clean, normalizeForLookup(c))
            if d <= 2, (best == nil || d < best!.dist) {
                best = (c, d)
            }
        }
        return best?.tok
    }

    /// Expand an unknown gloss token into fingerspelling clips, one token per
    /// character, if those clips exist in PoseLibrary.
    private func spellTokenIfPossible(_ token: String, db: PoseDatabase) async -> [String] {
        let chars = Array(token).filter { !$0.isWhitespace }
        guard !chars.isEmpty else { return [] }

        var out: [String] = []
        out.reserveCapacity(chars.count)
        for ch in chars {
            guard let fs = await fingerspellClipToken(for: ch, db: db) else {
                return []
            }
            out.append(fs)
        }
        return out
    }

    /// Resolve one character to a PoseLibrary token.
    private func fingerspellClipToken(for rawChar: Character, db: PoseDatabase) async -> String? {
        let c = Character(String(rawChar).uppercased())
        if let cached = fingerspellTokenByChar[c] {
            return cached
        }

        // Try common naming conventions; whichever exists in PoseLibrary wins.
        let key = String(c)
        let candidates = [
            key,                    // A
            "FS_\(key)",          // FS_A
            "LETTER_\(key)",      // LETTER_A
            "ALPHABET_\(key)",    // ALPHABET_A
            "\(key)_FS",          // A_FS
        ]
        for cand in candidates {
            if await db.contains(cand) {
                fingerspellTokenByChar[c] = cand
                return cand
            }
        }
        return nil
    }

    /// Startup diagnostic: report available A-Z/0-9 fingerspelling clips.
    private func fingerspellCoverageSummary(db: PoseDatabase) async -> String {
        let inventory = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        var available: [Character] = []
        var missing: [Character] = []
        available.reserveCapacity(inventory.count)
        missing.reserveCapacity(inventory.count)

        for ch in inventory {
            if await fingerspellClipToken(for: ch, db: db) != nil {
                available.append(ch)
            } else {
                missing.append(ch)
            }
        }
        return "available=\(available.count)/\(inventory.count) [\(String(available))] missing=[\(String(missing))]"
    }

    /// Canonicalize one translator token into one-or-more lookup candidates.
    /// Examples:
    ///   "tomorrow//" -> ["TOMORROW"]
    ///   "i-go"       -> ["I-GO", "I", "GO"]
    private func normalizeGlossToken(_ raw: String) -> [String] {
        let upper = raw.uppercased()
        let trimmed = upper.trimmingCharacters(in: glossSplitChars)
        if trimmed.isEmpty { return [] }

        // Remove trailing punctuation patterns used in dataset glosses.
        let cleaned = trimmed
            .replacingOccurrences(of: "//", with: "")
            .replacingOccurrences(of: "?", with: "")
            .replacingOccurrences(of: "!", with: "")
            .replacingOccurrences(of: ".", with: "")
            .replacingOccurrences(of: ",", with: "")

        if cleaned.isEmpty { return [] }

        // Keep the original cleaned token first (e.g. I-GO), then optional splits.
        var out = [cleaned]
        if cleaned.contains("-") {
            let parts = cleaned.split(separator: "-").map(String.init).filter { !$0.isEmpty }
            out.append(contentsOf: parts)
        }
        return out
    }

    private func normalizeForLookup(_ s: String) -> String {
        s.uppercased().filter { $0.isLetter || $0.isNumber }
    }

    private func editDistance(_ a: String, _ b: String) -> Int {
        let ac = Array(a), bc = Array(b)
        if ac.isEmpty { return bc.count }
        if bc.isEmpty { return ac.count }
        var prev = Array(0...bc.count)
        for i in 1...ac.count {
            var cur = Array(repeating: 0, count: bc.count + 1)
            cur[0] = i
            for j in 1...bc.count {
                let cost = ac[i - 1] == bc[j - 1] ? 0 : 1
                cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            }
            prev = cur
        }
        return prev[bc.count]
    }
}
