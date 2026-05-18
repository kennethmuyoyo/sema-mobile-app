import CoreMedia
import CoreVideo
import Foundation
import Observation

/// Wires camera frames → HolisticLandmarker → FrameRing → GlossTagger →
/// StreamingCTCDecoder, exposes a stream of `GlossToken`s.
///
/// One inference per `stride` frames once the ring is full. Resets the
/// streaming decoder when no hand has been detected for `noHandTimeout`.
@MainActor
@Observable
final class PathACoordinator {
    enum State: String {
        case idle = "Idle"
        case warmingUp = "Preparing…"
        case ready = "Ready"
        case running = "Listening"
        case failed = "Error"
    }

    var state: State = .idle
    var errorMessage: String? = nil
    /// Most recently emitted gloss tokens, newest last.
    var emittedTokens: [GlossToken] = []
    /// Live transcript built from emittedTokens labels.
    var transcript: String { emittedTokens.map(\.label).joined(separator: " ") }
    /// Finalized gloss phrase emitted when a signing burst ends
    /// (no-hand timeout). Consumed by sign->speech.
    var finalizedGlossPhrase: String = ""

    /// Most recent landmark frame from MediaPipe (45 joints × xyz, with a
    /// sibling per-joint mask for the overlay). SwiftUI overlay reads this
    /// to draw the skeleton on the camera PiP.
    var latestFrame: NormalizedFrame?

    /// Top-3 gloss candidates from the latest inference window, sorted by
    /// softmax peak. Independent of the CTC emission threshold — meant for
    /// the debug strip so you can see what the model is leaning towards
    /// even when nothing crosses the confidence bar.
    var topPredictions: [GlossToken] = []

    /// Stride: run inference once every this many landmark frames. At 24 fps
    /// (CameraSessionController pins the capture rate to match training) a
    /// stride of 4 = ~167 ms between inferences, which keeps the debug-strip
    /// probabilities visibly tracking the user's hand motion.
    private let stride: Int = 4
    /// Reset decoder when there have been no actively-signing frames for
    /// this long. "Actively signing" is defined by `isFrameActivelySigning`:
    /// at least one wrist must be raised into the signing space — hands
    /// resting at the user's sides don't count, even though MediaPipe still
    /// sees them.
    private let noHandTimeout: TimeInterval = 2.0
    /// A wrist counts as "in the signing space" when its y (in shoulder-
    /// normalised units, image-down) is below this threshold. Shoulders sit
    /// at y≈0, hips at y≈+1.5; 1.0 puts the boundary midway between, so a
    /// wrist at chest level or higher passes and a wrist hanging by the
    /// side fails. Tunable if signers tend to hold signs lower.
    private let signingWristYMax: Float = 1.0
    /// Minimum EMA-smoothed hand-joint velocity (mean per-finger 3-D
    /// displacement, in shoulder-widths per frame) below which we treat
    /// the frame as at-rest even when wrists are raised. MediaPipe jitter
    /// on stationary hands sits around 0.005–0.010; a moving sign is
    /// 0.02–0.05. Dropped from 0.012 to 0.005 because the per-frame
    /// motion was reading 0.0000 when MediaPipe carry-forwarded wrists
    /// (identical positions across consecutive frames → no motion). The
    /// looser bar lets gloss inference fire on slow / held signs.
    private let signingMotionMin: Float = 0.005
    /// EMA weight on the new frame's instantaneous motion. α=0.3 ≈ 3-frame
    /// half-life of decay — keeps the signal "warm" for ~120 ms after motion
    /// stops so brief mid-sign holds don't drop you out of signing state.
    private let motionEMAAlpha: Float = 0.3
    /// Floor for the top-K emission path. Calibrated for the **mean-over-time**
    /// metric in `topKCandidates`: with CTC, an actual sign typically averages
    /// 0.02–0.05 softmax-prob across the 128-frame window (high on the few
    /// frames at the sign boundary, near-zero elsewhere). 0.02 is permissive
    /// so we surface signal while we iterate. Cooldown + last-label dedup
    /// still prevent spam during a held sign.
    private let minTokenConfidence: Float = 0.02
    /// Emit at most one gloss every cooldown window. 0.35 s is fast enough
    /// to keep up with continuous signing without re-emitting the same
    /// gloss mid-hold (which the duplicate-suppression in the emit step
    /// already handles).
    private let emissionCooldown: TimeInterval = 0.35
    /// Safety cap per signing burst (until no-hand reset). Bumped from 3
    /// so longer phrases don't dead-end after the third gloss.
    private let maxTokensPerBurst: Int = 8
    /// Debug-only: persist every inference window's input + top-K to
    /// Documents/landmark_dumps/ so we can replay it offline with
    /// recognition/v3_infer.py. Flip off for shipped builds.
    private let dumpLandmarksForDebug: Bool = false
    /// Debug-only comparison against PoseLibrary inventory.
    private let enablePoseLibraryComparisonLogs = false
    private let compareLogMinConfidence: Float = 0.05
    private let compareLogCooldown: TimeInterval = 2.0

    @ObservationIgnored private var landmarker: HolisticLandmarker?
    /// CoreML v11 phonological recognizer. Loaded from `ksl_model.mlpackage`
    /// + `ksl_model.metadata.json` in Resources/. Previously TFLite, but
    /// `MediaPipeTasksVision` bundles its own TFLite runtime and force-loads
    /// its symbols — a second TFLite framework produced duplicate-symbol
    /// linker errors. CoreML shares no libraries with MediaPipe.
    @ObservationIgnored private var tagger: CoreMLGlossTagger?
    @ObservationIgnored private var ring: FrameRing?
    @ObservationIgnored private var decoder: StreamingCTCDecoder?
    /// Optional gloss allowlist loaded from Resources/demo_recognition.json.
    /// When non-nil and non-empty, `topKCandidates` only ranks tokens whose
    /// label is in this set and re-normalises their probabilities to sum to 1.
    @ObservationIgnored private var demoAllowlist: Set<String>?
    /// Optional deterministic fallback recognizer. Compares the live window
    /// against bundled PoseLibrary clips for each allowed gloss. When non-nil
    /// AND it returns a hit, its result overrides the trained tagger's output
    /// — used to keep the demo working while the recognizer is uncertain on
    /// the iOS input distribution. Loaded only if both `demoAllowlist` and
    /// `poseDB` are available at bootstrap.
    @ObservationIgnored private var templateMatcher: PoseTemplateMatcher?
    /// Optional comparison DB (the bundled pose library built from our dataset).
    /// If present, decoded glosses are checked against this inventory so we can
    /// confirm MediaPipe->GlossTagger outputs map to known data tokens.
    @ObservationIgnored private var poseDB: PoseDatabase?
    @ObservationIgnored private var lastCompareLogAt: [String: TimeInterval] = [:]

    @ObservationIgnored private var framesSinceLastInference: Int = 0
    @ObservationIgnored private var lastHandSeen: TimeInterval = 0
    /// Throttle for the per-frame gate-state log line in ingest(). Only
    /// prints once per second so the console stays readable while still
    /// surfacing whether camera frames are arriving and the signing gate
    /// is letting them through.
    @ObservationIgnored private var lastGateLogAt: TimeInterval = -1e9
    /// Sibling throttle for the per-frame motion-internals diagnostic.
    @ObservationIgnored private var lastMotionLogAt: TimeInterval = -1e9
    @ObservationIgnored private var inferenceTask: Task<Void, Never>? = nil
    @ObservationIgnored private var lastEmissionAt: TimeInterval = -1e9
    @ObservationIgnored private var emittedInBurst: Int = 0
    /// Previous frame snapshot used to compute hand-joint velocity. Cleared
    /// on burst finalize so the first frame after re-engagement isn't
    /// compared against a stale pose.
    @ObservationIgnored private var prevFrameForMotion: NormalizedFrame?
    /// EMA of recent hand-joint motion magnitude (shoulder-widths/frame).
    @ObservationIgnored private var smoothedHandMotion: Float = 0

    /// Marks the path ready without loading MediaPipe or the gloss tagger (previews).
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
            do {
                let lm = try HolisticLandmarker()
                let gt = try CoreMLGlossTagger()
                let cfg = await gt.configuration
                let ring = FrameRing(capacity: cfg.inputSeqLen)
                // Built but currently bypassed in the emission path (see ingest()
                // for the top-K based emission); kept warm so we can swap back
                // once the recognizer is reliable enough for stable-prefix gating.
                let decoder = StreamingCTCDecoder(
                    vocab: cfg.metadata.glossNameToId,
                    historyDepth: 2)
                try await gt.prewarm()
                let poseDB = try? PoseDatabase()
                // Demo allowlist and the dependent PoseTemplateMatcher are
                // disabled — the recognizer now uses its full 4145-vocab top-K
                // directly. Reload `demo_recognition.json` and reinstantiate
                // the matcher here if you need to gate down to the 13-token
                // demo set again.
                await MainActor.run {
                    guard let self else { return }
                    self.landmarker = lm
                    self.tagger = gt
                    self.ring = ring
                    self.decoder = decoder
                    self.poseDB = poseDB
                    self.demoAllowlist = nil
                    self.templateMatcher = nil
                    if poseDB == nil {
                        print("[PathA] PoseDatabase unavailable; skipping token-vs-dataset comparison")
                    } else if !self.enablePoseLibraryComparisonLogs {
                        print("[PathA] token-vs-dataset comparison available (logging disabled)")
                    } else {
                        print("[PathA] token comparison enabled (MediaPipe -> CoreMLGlossTagger -> PoseLibrary)")
                    }
                    self.state = .ready
                }
            } catch {
                await MainActor.run {
                    guard let self else { return }
                    self.state = .failed
                    self.errorMessage = "\(error)"
                }
            }
        }
    }

    func start() {
        guard state == .ready || state == .running else { return }
        state = .running
    }

    /// Top-k vocab entries from one inference window, ranked by their **mean**
    /// softmax probability across time. Blank/unk and `<...>` specials are
    /// skipped so the debug strip shows meaningful glosses.
    ///
    /// Why mean and not max: with CTC the model is sharply confident on a
    /// handful of frames per sign and outputs `<blank>` everywhere else. Max-
    /// over-time scoring is brittle — a single spurious 99% peak on a random
    /// token (common with out-of-distribution iOS inputs, e.g. heavy MediaPipe
    /// hand dropout) dominates the ranking. Mean-over-time rewards tokens that
    /// the model believes in over a span of frames, which is what a true sign
    /// produces.
    static func topKCandidates(
        logits: [Float],
        seqLen: Int,
        vocab: [String: Int],
        k: Int,
        timestamp: TimeInterval,
        allowlist: Set<String>? = nil
    ) -> [GlossToken] {
        let vocabSize = vocab.count
        guard logits.count == seqLen * vocabSize, vocabSize > 0, k > 0 else { return [] }

        var labels = [String](repeating: "<unk>", count: vocabSize)
        for (label, id) in vocab where id >= 0 && id < vocabSize {
            labels[id] = label
        }

        // Mean softmax-prob per vocab id across all time steps.
        var sumProb = [Float](repeating: 0, count: vocabSize)
        for t in 0..<seqLen {
            let base = t * vocabSize
            var rowMax: Float = -.infinity
            for v in 0..<vocabSize where logits[base + v] > rowMax { rowMax = logits[base + v] }
            var sumExp: Float = 0
            for v in 0..<vocabSize { sumExp += exp(logits[base + v] - rowMax) }
            for v in 0..<vocabSize {
                sumProb[v] += exp(logits[base + v] - rowMax) / sumExp
            }
        }
        let inv = 1.0 / Float(seqLen)
        var bestProb = [Float](repeating: 0, count: vocabSize)
        for v in 0..<vocabSize { bestProb[v] = sumProb[v] * inv }

        struct Candidate { var id: Int; var prob: Float }
        var pool: [Candidate] = []
        pool.reserveCapacity(vocabSize)
        var allowedMass: Float = 0
        for v in 0..<vocabSize {
            if v == StreamingCTCDecoder.blankID { continue }
            let label = labels[v]
            if label.hasPrefix("<") { continue }   // skips <unk>, <blank>, etc.
            if let allowed = allowlist, !allowed.contains(label) { continue }
            pool.append(Candidate(id: v, prob: bestProb[v]))
            allowedMass += bestProb[v]
        }
        // When an allowlist is in effect, renormalise so the surviving
        // tokens' probabilities sum to ~1. This makes the demo UI readable
        // (a clear winner shows ~0.7 instead of ~0.03 of total softmax mass).
        if allowlist != nil, allowedMass > 0 {
            let inv = 1.0 / allowedMass
            for i in 0..<pool.count { pool[i].prob *= inv }
        }
        pool.sort { $0.prob > $1.prob }
        return pool.prefix(k).map { c in
            GlossToken(id: c.id, label: labels[c.id], timestamp: timestamp, confidence: c.prob)
        }
    }

    /// Window-shape variant of `topKCandidates` for the v11 LiteRT recognizer:
    /// `logits` is `(vocab,)` — a single per-window prediction, no time axis to
    /// average over. Otherwise applies the same softmax / blank+special-token
    /// filtering / allowlist renormalisation as the time-axis version.
    static func topKFromWindowLogits(
        logits: [Float],
        vocab: [String: Int],
        k: Int,
        timestamp: TimeInterval,
        allowlist: Set<String>? = nil
    ) -> [GlossToken] {
        let vocabSize = vocab.count
        guard logits.count == vocabSize, vocabSize > 0, k > 0 else { return [] }

        var labels = [String](repeating: "<unk>", count: vocabSize)
        for (label, id) in vocab where id >= 0 && id < vocabSize {
            labels[id] = label
        }

        // Softmax once. Subtract row-max for numerical stability.
        var rowMax: Float = -.infinity
        for v in 0..<vocabSize where logits[v] > rowMax { rowMax = logits[v] }
        var sumExp: Float = 0
        var prob = [Float](repeating: 0, count: vocabSize)
        for v in 0..<vocabSize {
            let e = exp(logits[v] - rowMax)
            prob[v] = e
            sumExp += e
        }
        let invSum = 1.0 / sumExp
        for v in 0..<vocabSize { prob[v] *= invSum }

        struct Candidate { var id: Int; var prob: Float }
        var pool: [Candidate] = []
        pool.reserveCapacity(vocabSize)
        var allowedMass: Float = 0
        for v in 0..<vocabSize {
            if v == StreamingCTCDecoder.blankID { continue }
            let label = labels[v]
            if label.hasPrefix("<") { continue }
            if let allowed = allowlist, !allowed.contains(label) { continue }
            pool.append(Candidate(id: v, prob: prob[v]))
            allowedMass += prob[v]
        }
        if allowlist != nil, allowedMass > 0 {
            let inv = 1.0 / allowedMass
            for i in 0..<pool.count { pool[i].prob *= inv }
        }
        pool.sort { $0.prob > $1.prob }
        return pool.prefix(k).map { c in
            GlossToken(id: c.id, label: labels[c.id], timestamp: timestamp, confidence: c.prob)
        }
    }

    /// True if MediaPipe detected (or the carry-forward kept alive) at least
    /// one finger joint of either hand on this frame. Used to gate ring.push
    /// so the inference window stays in the same distribution as the training
    /// segments, which were filtered by `visibility_bursts` in the notebook
    /// (only kept frames where ≥30 % of one hand was visible). Without this
    /// gate, all-body-no-hands frames leak in, and the model's input
    /// `LayerNorm(135)` blows up the activations on the 120 zero features.
    ///
    /// Hand-finger joints live at indices 15..29 (left) and 30..44 (right).
    /// `mask > 0` includes both real detections (1.0) and recent
    /// carry-forwards (0.5) — both have plausible values for the LayerNorm.
    private static func frameHasAnyHand(_ frame: NormalizedFrame) -> Bool {
        for j in 15..<Landmark45.count {
            if frame.mask[j] > 0 { return true }
        }
        return false
    }

    /// Loads `Resources/demo_recognition.json` (if present) into a Set of
    /// allowed gloss labels. Returns `nil` when the file is missing, malformed,
    /// or has an empty `glosses` array — `topKCandidates` then ranks the full
    /// vocab as before.
    private static func loadDemoAllowlist() -> Set<String>? {
        guard let url = Bundle.main.url(forResource: "demo_recognition", withExtension: "json"),
              let data = try? Data(contentsOf: url) else {
            print("[PathA] demo_recognition.json not bundled; allowlist disabled")
            return nil
        }
        struct Manifest: Decodable { let glosses: [String] }
        guard let decoded = try? JSONDecoder().decode(Manifest.self, from: data) else {
            print("[PathA] demo_recognition.json malformed; allowlist disabled")
            return nil
        }
        let set = Set(decoded.glosses)
        if set.isEmpty {
            print("[PathA] demo_recognition.json present but empty; allowlist disabled")
            return nil
        }
        print("[PathA] demo_recognition allowlist active: \(set.sorted().joined(separator: ", "))")
        return set
    }

    func pause() {
        guard state == .running else { return }
        state = .ready
        inferenceTask?.cancel()
        inferenceTask = nil
        framesSinceLastInference = 0
        // Reset the local per-frame motion buffer; HolisticLandmarker's own
        // carry-forward buffers are wiped when the orchestrator re-enters
        // watching mode (see `resetLandmarkerForNewSession`).
        prevFrameForMotion = nil
        smoothedHandMotion = 0
        // Mirror the no-hand-timeout finalize: when the camera goes away
        // (user switched to listening, or tapped Stop) commit the current
        // burst as a finalized phrase and clear it. Otherwise stale tokens
        // from this watching session leak into the next mode's finalize and
        // show up as a phantom recognition the user never made.
        if !emittedTokens.isEmpty {
            finalizedGlossPhrase = emittedTokens.map(\.label).joined(separator: " ")
            emittedTokens.removeAll(keepingCapacity: true)
            emittedInBurst = 0
            lastEmissionAt = -1e9
        }
    }

    /// Clear MediaPipe's carry-forward + last-detection state so the next
    /// watching session starts from scratch. Called by the orchestrator
    /// during `enterWatchingMode` — without this, the first frame after the
    /// camera comes back uses `lastNormalized` from the previous session.
    func resetLandmarkerForNewSession() async {
        await landmarker?.reset()
    }

    /// True iff at least one wrist is detected AND raised into the signing
    /// space (y above `signingWristYMax`, where y is image-down in shoulder-
    /// normalised units: shoulders at y≈0, hips at y≈+1.5). Both wrists
    /// hanging at the user's sides → false → recognizer treats the frame
    /// like a no-hand frame: inference is skipped and the inactivity
    /// timeout starts running.
    private func isFrameActivelySigning(_ frame: NormalizedFrame) -> Bool {
        let lIdx = Landmark45.index(of: "left_wrist")
        let rIdx = Landmark45.index(of: "right_wrist")
        // NormalizedFrame.values is stride 3 (x, y, z per joint). The previous
        // `* 4` reach was reading the WRONG joint's coordinate — for left_wrist
        // (idx 7) it landed on left_hip's z, making `raised` essentially random.
        // Mask > 0 accepts both real (1.0) and carry-forward (0.5) wrists; a
        // wrist that briefly drops out of body-landmarker detection shouldn't
        // make us think the user lowered their hand.
        let lActive = frame.mask[lIdx] > 0
            && frame.values[lIdx * 3 + 1] < signingWristYMax
        let rActive = frame.mask[rIdx] > 0
            && frame.values[rIdx * 3 + 1] < signingWristYMax
        return lActive || rActive
    }

    /// Updates and returns the EMA-smoothed mean 3-D displacement of hand
    /// joints (indices 15..44 — 30 finger landmarks across both hands)
    /// between this frame and the previous one. Stationary hands (even
    /// raised) hover around 0.005–0.010 from MediaPipe jitter; a sign in
    /// motion drives the value up to 0.02–0.05. Compared against
    /// `signingMotionMin` to drop spurious "I'm thinking" frames from
    /// inference.
    private func updateAndGetHandMotion(_ current: NormalizedFrame) -> Float {
        defer { prevFrameForMotion = current }
        guard let prev = prevFrameForMotion else { return smoothedHandMotion }
        // Include every joint that has *some* value on both frames (real or
        // carry-forward); skip only pure-carry-forward pairs (exactly zero
        // displacement). Real motion always produces a non-zero displacement
        // because MediaPipe adds at least sub-mm jitter on every detection.
        var sum: Float = 0
        var count: Int = 0
        var maxDisp: Float = 0
        let lwIdx = Landmark45.index(of: "left_wrist")
        let rwIdx = Landmark45.index(of: "right_wrist")
        var joints = Array(15..<Landmark45.count)
        joints.append(lwIdx)
        joints.append(rwIdx)
        for j in joints {
            guard current.mask[j] > 0, prev.mask[j] > 0 else { continue }
            let base = j * 3
            let dx = current.values[base]     - prev.values[base]
            let dy = current.values[base + 1] - prev.values[base + 1]
            let dz = current.values[base + 2] - prev.values[base + 2]
            let d2 = dx * dx + dy * dy + dz * dz
            if d2 == 0 { continue }
            let d = d2.squareRoot()
            sum += d
            count += 1
            if d > maxDisp { maxDisp = d }
        }
        let inst: Float = count > 0 ? sum / Float(count) : 0
        smoothedHandMotion = motionEMAAlpha * inst
            + (1 - motionEMAAlpha) * smoothedHandMotion

        // One-shot diagnostic (~1 Hz, matched to the gate log) so we can see
        // whether real displacement is flowing through. If the gate log
        // reads `motion=0.0000` while this prints `maxDisp=0.0000 count=0`,
        // MediaPipe is returning identical poses across frames (camera-buffer
        // issue). If `count>0 maxDisp>0` but the gate still reads 0.0000,
        // the EMA is being reset elsewhere.
        let nowSec = current.timestamp
        if nowSec - lastMotionLogAt >= 1.0 {
            lastMotionLogAt = nowSec
            let lwDx = current.mask[lwIdx] > 0 && prev.mask[lwIdx] > 0
                ? abs(current.values[lwIdx * 3]     - prev.values[lwIdx * 3])
                : Float.nan
            let lwDy = current.mask[lwIdx] > 0 && prev.mask[lwIdx] > 0
                ? abs(current.values[lwIdx * 3 + 1] - prev.values[lwIdx * 3 + 1])
                : Float.nan
            print("[Motion] count=\(count) inst=\(String(format: "%.4f", inst)) max=\(String(format: "%.4f", maxDisp)) smoothed=\(String(format: "%.4f", smoothedHandMotion)) lw(dx=\(String(format: "%.4f", lwDx)) dy=\(String(format: "%.4f", lwDy)) maskC=\(current.mask[lwIdx]) maskP=\(prev.mask[lwIdx]))")
        }
        return smoothedHandMotion
    }
}

extension PathACoordinator: CameraFrameDelegate {
    nonisolated func camera(_ controller: CameraSessionController,
                            didProduce pixelBuffer: CVPixelBuffer,
                            at presentationTime: CMTime) {
        Task { @MainActor [weak self] in
            await self?.ingest(pixelBuffer: pixelBuffer, presentationTime: presentationTime)
        }
    }

    private func ingest(pixelBuffer: CVPixelBuffer, presentationTime: CMTime) async {
        guard state == .running,
              let lm = landmarker,
              let ring = ring,
              let decoder = decoder,
              let tagger = tagger
        else { return }

        let timestampMs = max(0, Int(CMTimeGetSeconds(presentationTime) * 1000))
        let frame: NormalizedFrame?
        do {
            frame = try await lm.process(pixelBuffer: pixelBuffer, timestampMs: timestampMs)
        } catch {
            errorMessage = "landmark error: \(error)"
            return
        }

        // Signing-activity gate. Two conjoined signals:
        //   • `raised`        — at least one wrist body landmark is in the
        //                       signing space (y < signingWristYMax).
        //   • `handPresent`   — at least one of the 30 finger-joint slots
        //                       has mask > 0 (real detection or recent
        //                       carry-forward).
        // Wrists alone aren't enough: the body landmarker keeps reporting
        // wrist positions even when the user's arms hang at their sides, so
        // a wrist-only check left the gate stuck at YES across the entire
        // post-signing rest period. Requiring `handPresent` flips the gate
        // to NO the moment MediaPipe's hand landmarker stops seeing fingers
        // AND the 30-frame carry-forward expires (~1.25 s). After 2 s of
        // !isSigning the no-hand timeout fires below and the burst gets
        // finalized — which is exactly what triggers the bridge to speak
        // the recognized sentence.
        let motion: Float = frame.map { updateAndGetHandMotion($0) } ?? 0
        let raised: Bool = frame.map { isFrameActivelySigning($0) } ?? false
        let handPresent: Bool = frame.map { Self.frameHasAnyHand($0) } ?? false
        let isSigning: Bool = raised && handPresent
        let now = TimeInterval(timestampMs) / 1000.0
        // Throttled gate log: print at most once per second so we can see
        // whether the signing gate is letting frames through. If you see
        // gate=NO for every line, the recogniser will never fire. The
        // `cause=` tag tells you WHY — `no-pose` means MediaPipe didn't
        // detect a body (point the camera at the user), `low-motion`
        // means hands are present but holding still, `wrists-down` means
        // arms are at the user's sides. Tune `signingMotionMin` /
        // `signingWristYMax` downward if signs are being missed.
        if now - lastGateLogAt >= 1.0 {
            lastGateLogAt = now
            let cause: String
            if frame == nil {
                cause = "no-pose"
            } else if !handPresent {
                cause = "no-hand"
            } else if !raised {
                cause = "wrists-down"
            } else if motion < signingMotionMin {
                // Informational only — gate no longer blocks on motion.
                cause = "ok(low-motion)"
            } else {
                cause = "ok"
            }
            // Mask coverage: `real` is the count of joints actually detected by
            // MediaPipe this frame (mask >= 1.0); `carry` is the count of
            // joints filled in by HolisticLandmarker's carry-forward (mask =
            // 0.5). `hand` shows whether at least one finger joint is present
            // (real or carry) — frames without this aren't pushed to the
            // inference ring, matching the training-time visibility-bursts
            // filter.
            let realCount: Int  = frame.map { $0.mask.reduce(0) { $0 + ($1 >= 1.0 ? 1 : 0) } } ?? 0
            let carryCount: Int = frame.map { $0.mask.reduce(0) { $0 + (($1 > 0 && $1 < 1.0) ? 1 : 0) } } ?? 0
            let handPresent: Bool = frame.map { Self.frameHasAnyHand($0) } ?? false
            print("[PathA] gate t=\(String(format: "%.1f", now)) raised=\(raised) motion=\(String(format: "%.4f", motion)) (min=\(signingMotionMin)) real=\(realCount)/45 carry=\(carryCount) hand=\(handPresent ? "Y" : "N") gate=\(isSigning ? "YES" : "NO") cause=\(cause)")
        }
        if isSigning {
            lastHandSeen = now
        } else if now - lastHandSeen > noHandTimeout {
            // Finalize current burst before clearing.
            if !emittedTokens.isEmpty {
                finalizedGlossPhrase = emittedTokens.map(\.label).joined(separator: " ")
            }
            decoder.reset()
            await ring.reset()
            framesSinceLastInference = 0
            emittedInBurst = 0
            lastEmissionAt = -1e9
            // End of signing burst -> clear transcript so next gesture starts fresh.
            emittedTokens.removeAll(keepingCapacity: true)
            // Reset motion tracking so the next session starts cold.
            prevFrameForMotion = nil
            smoothedHandMotion = 0
        }

        guard let frame else { return }
        latestFrame = frame
        // Match the training-time `visibility_bursts` filter: only push frames
        // that have at least one detected hand into the inference ring. The
        // notebook (cell 7) excluded all-body-no-hands frames from training
        // segments, so the model's per-frame LayerNorm(135) has never seen
        // 120 zeros + 15 body coords — when it does, the frame variance
        // collapses and logits explode (we observed lmax ≈ 60000+). Skipping
        // these frames keeps the ring filled with the same distribution the
        // model was trained on.
        if Self.frameHasAnyHand(frame) {
            await ring.push(frame)
            framesSinceLastInference += 1
        }

        // Slide: when the ring is full and stride threshold reached, run
        // inference — but only if the user is currently in the signing
        // space. Arms-at-rest frames don't trigger emissions; the ring
        // keeps filling in case they raise their hands again so context is
        // still available the moment signing resumes.
        let isFull = await ring.isFull
        guard isFull, framesSinceLastInference >= stride, isSigning else { return }
        framesSinceLastInference = 0

        // Inference + decode runs serialised — drop frames rather than queue.
        guard inferenceTask == nil || inferenceTask?.isCancelled == true else { return }
        inferenceTask = Task { [weak self] in
            guard let self else { return }
            do {
                guard let snapshot = await ring.snapshot() else { return }
                let flat = snapshot.flatMap(\.values)
                // CoreML v11 returns a single window-level prediction
                // `(gloss_logits: (vocab,), aux_indices: (30,))` — no time
                // axis to average over, unlike the v3 CoreML model.
                let inferT0 = Date()
                let (glossLogits, _) = try await tagger.predict(features: flat)
                // Microsecond timing so a fast inference doesn't print "0 ms"
                // and read as "the model didn't run". A real CoreML pass on
                // this (1, 64, 135) model is ~3–20 ms on the Neural Engine.
                let inferUs = Int(Date().timeIntervalSince(inferT0) * 1_000_000)
                let cfg = await tagger.configuration
                // Diagnostics — `lmin / lmax` show the logit spread. With FP32
                // weights both numbers should be in the single-to-low-double-
                // digit range; if they spike into thousands again, the model
                // export has regressed.
                let lmin: Float = glossLogits.min() ?? 0
                let lmax: Float = glossLogits.max() ?? 0
                let taggerTop = Self.topKFromWindowLogits(
                    logits: glossLogits,
                    vocab: cfg.metadata.glossNameToId,
                    k: 3,
                    timestamp: frame.timestamp,
                    allowlist: nil   // full 4145-vocab — no demo restriction
                )
                let topStr = taggerTop.map { "\($0.label)=\(String(format: "%.2f", $0.confidence))" }.joined(separator: " ")
                print("[PathA] infer \(inferUs)μs lmin=\(String(format: "%.2f", lmin)) lmax=\(String(format: "%.2f", lmax)) top3: \(topStr.isEmpty ? "<empty>" : topStr)")
                // Fallback: ask the deterministic pose-template matcher.
                // It returns an empty array when no allowed template is
                // close enough to the live window — in which case we keep
                // the tagger's output. When it does hit, its result is
                // authoritative for the demo.
                var top3 = taggerTop
                var usedMatcher = false
                if let matcher = self.templateMatcher {
                    let matcherResults = await matcher.match(window: snapshot, k: 3)
                    if !matcherResults.isEmpty {
                        usedMatcher = true
                        top3 = matcherResults.map { r in
                            GlossToken(
                                id: cfg.metadata.glossNameToId[r.gloss] ?? -1,
                                label: r.gloss,
                                timestamp: frame.timestamp,
                                confidence: r.confidence
                            )
                        }
                    }
                }
                await MainActor.run { [weak self] in
                    self?.topPredictions = top3
                }
                if self.dumpLandmarksForDebug {
                    let labelPrefix = usedMatcher ? "tmpl:" : "tag:"
                    LandmarkDump.dump(
                        features: flat,
                        inputSeqLen: cfg.inputSeqLen,
                        featureDim: cfg.featureDim,
                        topK: top3.map { .init(label: labelPrefix + $0.label, confidence: $0.confidence) },
                        timestamp: frame.timestamp
                    )
                }
                // Bypass StreamingCTCDecoder for now: emit the strongest top-K
                // candidate directly. The decoder's depth-2 stable-prefix gate
                // was filtering every emission to nothing under flaky landmarks
                // — better to surface noisy recognitions and tune the model
                // than to stay silent. Cooldown + last-label dedup +
                // maxTokensPerBurst still keep the transcript from spamming.
                // topKCandidates already drops <blank>/<unk>/<…>-prefixed.
                let filtered = top3.filter { $0.confidence >= self.minTokenConfidence }
                if let db = self.poseDB, self.enablePoseLibraryComparisonLogs,
                   let best = filtered.max(by: { $0.confidence < $1.confidence }),
                   best.confidence >= self.compareLogMinConfidence {
                    let now = best.timestamp
                    let last = self.lastCompareLogAt[best.label] ?? -1e9
                    if now - last >= self.compareLogCooldown {
                        let hit = await db.contains(best.label)
                        self.lastCompareLogAt[best.label] = now
                        print("[PathA] compare token='\(best.label)' poseLibrary=\(hit ? "hit" : "miss") conf=\(String(format: "%.2f", best.confidence))")
                    }
                }
                if !filtered.isEmpty {
                    await MainActor.run {
                        guard self.emittedInBurst < self.maxTokensPerBurst else { return }
                        let nowTs = frame.timestamp
                        guard nowTs - self.lastEmissionAt >= self.emissionCooldown else { return }

                        // Emit only the strongest token from this inference step.
                        guard let best = filtered.max(by: { $0.confidence < $1.confidence }) else { return }
                        if self.emittedTokens.last?.label == best.label { return }

                        self.emittedTokens.append(best)
                        self.lastEmissionAt = nowTs
                        self.emittedInBurst += 1

                        if self.emittedTokens.count > 6 {
                            self.emittedTokens.removeFirst(self.emittedTokens.count - 6)
                        }
                    }
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = "inference error: \(error)"
                }
            }
            await MainActor.run { self.inferenceTask = nil }
        }
    }
}
