import Foundation

/// Deterministic fallback recognizer for the demo allowlist.
///
/// For each gloss in `demo_recognition.json`, this loads the bundled
/// PoseLibrary template clip and matches the live 128-frame landmark
/// window against it via mean per-joint L2 distance, sliding the template
/// across the window to absorb timing variance. The best-matching template
/// wins, scored as a probability via softmax over negative distances.
///
/// Used by `PathACoordinator` when the trained tagger's absolute confidence
/// is below `minAbsoluteConfidence` — i.e. the CoreML model has no signal
/// on the iOS input distribution. Templates are bundled by
/// `generation/pose_library/build_index.py`.
///
/// Coordinate frame: both template clips and live `NormalizedFrame.values`
/// are in shoulder-mid origin, shoulder-width unit scale, so the L2 distance
/// is in "shoulder-widths" — directly comparable across signs and signers.
actor PoseTemplateMatcher {

    struct Result: Sendable {
        let gloss: String
        let confidence: Float
        /// Raw mean per-joint L2 distance (shoulder-widths). Lower = closer.
        let distance: Float
    }

    private struct Template {
        let gloss: String
        let frameCount: Int
        /// Flat `(T * 45 * 3)` float32 in (t, j, c) C-order.
        let frames: [Float]
    }

    private let templates: [Template]
    /// Softmax temperature for distance → probability. Smaller = sharper.
    /// 0.10 means a 0.20 shoulder-width difference is ~7× in probability.
    private let temperature: Float = 0.10
    /// If the best template is farther than this, return nothing — the
    /// user almost certainly isn't making any of the demo signs right now.
    private let maxAcceptableDistance: Float = 0.6
    /// Slide stride when scanning template across the live window.
    /// 4 frames at 24 fps ≈ 167 ms — same cadence as the tagger inference.
    private let slideStride = 4

    /// Returns nil if no allowlist templates resolved in the bundled PoseLibrary.
    init?(allowlist: Set<String>, poseDB: PoseDatabase) async {
        var loaded: [Template] = []
        for gloss in allowlist.sorted() {
            do {
                guard let clip = try await poseDB.lookup(gloss) else {
                    print("[PoseTemplateMatcher] no PoseLibrary clip for '\(gloss)'")
                    continue
                }
                loaded.append(Template(
                    gloss: gloss,
                    frameCount: clip.frameCount,
                    frames: clip.frames
                ))
                print("[PoseTemplateMatcher] loaded template '\(gloss)' frames=\(clip.frameCount)")
            } catch {
                print("[PoseTemplateMatcher] failed to load '\(gloss)': \(error)")
            }
        }
        guard !loaded.isEmpty else {
            print("[PoseTemplateMatcher] no templates loaded; matcher disabled")
            return nil
        }
        self.templates = loaded
    }

    /// Compare `window` against every loaded template, return up to `k`
    /// best matches ranked by softmax probability. Returns an empty array
    /// when the best distance exceeds `maxAcceptableDistance` — i.e. the
    /// pose really doesn't look like any of the demo signs.
    func match(window: [NormalizedFrame], k: Int) -> [Result] {
        guard !window.isEmpty, !templates.isEmpty, k > 0 else { return [] }
        let W = window.count

        var distances = [Float](repeating: .infinity, count: templates.count)
        for (i, template) in templates.enumerated() {
            let T = template.frameCount
            if T > W {
                // Template longer than window — end-align: compare the
                // window against the template's last W frames.
                distances[i] = frameDistance(
                    template: template.frames,
                    templateLen: T,
                    templateOffset: T - W,
                    window: window,
                    startInWindow: 0,
                    length: W
                )
                continue
            }
            // Slide the template over the window with stride; keep the
            // alignment with the smallest mean joint distance.
            var best: Float = .infinity
            var s = 0
            while s + T <= W {
                let d = frameDistance(
                    template: template.frames,
                    templateLen: T,
                    templateOffset: 0,
                    window: window,
                    startInWindow: s,
                    length: T
                )
                if d < best { best = d }
                s += slideStride
            }
            distances[i] = best
        }

        // Gate: if even the closest template is far away, the user isn't
        // making any demo sign. Return nothing so the fusion layer falls
        // through to "no emission" rather than guessing.
        let bestOverall = distances.min() ?? .infinity
        if bestOverall > maxAcceptableDistance { return [] }

        // softmax(−d / τ) over the templates. `expf` (C standard library
        // single-precision variant) is used explicitly so the compiler can't
        // pick up a non-Float `exp` overload (e.g. the `Duration` one).
        let logits: [Float] = distances.map { -$0 / temperature }
        var maxLogit: Float = -.infinity
        for l in logits where l > maxLogit { maxLogit = l }
        let expArr: [Float] = logits.map { expf($0 - maxLogit) }
        var sumExp: Float = 0
        for e in expArr { sumExp += e }
        let probs: [Float] = expArr.map { $0 / sumExp }

        var results: [Result] = []
        for i in 0..<templates.count {
            results.append(Result(
                gloss: templates[i].gloss,
                confidence: probs[i],
                distance: distances[i]
            ))
        }
        results.sort { $0.confidence > $1.confidence }
        return Array(results.prefix(k))
    }

    /// Mean per-joint L2 distance between template[templateOffset..<+length]
    /// and window[startInWindow..<+length]. Skips joints with mask=0 on the
    /// live side so MediaPipe dropouts don't dominate the score.
    private func frameDistance(
        template: [Float],
        templateLen: Int,
        templateOffset: Int,
        window: [NormalizedFrame],
        startInWindow: Int,
        length: Int
    ) -> Float {
        let J = Landmark45.count
        var totalSq: Float = 0
        var counted: Int = 0
        for t in 0..<length {
            let wFrame = window[startInWindow + t]
            let tFrameBase = (templateOffset + t) * J * 3
            for j in 0..<J {
                // Live mask of 0 = MediaPipe dropped this joint → skip so
                // a missing right hand on iOS doesn't blow up the distance.
                if wFrame.mask[j] <= 0 { continue }
                // NormalizedFrame.values is stride 3 (x, y, z per joint), not 4.
                // The earlier `j * 4` reach went out of bounds for j ≥ 34 the
                // moment MediaPipe actually started detecting joints — the
                // mask guard had been silently saving it while the recognizer
                // was getting near-empty frames.
                let wBase = j * 3
                let tBase = tFrameBase + j * 3
                let dx = wFrame.values[wBase]     - template[tBase]
                let dy = wFrame.values[wBase + 1] - template[tBase + 1]
                let dz = wFrame.values[wBase + 2] - template[tBase + 2]
                totalSq += dx * dx + dy * dy + dz * dz
                counted += 1
            }
        }
        guard counted > 0 else { return .infinity }
        return (totalSq / Float(counted)).squareRoot()
    }
}
