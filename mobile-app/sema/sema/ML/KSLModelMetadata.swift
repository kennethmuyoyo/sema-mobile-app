import Foundation

/// Sidecar metadata for the v11 KSL recognizer model.
///
/// Loaded once at startup from `ksl_model.metadata.json` (bundled alongside
/// `ksl_model.mlpackage`). The same JSON schema is produced by both
/// `recognition/convert_to_litert.py` and `recognition/export/to_coreml_v11.py`
/// so this type is runtime-agnostic — it doesn't care whether the model
/// runs through CoreML, LiteRT, or any future backend.
///
/// What lives here:
///   - vocabularies for the main gloss head (id ↔ name maps)
///   - phonological aux head label spaces (location, palm orientation,
///     handshape cluster, etc.)
///   - per-gloss phonological centroids for Hamming retrieval
///   - the canonical 45-joint name order the recogniser was trained on
///   - the curated "demo glosses" list — glosses with ≥10 training examples
struct KSLModelMetadata: Decodable, Sendable {
    let version: String?
    let architecture: String?
    let window: Int
    let featureDim: Int          // `n_feat`
    let nJoints: Int             // `n_joints`
    let vocabSize: Int

    /// Gloss-id → human-readable gloss name. Keys are stringified ints because
    /// JSON object keys must be strings; `glossNameToId` is the inverse.
    let glossIdToName: [String: String]
    let glossNameToId: [String: Int]

    /// Order in which the model emits aux predictions. `auxIndices[i]` is the
    /// argmax over the head named `auxKeys[i]`; look up that integer in the
    /// vocabulary table for that head (loc / palm / etc.) to get a string.
    /// Currently empty in the CoreML export — see CoreMLGlossTagger.swift.
    let auxKeys: [String]
    let auxSizes: [String: Int]

    /// Per-head label vocabularies. The model emits an integer argmax in
    /// `auxIndices`; index into these to render a readable string.
    let locationLabels: [String]
    let moveTypes: [String]
    let moveDirs: [String]
    let orientLabels: [String]
    let contactHandsLabels: [String]
    let contactBodyLabels: [String]

    /// Per-gloss phonological centroid table for Hamming-distance retrieval.
    /// `phonologicalCentroids[glossName][auxKey] = expectedIndex`. The mobile
    /// app can rank glosses by mismatching count against the model's predicted
    /// aux_indices even when the gloss head's argmax is wrong.
    let phonologicalCentroids: [String: [String: Int]]

    /// Optional list of glosses with enough training data to demo reliably
    /// (≥10 examples in the training set). For the demo UI.
    let demoGlosses: [String]

    /// 45 joint names in the canonical order the model was trained on.
    /// The iOS feature builder must emit joints in this order; mismatch =
    /// garbage predictions even if the model loads.
    let jointOrder: [String]

    /// K-means centroids for handshape clustering. Optional — only for
    /// explainability ("predicted handshape = cluster 5 because the curl
    /// vector is closest to centroid #5"). Not needed for normal inference.
    let handshapeKmeansCentroids: [[Float]]?

    enum CodingKeys: String, CodingKey {
        case version, architecture, window
        case featureDim = "n_feat"
        case nJoints    = "n_joints"
        case vocabSize  = "vocab_size"
        case glossIdToName        = "gloss_id_to_name"
        case glossNameToId        = "gloss_name_to_id"
        case auxKeys              = "aux_keys"
        case auxSizes             = "aux_sizes"
        case locationLabels       = "location_labels"
        case moveTypes            = "move_types"
        case moveDirs             = "move_dirs"
        case orientLabels         = "orient_labels"
        case contactHandsLabels   = "contact_hands_labels"
        case contactBodyLabels    = "contact_body_labels"
        case phonologicalCentroids = "phonological_centroids"
        case demoGlosses          = "demo_glosses"
        case jointOrder           = "joint_order"
        case handshapeKmeansCentroids = "handshape_kmeans_centroids"
    }

    /// Load the sidecar from the app bundle. Expects
    /// `ksl_model.metadata.json` to be present.
    static func loadFromBundle() throws -> KSLModelMetadata {
        guard let url = Bundle.main.url(forResource: "ksl_model.metadata",
                                          withExtension: "json") else {
            throw KSLModelMetadataError.notInBundle
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(KSLModelMetadata.self, from: data)
    }

    /// Render an `aux_indices` row as a `(headName, labelString)` table.
    /// `indices.count` must equal `auxKeys.count`. Empty in the current
    /// CoreML export.
    func decode(auxIndices indices: [Int32]) -> [(head: String, value: String)] {
        precondition(indices.count == auxKeys.count,
                     "aux_indices count \(indices.count) ≠ aux_keys count \(auxKeys.count)")
        return zip(auxKeys, indices).map { (head, idx) in
            (head, label(for: head, index: Int(idx)))
        }
    }

    /// Map a single (head, index) pair to its readable label.
    func label(for head: String, index: Int) -> String {
        let vocab = labelVocab(for: head)
        guard index >= 0 && index < vocab.count else { return "?\(index)" }
        return vocab[index]
    }

    /// Per-head label vocabulary. Picks the table by name suffix — every
    /// head ends in `_loc`, `_palm`, `_hs`, `_mt`, `_md`, `_contact_hand`,
    /// or `_contact_body`. Heads that don't match fall back to a numeric label.
    private func labelVocab(for head: String) -> [String] {
        if head.hasSuffix("_loc")          { return locationLabels }
        if head.hasSuffix("_palm")         { return orientLabels }
        if head.hasSuffix("_mt")           { return moveTypes }
        if head.hasSuffix("_md")           { return moveDirs }
        if head.hasSuffix("_contact_hand") { return contactHandsLabels }
        if head.hasSuffix("_contact_body") { return contactBodyLabels }
        return []
    }

    /// Phonological Hamming retrieval — for each gloss with a centroid,
    /// count how many of the predicted aux head argmaxes match. Returns
    /// glosses sorted by ascending distance.
    func retrieveByHamming(predictedAux indices: [Int32]) -> [(gloss: String, distance: Int)] {
        guard indices.count == auxKeys.count else { return [] }
        var scored: [(String, Int)] = []
        scored.reserveCapacity(phonologicalCentroids.count)
        for (gloss, centroid) in phonologicalCentroids {
            var d = 0
            for (i, key) in auxKeys.enumerated() {
                if let expected = centroid[key], expected != Int(indices[i]) {
                    d += 1
                }
            }
            scored.append((gloss, d))
        }
        scored.sort { $0.1 < $1.1 }
        return scored
    }
}

enum KSLModelMetadataError: Error, CustomStringConvertible {
    case notInBundle
    var description: String {
        switch self {
        case .notInBundle:
            return "ksl_model.metadata.json not present in app bundle. " +
                "Check Build Phases → Copy Bundle Resources."
        }
    }
}
