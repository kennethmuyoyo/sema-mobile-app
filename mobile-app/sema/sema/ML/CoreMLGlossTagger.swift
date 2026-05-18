import CoreML
import Foundation

/// CoreML-backed v11 phonological gloss recognizer.
///
/// Loads `ksl_model.mlpackage` from the app bundle and runs single-window
/// inference on the `(1, 64, 135)` MediaPipe-derived feature tensor produced
/// by `HolisticLandmarker`. Same public API as the previous `LiteRTGlossTagger`
/// (`predict(features:)` returns gloss logits) so `PathACoordinator` doesn't
/// care which runtime is underneath.
///
/// Why CoreML and not TFLite: MediaPipeTasksVision (used by the upstream
/// pose+hand landmarker) bundles its own TFLite runtime and force-loads its
/// symbols at link time. Adding a second TFLite framework via the
/// `TensorFlowLiteSwift` pod produced 48 duplicate-symbol linker errors that
/// modern `ld` won't suppress. CoreML uses Apple's Espresso runtime which
/// shares zero libraries with MediaPipe's TFLite, so the two coexist
/// cleanly and we get Neural Engine acceleration for free.
///
/// Aux phonological heads (the 30-dim argmax output) were dropped from this
/// export — coremltools couldn't trace the int-cast in their argmax+stack
/// path. If/when phonological Hamming retrieval comes back online, re-export
/// the model with aux heads cast to int32 inside the wrapper (see comments
/// in `recognition/export/to_coreml_v11.py`).
actor CoreMLGlossTagger {

    struct Configuration: Sendable {
        let inputSeqLen: Int          // 64 — fixed window the model traces with
        let featureDim: Int           // 135 — 45 joints × xyz
        let vocabSize: Int            // 1900 (v11 small) or 4146 (v11 full)
        /// Sidecar metadata loaded from `ksl_model.metadata.json`.
        let metadata: KSLModelMetadata
    }

    let configuration: Configuration
    private let model: MLModel

    init() throws {
        // Locate the .mlpackage in the app bundle. Xcode compiles
        // .mlpackage -> .mlmodelc at build time and copies the .mlmodelc
        // into the bundle root, so we look for the .mlmodelc form first.
        let resourceName = "ksl_model"
        var modelURL: URL?
        if let url = Bundle.main.url(forResource: resourceName, withExtension: "mlmodelc") {
            modelURL = url
        } else if let url = Bundle.main.url(forResource: resourceName, withExtension: "mlpackage") {
            // Dev builds occasionally ship the uncompiled .mlpackage; compile
            // it on the fly so this still works.
            modelURL = try MLModel.compileModel(at: url)
        }
        guard let url = modelURL else {
            throw CoreMLGlossTaggerError.modelNotInBundle
        }

        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all     // Neural Engine + GPU + CPU
        let model = try MLModel(contentsOf: url, configuration: cfg)
        self.model = model

        let metadata = try KSLModelMetadata.loadFromBundle()

        // Verify shape contract: the .mlpackage exposes an input named
        // "features" with shape (1, 64, 135). If the export gets re-done
        // with different dims this catches drift on first launch.
        guard let inputDesc = model.modelDescription.inputDescriptionsByName["features"],
              let cnst      = inputDesc.multiArrayConstraint else {
            throw CoreMLGlossTaggerError.unexpectedShape("model has no 'features' multi-array input")
        }
        let dims = cnst.shape.map(\.intValue)
        guard dims.count == 3, dims[0] == 1 else {
            throw CoreMLGlossTaggerError.unexpectedShape("input shape \(dims) expected (1, T, F)")
        }
        let inputSeqLen = dims[1]
        let featureDim  = dims[2]
        guard featureDim == metadata.featureDim else {
            throw CoreMLGlossTaggerError.metadataMismatch(
                "feature dim \(featureDim) ≠ metadata.featureDim \(metadata.featureDim)")
        }

        // The output 'gloss_logits' shape includes vocab_size — cross-check.
        var modelVocab = -1
        if let outDesc = model.modelDescription.outputDescriptionsByName["gloss_logits"],
           let cnst    = outDesc.multiArrayConstraint {
            modelVocab = cnst.shape.last?.intValue ?? -1
        }
        guard modelVocab == metadata.vocabSize else {
            throw CoreMLGlossTaggerError.metadataMismatch(
                "model output vocab \(modelVocab) ≠ metadata.vocabSize \(metadata.vocabSize)")
        }

        self.configuration = Configuration(
            inputSeqLen: inputSeqLen,
            featureDim: featureDim,
            vocabSize: modelVocab,
            metadata: metadata,
        )

        print("[CoreMLGlossTagger] loaded \(url.lastPathComponent)")
        print("[CoreMLGlossTagger]   in:  (1, \(inputSeqLen), \(featureDim))")
        print("[CoreMLGlossTagger]   out: gloss=(1, \(modelVocab))")
        print("[CoreMLGlossTagger]   vocab: \(metadata.glossIdToName.count) glosses")
    }

    /// Run inference on a single 64-frame window of MediaPipe-normalised
    /// landmarks. `features.count` must equal `inputSeqLen * featureDim`.
    /// Returns gloss logits as a flat `[Float]` of length `vocabSize`.
    ///
    /// Returns a tuple to match `LiteRTGlossTagger.predict(features:)`'s
    /// public shape — the second element is empty `[]` for now since the
    /// CoreML export drops aux heads (see file header).
    func predict(features: [Float]) throws -> (glossLogits: [Float], auxIndices: [Int32]) {
        let cfg = configuration
        guard features.count == cfg.inputSeqLen * cfg.featureDim else {
            throw CoreMLGlossTaggerError.unexpectedShape(
                "feature count \(features.count) ≠ \(cfg.inputSeqLen * cfg.featureDim)")
        }
        let mlArray = try MLMultiArray(
            shape: [1, NSNumber(value: cfg.inputSeqLen), NSNumber(value: cfg.featureDim)],
            dataType: .float32
        )
        let ptr = mlArray.dataPointer.bindMemory(to: Float.self, capacity: features.count)
        features.withUnsafeBufferPointer { ptr.update(from: $0.baseAddress!, count: features.count) }

        let input = try MLDictionaryFeatureProvider(dictionary: ["features": MLFeatureValue(multiArray: mlArray)])
        let output = try model.prediction(from: input)

        guard let logitsArray = output.featureValue(for: "gloss_logits")?.multiArrayValue else {
            throw CoreMLGlossTaggerError.unexpectedShape("missing 'gloss_logits' in model output")
        }
        let n = logitsArray.count
        var logits = [Float](repeating: 0, count: n)
        let outPtr = logitsArray.dataPointer.bindMemory(to: Float.self, capacity: n)
        for i in 0..<n { logits[i] = outPtr[i] }
        return (glossLogits: logits, auxIndices: [])
    }

    /// One-time JIT compile + warmup. Loads the kernels onto Neural
    /// Engine/GPU so the first user-visible inference doesn't pay the
    /// cold-start cost.
    func prewarm() throws {
        let cfg = configuration
        let zero = [Float](repeating: 0, count: cfg.inputSeqLen * cfg.featureDim)
        _ = try predict(features: zero)
    }
}

enum CoreMLGlossTaggerError: Error, CustomStringConvertible {
    case modelNotInBundle
    case unexpectedShape(String)
    case metadataMismatch(String)

    var description: String {
        switch self {
        case .modelNotInBundle:
            return "ksl_model.mlpackage / .mlmodelc not present in app bundle. " +
                "Check that Resources/ksl_model.mlpackage is included in the target."
        case .unexpectedShape(let msg):
            return "CoreMLGlossTagger shape mismatch: \(msg)"
        case .metadataMismatch(let msg):
            return "CoreMLGlossTagger metadata mismatch: \(msg)"
        }
    }
}
