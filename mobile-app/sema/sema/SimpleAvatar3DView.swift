import SwiftUI
import SceneKit
import simd

struct SimpleAvatar3DView: UIViewRepresentable {
    let frame: PoseFrame?
    
    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.backgroundColor = .clear
        view.autoenablesDefaultLighting = true
        view.allowsCameraControl = false
        // Cap the rig render loop at 30 fps. SCNView defaults to the display's
        // native refresh (60 Hz on standard, 120 Hz on ProMotion), which on
        // this rig burns enough GPU + main-thread time to starve the focus
        // engine and SwiftUI gesture recognition. The avatar source frames
        // arrive at 24 Hz anyway, so anything above ~30 is wasted.
        view.preferredFramesPerSecond = 30
        // The avatar is a passive viewer; pulling it out of the focus chain
        // silences the recurring "focusItemsInRect: - caching for linear focus
        // movement is limited as long as this view is on screen" warning and
        // stops UIFocusSystem from continuously polling it.
        view.isUserInteractionEnabled = false

        // Previews time out if we synchronously load the full USD rig on first frame.
        if TestEnvironment.isPreview {
            return view
        }

        guard let url = Bundle.main.url(forResource: "hackathon", withExtension: "usdc") else {
            print("Could not find hackathon.usdc")
            return view
        }
        
        guard let scene = try? SCNScene(url: url, options: nil) else {
            print("Could not load scene")
            return view
        }

        Self.remapTexturePathsToBundle(scene)

        // Fix Z-up to Y-up
        let root = SCNNode()
        for child in scene.rootNode.childNodes {
            root.addChildNode(child)
        }
        root.simdEulerAngles = SIMD3<Float>(-.pi / 2, 0, 0)

        let wrapperScene = SCNScene()
        wrapperScene.rootNode.addChildNode(root)

        // Frame the upper body (torso + head) rather than the entire rig.
        // Using the full bbox includes legs and (at T-pose init) extended
        // arms — both push the camera back so far the hands end up too small
        // to read once the avatar is actually signing.
        let upperBox = Self.upperBodyBounds(in: wrapperScene.rootNode)
            ?? wrapperScene.rootNode.boundingBox
        let center = SCNVector3(
            (upperBox.min.x + upperBox.max.x) * 0.5,
            (upperBox.min.y + upperBox.max.y) * 0.5,
            (upperBox.min.z + upperBox.max.z) * 0.5
        )
        let width = max(upperBox.max.x - upperBox.min.x, 0.1)
        let fovDegrees: Float = 38
        let halfFov = fovDegrees * .pi / 360
        // Width-only framing keeps the avatar close while preserving hand room.
        let distance = (width * 0.72) / tan(halfFov)

        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.fieldOfView = CGFloat(fovDegrees)
        cameraNode.camera?.zNear = 0.05
        cameraNode.camera?.zFar = max(50, Double(distance) * 4)
        cameraNode.position = SCNVector3(center.x, center.y, center.z + distance)

        let lookAtTarget = SCNNode()
        lookAtTarget.position = center
        wrapperScene.rootNode.addChildNode(lookAtTarget)

        let lookAt = SCNLookAtConstraint(target: lookAtTarget)
        cameraNode.constraints = [lookAt]
        wrapperScene.rootNode.addChildNode(cameraNode)

        view.pointOfView = cameraNode

        // Add lighting
        let ambientLightNode = SCNNode()
        ambientLightNode.light = SCNLight()
        ambientLightNode.light!.type = .ambient
        ambientLightNode.light!.color = UIColor(white: 0.6, alpha: 1.0)
        wrapperScene.rootNode.addChildNode(ambientLightNode)
        
        let directionalLightNode = SCNNode()
        directionalLightNode.light = SCNLight()
        directionalLightNode.light!.type = .directional
        directionalLightNode.light!.color = UIColor(white: 0.8, alpha: 1.0)
        directionalLightNode.eulerAngles = SCNVector3(x: -.pi/4, y: -.pi/4, z: 0)
        wrapperScene.rootNode.addChildNode(directionalLightNode)
        view.scene = wrapperScene
        
        context.coordinator.setup(in: wrapperScene.rootNode)
        return view
    }

    /// World-space bounds of the upper torso and head, expanded horizontally
    /// for signing hands without pulling the camera back to show the full body.
    private static func upperBodyBounds(in root: SCNNode) -> (min: SCNVector3, max: SCNVector3)? {
        let torsoNames = ["spine1", "spine2", "spine3", "neck", "head", "left_shoulder", "right_shoulder"]
        var positions: [SIMD3<Float>] = []
        for name in torsoNames {
            guard let node = root.childNode(withName: name, recursively: true) else { continue }
            positions.append(node.simdWorldPosition)
        }
        guard !positions.isEmpty else { return nil }

        var minP = positions[0]
        var maxP = positions[0]
        for p in positions {
            minP = simd_min(minP, p)
            maxP = simd_max(maxP, p)
        }

        let spine1Y = root.childNode(withName: "spine1", recursively: true)?.simdWorldPosition.y
        if let spine1Y {
            minP.y = spine1Y - 0.08
        }

        // Use wrist X positions for hand room, but do not include wrist Y or
        // the camera zooms out to show the full hanging arms and legs.
        let lw = root.childNode(withName: "left_wrist", recursively: true)?.simdWorldPosition
        let rw = root.childNode(withName: "right_wrist", recursively: true)?.simdWorldPosition
        let wristMargin: Float = 0.08
        if let lw {
            minP.x = min(minP.x, lw.x - wristMargin)
            maxP.x = max(maxP.x, lw.x + wristMargin)
        }
        if let rw {
            minP.x = min(minP.x, rw.x - wristMargin)
            maxP.x = max(maxP.x, rw.x + wristMargin)
        }
        maxP.y += 0.16

        return (
            min: SCNVector3(minP.x, minP.y, minP.z),
            max: SCNVector3(maxP.x, maxP.y, maxP.z)
        )
    }

    private static func remapTexturePathsToBundle(_ scene: SCNScene) {
        // The USDC was authored with `@./textures/Foo.png@` references.
        // SceneKit's USD loader resolves those against the USDC's directory,
        // but Xcode 16's synchronized root group flattens every source file
        // into the bundle root, so the resolved URL points at a non-existent
        // `<bundle>/textures/Foo.png`. By the time we get here the original
        // filename is gone (property.contents reads back as a stub URL whose
        // last path component is just "/"), so we bind textures by walking
        // the bundled images and matching the material name's prefix.
        let bundled = indexBundledImages()
        let allNodes = scene.rootNode.childNodes(passingTest: { _, _ in true })

        // Sort so L/R variants (and .001/.002 versions) resolve deterministically:
        // the alphabetically-last file wins each channel.
        let sortedBundled = bundled.sorted { $0.key < $1.key }

        var assigned = 0
        var materialsSeen = 0
        var materialsMatched = 0
        var unboundMaterials: [String] = []
        for node in allNodes {
            guard let geometry = node.geometry else { continue }
            for material in geometry.materials {
                materialsSeen += 1
                let name = material.name ?? ""
                let base = stripVariantSuffix(name)
                guard !base.isEmpty else { continue }

                var bound = bindByPrefix(material: material, base: base, bundled: sortedBundled)
                if bound == 0 {
                    bound = bindByToken(material: material, base: base, bundled: sortedBundled)
                }
                if bound > 0 {
                    materialsMatched += 1
                    assigned += bound
                } else {
                    unboundMaterials.append(base)
                }

                configureUnboundMaterial(material: material, base: base)
            }
        }
        print("[Avatar3D] textures bound=\(assigned) materialsMatched=\(materialsMatched)/\(materialsSeen)")
        if !unboundMaterials.isEmpty {
            print("[Avatar3D] unbound materials: \(unboundMaterials.sorted().joined(separator: ", "))")
        }

        if let env = bundled["color_121212.hdr"] {
            scene.lightingEnvironment.contents = env
            scene.lightingEnvironment.intensity = 0.4
        }
    }

    /// Strict prefix match — the historical path. Works when the material name
    /// is a literal prefix of the texture file name (e.g. material `Basic_T_shirts`
    /// → `Basic_T_shirts_Diffuse.001.png`).
    private static func bindByPrefix(
        material: SCNMaterial,
        base: String,
        bundled: [(key: String, value: URL)]
    ) -> Int {
        var assigned = 0
        for (fileName, url) in bundled where fileName.hasPrefix("\(base)_") {
            if assignTextureChannel(material: material, fileName: fileName, url: url) {
                assigned += 1
            }
        }
        return assigned
    }

    /// Token-equality fallback for materials whose texture files use a different
    /// vendor prefix than the material name (e.g. material `Eye` → `Std_Eye_L_Diffuse.001.jpg`,
    /// material `Eyebrows` → `Toon_Eyebrows_Transparency_Diffuse.001.png`). We split
    /// both names on `_`/`-`/`.` and require a whole-token match so `Eye` doesn't
    /// silently grab `Eyelash`/`Eyebrows` textures.
    private static func bindByToken(
        material: SCNMaterial,
        base: String,
        bundled: [(key: String, value: URL)]
    ) -> Int {
        let materialTokens = tokenize(base)
            .filter { !$0.isEmpty && !$0.allSatisfy(\.isNumber) }
        guard !materialTokens.isEmpty else { return 0 }

        // Try the longest, most-distinctive token first so e.g. `Tongue` beats `CC`/`Base`.
        let ordered = materialTokens.sorted { $0.count > $1.count }
        for token in ordered {
            var assigned = 0
            for (fileName, url) in bundled where tokenize(fileName).contains(token) {
                if assignTextureChannel(material: material, fileName: fileName, url: url) {
                    assigned += 1
                }
            }
            if assigned > 0 { return assigned }
        }
        return 0
    }

    private static func tokenize(_ name: String) -> Set<String> {
        let stem = (name as NSString).deletingPathExtension
        return Set(stem.split(whereSeparator: { "_-.".contains($0) }).map(String.init))
    }

    private static func assignTextureChannel(material: SCNMaterial, fileName: String, url: URL) -> Bool {
        if fileName.contains("_Diffuse") {
            material.diffuse.contents = url
            // Reallusion/CC textures bake opacity into the diffuse PNG with names like
            // `Std_Eyelash_Diffuse-Std_Eyelash_Opacity.001.png`. Route the alpha channel
            // into `transparent` so eyelashes/eyebrows/hair don't render as opaque cards.
            if fileName.contains("_Opacity") {
                material.transparent.contents = url
                material.transparencyMode = .aOne
                material.isDoubleSided = true
            }
            return true
        } else if fileName.contains("_Normal") {
            material.normal.contents = url
            return true
        } else if fileName.contains("_Specular") {
            material.specular.contents = url
            return true
        }
        return false
    }

    /// Post-process a material that didn't find a diffuse texture.
    ///
    /// CC/Reallusion rigs ship multiple geometry layers per eye — the textured
    /// eyeball plus a clear `Cornea` shell and a thin `Wet` (tearline) layer
    /// the artist authored as glass. SceneKit's USD import drops the shader
    /// graph and leaves them with an opaque default material, so they render
    /// as a chalky bubble that hides the iris. Hide these so the textured
    /// eyeball below shows through cleanly.
    ///
    /// Sport_Pants and Material_003 export from Blender without a diffuse
    /// connection on the SceneKit side, and the bundle has no matching
    /// `*_Diffuse.*` file either (only a `*_Normal.*` map exists for the
    /// pants — that gives bumpiness but no colour). Without a fallback they
    /// render as a flat untextured white blob. Apply a sensible fabric
    /// colour instead.
    ///
    /// We previously gated this on a `bound: Bool` flag, but that was true
    /// for any channel — so Sport_Pants having only a normal map made the
    /// guard bail out, leaving the trousers white. Now we inspect the
    /// diffuse channel directly: if no real image is attached, run the
    /// fallback regardless of what other channels did or didn't pick up.
    private static func configureUnboundMaterial(material: SCNMaterial, base: String) {
        if hasDiffuseTexture(material) { return }
        let tokens = tokenize(base)
        let isOverlay = tokens.contains("Cornea")
            || tokens.contains("Wet")
            || tokens.contains("Tearline")
            || base.lowercased().contains("occlusion")
            || base == "eye_smplhf"
        if isOverlay {
            material.transparency = 0.0
            material.writesToDepthBuffer = false
            return
        }

        if let color = fallbackClothingColor(forBase: base) {
            material.diffuse.contents = color
            material.roughness.contents = 0.85
            material.metalness.contents = 0.0
            material.lightingModel = .physicallyBased
        }
    }

    /// True iff the material's diffuse channel currently holds an actual
    /// image (URL, UIImage, or filesystem path string) — i.e. a real texture
    /// is bound. SceneKit's default diffuse content is a flat `UIColor`,
    /// which we want to overwrite with a sensible fallback.
    ///
    /// `CGImage` is intentionally NOT checked: it's a Core Foundation type,
    /// and Swift's `is CGImage` against `Any?` is a CF bridging cast that
    /// always succeeds — it'd silently match a `UIColor` default and short-
    /// circuit the fallback. The image types our binder actually assigns
    /// (`material.diffuse.contents = url`) are URLs, so checking those is
    /// sufficient.
    private static func hasDiffuseTexture(_ material: SCNMaterial) -> Bool {
        let contents = material.diffuse.contents
        return contents is URL
            || contents is UIImage
            || contents is String
    }

    /// Sensible default colors for clothing materials whose diffuse export
    /// is missing. Names mirror the USDC material names so we can extend the
    /// table as the rig gains new clothing pieces.
    private static func fallbackClothingColor(forBase base: String) -> UIColor? {
        let lower = base.lowercased()
        if lower.contains("pants") || lower.contains("trouser") {
            // Charcoal navy — reads as athletic/joggers fabric.
            return UIColor(red: 0.16, green: 0.19, blue: 0.24, alpha: 1.0)
        }
        if lower == "material_003" {
            // Likely belt / waistband / accent — slightly darker so it reads
            // as trim rather than fabric.
            return UIColor(red: 0.10, green: 0.10, blue: 0.12, alpha: 1.0)
        }
        return nil
    }

    /// Build a `filename → URL` index over every image at the bundle root.
    /// PoseLibrary/textures is also probed in case a future change preserves
    /// the directory hierarchy.
    private static func indexBundledImages() -> [String: URL] {
        var index: [String: URL] = [:]
        for ext in ["png", "jpg", "jpeg", "hdr"] {
            let roots: [String?] = [nil, "PoseLibrary/textures"]
            for sub in roots {
                let urls = Bundle.main.urls(forResourcesWithExtension: ext, subdirectory: sub) ?? []
                for url in urls {
                    index[url.lastPathComponent] = url
                }
            }
        }
        return index
    }

    /// USDC merges from Blender often append `_001`, `_002` etc. to material
    /// names. Strip that so the material's base name matches its texture file
    /// prefix (e.g. `Basic_T_shirts_002` → `Basic_T_shirts`).
    private static func stripVariantSuffix(_ name: String) -> String {
        guard let range = name.range(of: #"_\d{3}$"#, options: .regularExpression) else { return name }
        return String(name[..<range.lowerBound])
    }
    
    func updateUIView(_ uiView: SCNView, context: Context) {
        if let frame = frame {
            context.coordinator.apply(frame: frame)
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator {
        struct RigBoneBinding {
            let node: SCNNode
            let restLocalOrientation: simd_quatf
            let restWorldOrientation: simd_quatf
        }

        struct FingerBoneBinding {
            let node: SCNNode
            let restOrientation: simd_quatf
            let restDirection: SIMD3<Float>
        }

        var upperLeftNode: SCNNode?
        var lowerLeftNode: SCNNode?
        var handLeftNode: SCNNode?
        var upperRightNode: SCNNode?
        var lowerRightNode: SCNNode?
        var handRightNode: SCNNode?

        var leftShoulderNode: SCNNode?
        var rightShoulderNode: SCNNode?

        var restUpperLeftOrientation = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
        var restLowerLeftOrientation = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
        var restUpperRightOrientation = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
        var restLowerRightOrientation = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))

        var restUpperLeftDirection = SIMD3<Float>(1, 0, 0)
        var restLowerLeftDirection = SIMD3<Float>(1, 0, 0)
        var restUpperRightDirection = SIMD3<Float>(-1, 0, 0)
        var restLowerRightDirection = SIMD3<Float>(-1, 0, 0)

        var orderedRigBindings: [RigBoneBinding?] = []
        var rigParentIndex: [Int] = []
        var perBoneLocalOffsets: [String: simd_quatf] = [:]
        var fingerBindings: [String: FingerBoneBinding] = [:]
        var hasLoggedBVHRotationMode = false

        var modelShoulderDist: Float = 0.33
        
        func setup(in root: SCNNode) {
            // Always register the rig bindings so the quaternion sidecar path
            // can drive every joint. The body-IK fallback bones (upperarm_l
            // etc.) don't exist on this rig — leave the optionals nil so the
            // fallback path becomes a no-op rather than killing setup.
            registerRigBones(in: root)
            registerFingerBones(in: root)

            let resolved = orderedRigBindings.compactMap { $0 }.count
            print("[Avatar3D] setup resolved \(resolved)/\(BVHRigRotationLayout.jointCount) rig bones, " +
                  "\(fingerBindings.count) finger bones")
        }
        
        func apply(frame: PoseFrame) {
            if applyRigRotationsIfAvailable(frame) {
                return
            }

            guard
                let upperL = upperLeftNode,
                let lowerL = lowerLeftNode,
                let upperR = upperRightNode,
                let lowerR = lowerRightNode,
                let sl = leftShoulderNode,
                let sr = rightShoulderNode
            else {
                return
            }

            // MediaPipe coordinates: X right, Y down, Z depth.
            // SceneKit mapping here is X right, Y up, Z depth.
            let mpShoulderL = frame.point(at: Landmark45.index(of: "left_shoulder"))
            let mpShoulderR = frame.point(at: Landmark45.index(of: "right_shoulder"))
            let mpElbowL = frame.point(at: Landmark45.index(of: "left_elbow"))
            let mpElbowR = frame.point(at: Landmark45.index(of: "right_elbow"))
            let mpWristL = frame.point(at: Landmark45.index(of: "left_wrist"))
            let mpWristR = frame.point(at: Landmark45.index(of: "right_wrist"))

            let mpShoulderDist = simd_distance(mpShoulderL, mpShoulderR)
            let scale = modelShoulderDist / max(mpShoulderDist, 0.001)

            func mapVector(_ v: SIMD3<Float>) -> SIMD3<Float> {
                // Keep depth sign aligned with the model so elbow bend stays forward.
                SIMD3<Float>(v.x, -v.y, v.z) * scale
            }

            let leftShoulderWorld = sl.simdWorldPosition
            let rightShoulderWorld = sr.simdWorldPosition
            let leftElbowTarget = leftShoulderWorld + mapVector(mpElbowL - mpShoulderL)
            let rightElbowTarget = rightShoulderWorld + mapVector(mpElbowR - mpShoulderR)
            let leftWristTarget = leftShoulderWorld + mapVector(mpWristL - mpShoulderL)
            let rightWristTarget = rightShoulderWorld + mapVector(mpWristR - mpShoulderR)

            orientBone(
                upperL,
                restOrientation: restUpperLeftOrientation,
                restDirection: restUpperLeftDirection,
                targetDirection: leftElbowTarget - upperL.simdWorldPosition
            )
            orientBone(
                upperR,
                restOrientation: restUpperRightOrientation,
                restDirection: restUpperRightDirection,
                targetDirection: rightElbowTarget - upperR.simdWorldPosition
            )

            orientBone(
                lowerL,
                restOrientation: restLowerLeftOrientation,
                restDirection: restLowerLeftDirection,
                targetDirection: leftWristTarget - lowerL.simdWorldPosition
            )
            orientBone(
                lowerR,
                restOrientation: restLowerRightOrientation,
                restDirection: restLowerRightDirection,
                targetDirection: rightWristTarget - lowerR.simdWorldPosition
            )

            applyFingerTracking(
                frame: frame,
                mpLeftWrist: mpWristL,
                mpRightWrist: mpWristR,
                mapVector: mapVector
            )
        }

        private func registerRigBones(in root: SCNNode) {
            orderedRigBindings = BVHRigRotationLayout.jointOrder.map { name in
                guard let node = root.childNode(withName: name, recursively: true) else { return nil }
                return RigBoneBinding(
                    node: node,
                    restLocalOrientation: node.simdOrientation,
                    restWorldOrientation: node.simdWorldOrientation
                )
            }
            rigParentIndex = BVHRigRotationLayout.parentIndex
        }

        private func applyRigRotationsIfAvailable(_ frame: PoseFrame) -> Bool {
            // Sidecar shape: (jointCount, 4) flat = jointCount * 4 floats per frame,
            // each joint stored as [ix, iy, iz, r] (i.e. simd_quatf init order).
            // The Python retargeter has already converted from BVH source-rig
            // rotations into target-rig parent-local quaternions, so the iOS
            // side is a straight assignment.
            guard let q = frame.rigRotations else { return false }
            let count = BVHRigRotationLayout.jointCount
            guard q.count == count * 4 else { return false }
            guard orderedRigBindings.count == count else { return false }
            if !hasLoggedBVHRotationMode {
                hasLoggedBVHRotationMode = true
                let resolved = orderedRigBindings.compactMap { $0 }.count
                print("[Avatar3D] using rig quaternions (\(resolved)/\(count) joints resolved)")
            }
            for i in 0..<count {
                guard let binding = orderedRigBindings[i] else { continue }
                let base = i * 4
                binding.node.simdOrientation = simd_quatf(
                    ix: q[base], iy: q[base + 1], iz: q[base + 2], r: q[base + 3]
                )
            }
            return true
        }

        private func normalizedDirection(from: SCNNode, to: SCNNode) -> SIMD3<Float> {
            let delta = to.simdWorldPosition - from.simdWorldPosition
            let length = simd_length(delta)
            guard length > 0.0001 else { return SIMD3<Float>(0, 1, 0) }
            return delta / length
        }

        private func quaternionXYZDegrees(_ eulerDegrees: SIMD3<Float>) -> simd_quatf {
            let rad = eulerDegrees * (.pi / 180.0)
            let qx = simd_quatf(angle: rad.x, axis: SIMD3<Float>(1, 0, 0))
            let qy = simd_quatf(angle: rad.y, axis: SIMD3<Float>(0, 1, 0))
            let qz = simd_quatf(angle: rad.z, axis: SIMD3<Float>(0, 0, 1))
            return simd_normalize(qx * qy * qz)
        }

        private func orientBone(
            _ node: SCNNode,
            restOrientation: simd_quatf,
            restDirection: SIMD3<Float>,
            targetDirection: SIMD3<Float>
        ) {
            let targetLength = simd_length(targetDirection)
            guard targetLength > 0.0001 else { return }
            let targetDir = targetDirection / targetLength
            let rotation = rotationBetween(restDirection, targetDir)
            node.simdWorldOrientation = simd_normalize(rotation * restOrientation)
        }

        private func rotationBetween(_ from: SIMD3<Float>, _ to: SIMD3<Float>) -> simd_quatf {
            let a = simd_normalize(from)
            let b = simd_normalize(to)
            let dot = simd_dot(a, b)

            if dot > 0.9999 {
                return simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
            }
            if dot < -0.9999 {
                let fallbackAxis = abs(a.x) < 0.9
                    ? simd_normalize(simd_cross(a, SIMD3<Float>(1, 0, 0)))
                    : simd_normalize(simd_cross(a, SIMD3<Float>(0, 1, 0)))
                return simd_quatf(angle: .pi, axis: fallbackAxis)
            }
            return simd_quatf(from: a, to: b)
        }

        private func registerFingerBones(in root: SCNNode) {
            let fingers = ["thumb", "index", "middle", "ring", "pinky"]

            for finger in fingers {
                registerFingerBone(
                    name: "\(finger)_01_l",
                    nextName: "\(finger)_02_l",
                    in: root
                )
                registerFingerBone(
                    name: "\(finger)_02_l",
                    nextName: "\(finger)_03_l",
                    in: root
                )
                registerFingerBone(
                    name: "\(finger)_03_l",
                    nextName: nil,
                    in: root
                )

                registerFingerBone(
                    name: "\(finger)_01_r",
                    nextName: "\(finger)_02_r",
                    in: root
                )
                registerFingerBone(
                    name: "\(finger)_02_r",
                    nextName: "\(finger)_03_r",
                    in: root
                )
                registerFingerBone(
                    name: "\(finger)_03_r",
                    nextName: nil,
                    in: root
                )
            }
        }

        private func registerFingerBone(name: String, nextName: String?, in root: SCNNode) {
            guard let node = root.childNode(withName: name, recursively: true) else { return }

            let restDirection: SIMD3<Float>
            if let nextName, let nextNode = root.childNode(withName: nextName, recursively: true) {
                restDirection = normalizedDirection(from: node, to: nextNode)
            } else if let firstChild = node.childNodes.first {
                restDirection = normalizedDirection(from: node, to: firstChild)
            } else if let parent = node.parent {
                let parentToNode = node.simdWorldPosition - parent.simdWorldPosition
                let length = simd_length(parentToNode)
                restDirection = length > 0.0001 ? (parentToNode / length) : SIMD3<Float>(1, 0, 0)
            } else {
                restDirection = SIMD3<Float>(1, 0, 0)
            }

            fingerBindings[name] = FingerBoneBinding(
                node: node,
                restOrientation: node.simdWorldOrientation,
                restDirection: restDirection
            )
        }

        private func applyFingerTracking(
            frame: PoseFrame,
            mpLeftWrist: SIMD3<Float>,
            mpRightWrist: SIMD3<Float>,
            mapVector: (SIMD3<Float>) -> SIMD3<Float>
        ) {
            guard let handL = handLeftNode, let handR = handRightNode else { return }

            let leftWristWorld = handL.simdWorldPosition
            let rightWristWorld = handR.simdWorldPosition
            let fingers = ["thumb", "index", "middle", "ring", "pinky"]

            for finger in fingers {
                applyFingerChain(
                    finger: finger,
                    side: "left",
                    modelSuffix: "_l",
                    frame: frame,
                    wristWorld: leftWristWorld,
                    wristMP: mpLeftWrist,
                    mapVector: mapVector
                )
                applyFingerChain(
                    finger: finger,
                    side: "right",
                    modelSuffix: "_r",
                    frame: frame,
                    wristWorld: rightWristWorld,
                    wristMP: mpRightWrist,
                    mapVector: mapVector
                )
            }
        }

        private func applyFingerChain(
            finger: String,
            side: String,
            modelSuffix: String,
            frame: PoseFrame,
            wristWorld: SIMD3<Float>,
            wristMP: SIMD3<Float>,
            mapVector: (SIMD3<Float>) -> SIMD3<Float>
        ) {
            let j1Name = "\(side)_\(finger)1"
            let j2Name = "\(side)_\(finger)2"
            let j3Name = "\(side)_\(finger)3"

            let mp1 = frame.point(at: Landmark45.index(of: j1Name))
            let mp2 = frame.point(at: Landmark45.index(of: j2Name))
            let mp3 = frame.point(at: Landmark45.index(of: j3Name))

            let p1 = wristWorld + mapVector(mp1 - wristMP)
            let p2 = wristWorld + mapVector(mp2 - wristMP)
            let p3 = wristWorld + mapVector(mp3 - wristMP)

            orientFingerBone(
                "\(finger)_01\(modelSuffix)",
                targetDirection: p1 - wristWorld
            )
            orientFingerBone(
                "\(finger)_02\(modelSuffix)",
                targetDirection: p2 - p1
            )
            orientFingerBone(
                "\(finger)_03\(modelSuffix)",
                targetDirection: p3 - p2
            )
        }

        private func orientFingerBone(_ name: String, targetDirection: SIMD3<Float>) {
            guard let binding = fingerBindings[name] else { return }
            orientBone(
                binding.node,
                restOrientation: binding.restOrientation,
                restDirection: binding.restDirection,
                targetDirection: targetDirection
            )
        }
    }
}
