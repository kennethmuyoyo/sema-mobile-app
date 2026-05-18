//
//  DebugLandmarkSkeleton.swift
//  sema
//
//  Shared SwiftUI Canvas that draws a 45-joint NormalizedFrame as a wire
//  skeleton — head + shoulders, arm chains, finger chains. Used by both the
//  live camera PiP (DebugCameraLandmarkPiP) and the canned MediaPipe template
//  preview (DebugMediaPipePreview). Mirror policy matches what the front
//  camera produces so the canned and live views read the same way.
//

import SwiftUI

struct DebugLandmarkSkeletonOverlay: View {
    let frame: NormalizedFrame?
    /// If true, project landmarks as if they came from the user-facing front
    /// camera (horizontal mirror). False = no mirror — used by the canned
    /// template preview so the reference looks like the rendered SMPL-X
    /// avatar, not a selfie.
    var mirrored: Bool = true

    var body: some View {
        Canvas { context, size in
            guard let frame else { return }
            var bonePath = Path()
            for edge in LandmarkSkeletonEdges.pairs {
                guard let a = project(jointIndex: edge.0, frame: frame, size: size),
                      let b = project(jointIndex: edge.1, frame: frame, size: size)
                else { continue }
                bonePath.move(to: a)
                bonePath.addLine(to: b)
            }
            context.stroke(
                bonePath,
                with: .color(Design.BrandColor.accent.opacity(0.9)),
                lineWidth: 1.5
            )

            for index in 0..<Landmark45.count {
                guard frame.mask[index] > 0,
                      let point = project(jointIndex: index, frame: frame, size: size)
                else { continue }
                let dot = CGRect(x: point.x - 2, y: point.y - 2, width: 4, height: 4)
                context.fill(Path(ellipseIn: dot), with: .color(.white))
            }
        }
        .allowsHitTesting(false)
    }

    private func project(jointIndex: Int, frame: NormalizedFrame, size: CGSize) -> CGPoint? {
        guard frame.mask[jointIndex] > 0 else { return nil }
        let joint = frame.joint(at: jointIndex)
        let scale = min(size.width, size.height) * 0.38
        let centerX = size.width * 0.5
        let centerY = size.height * 0.40
        let xWorld: CGFloat = mirrored
            ? FrontCameraMirroring.landmarkX(jointX: joint.x, centerX: centerX, scale: scale)
            : centerX + CGFloat(joint.x) * scale
        return CGPoint(
            x: xWorld,
            y: centerY + CGFloat(joint.y) * scale
        )
    }
}

enum LandmarkSkeletonEdges {
    static let pairs: [(Int, Int)] = {
        func idx(_ name: String) -> Int { Landmark45.index(of: name) }
        func chain(_ names: [String]) -> [(Int, Int)] {
            let indices = names.map(idx)
            return zip(indices, indices.dropFirst()).map { ($0, $1) }
        }

        var edges: [(Int, Int)] = []
        edges.append((idx("left_shoulder"), idx("right_shoulder")))
        edges.append((idx("head"), idx("left_shoulder")))
        edges.append((idx("head"), idx("right_shoulder")))
        edges += chain(["left_shoulder", "left_elbow", "left_wrist"])
        edges += chain(["right_shoulder", "right_elbow", "right_wrist"])

        let fingers = ["thumb", "index", "middle", "ring", "pinky"]
        for finger in fingers {
            edges.append((idx("left_wrist"), idx("left_\(finger)1")))
            edges += chain((1...3).map { "left_\(finger)\($0)" })
        }
        for finger in fingers {
            edges.append((idx("right_wrist"), idx("right_\(finger)1")))
            edges += chain((1...3).map { "right_\(finger)\($0)" })
        }
        return edges
    }()
}
