import CoreGraphics
import Foundation

struct DemoPoseJoint: Identifiable {
    let id: String
    let position: CGPoint
    let depth: CGFloat
    let isHand: Bool
}

struct DemoMotionTrail {
    let id: String
    let points: [CGPoint]
}

struct DemoPoseFrame {
    let joints: [DemoPoseJoint]
    let edges: [(String, String)]
    let handTrails: [DemoMotionTrail]

    static func sample(phase: Double) -> DemoPoseFrame {
        let wave = CGFloat(sin(phase))
        let slowWave = CGFloat(sin(phase * 0.72))
        let handWave = CGFloat(sin(phase * 1.25))
        let handLift = CGFloat(cos(phase * 0.9))

        let leftWrist = CGPoint(
            x: 0.40 + slowWave * 0.05,
            y: 0.48 - handLift * 0.04
        )
        let rightWrist = CGPoint(
            x: 0.62 - handWave * 0.08,
            y: 0.43 + wave * 0.06
        )

        let leftBodyWrist = CGPoint(
            x: leftWrist.x - 0.02,
            y: leftWrist.y + 0.02
        )
        let rightBodyWrist = CGPoint(
            x: rightWrist.x + 0.02,
            y: rightWrist.y + 0.02
        )

        let joints: [DemoPoseJoint] = [
            .init(id: "head", position: CGPoint(x: 0.50 + slowWave * 0.01, y: 0.17), depth: 0.12, isHand: false),
            .init(id: "left_eye_smplhf", position: CGPoint(x: 0.47 + slowWave * 0.01, y: 0.16), depth: 0.11, isHand: false),
            .init(id: "right_eye_smplhf", position: CGPoint(x: 0.53 + slowWave * 0.01, y: 0.16), depth: 0.11, isHand: false),
            .init(id: "leftShoulder", position: CGPoint(x: 0.36, y: 0.32), depth: 0.20, isHand: false),
            .init(id: "rightShoulder", position: CGPoint(x: 0.64, y: 0.32), depth: 0.20, isHand: false),
            .init(id: "leftElbow", position: CGPoint(x: 0.30, y: 0.43 - wave * 0.03), depth: 0.26, isHand: false),
            .init(id: "rightElbow", position: CGPoint(x: 0.72, y: 0.40 + wave * 0.04), depth: 0.24, isHand: false),
            .init(id: "left_body_wrist", position: leftBodyWrist, depth: 0.19, isHand: false),
            .init(id: "right_body_wrist", position: rightBodyWrist, depth: 0.17, isHand: false),
            .init(id: "leftHip", position: CGPoint(x: 0.42, y: 0.61), depth: 0.28, isHand: false),
            .init(id: "rightHip", position: CGPoint(x: 0.58, y: 0.61), depth: 0.28, isHand: false),
            .init(id: "leftKnee", position: CGPoint(x: 0.41, y: 0.76), depth: 0.36, isHand: false),
            .init(id: "rightKnee", position: CGPoint(x: 0.59, y: 0.76), depth: 0.36, isHand: false),
            .init(id: "leftAnkle", position: CGPoint(x: 0.40, y: 0.91), depth: 0.44, isHand: false),
            .init(id: "rightAnkle", position: CGPoint(x: 0.60, y: 0.91), depth: 0.44, isHand: false)
        ]
        + handJoints(
            prefix: "left",
            wrist: leftWrist,
            handedness: -1,
            spread: 0.82 + wave * 0.18,
            curl: 0.18 + max(0, handWave) * 0.28,
            rotation: -0.10 + slowWave * 0.16,
            depth: 0.10
        )
        + handJoints(
            prefix: "right",
            wrist: rightWrist,
            handedness: 1,
            spread: 0.62 + handLift * 0.22,
            curl: 0.10 + max(0, -handWave) * 0.42,
            rotation: 0.18 + wave * 0.18,
            depth: 0.08
        )

        return DemoPoseFrame(
            joints: joints,
            edges: bodyEdges + handEdges(prefix: "left") + handEdges(prefix: "right"),
            handTrails: [
                DemoMotionTrail(id: "left_wrist", points: trailPoints(for: leftWrist, phase: phase, handedness: -1)),
                DemoMotionTrail(id: "right_wrist", points: trailPoints(for: rightWrist, phase: phase + 0.5, handedness: 1)),
                DemoMotionTrail(id: "right_index3", points: trailPoints(for: rightWrist.offsetBy(dx: -0.01, dy: -0.09), phase: phase + 0.8, handedness: 1))
            ]
        )
    }
}

private extension DemoPoseFrame {
    static let bodyEdges: [(String, String)] = [
        ("left_eye_smplhf", "right_eye_smplhf"),
        ("head", "leftShoulder"),
        ("head", "rightShoulder"),
        ("leftShoulder", "rightShoulder"),
        ("leftShoulder", "leftHip"),
        ("rightShoulder", "rightHip"),
        ("leftHip", "rightHip"),
        ("leftHip", "leftKnee"),
        ("leftKnee", "leftAnkle"),
        ("rightHip", "rightKnee"),
        ("rightKnee", "rightAnkle"),
        ("leftShoulder", "leftElbow"),
        ("leftElbow", "left_body_wrist"),
        ("left_body_wrist", "left_wrist"),
        ("rightShoulder", "rightElbow"),
        ("rightElbow", "right_body_wrist"),
        ("right_body_wrist", "right_wrist")
    ]

    static func handEdges(prefix: String) -> [(String, String)] {
        ["thumb", "index", "middle", "ring", "pinky"].flatMap { finger in
            [
                ("\(prefix)_wrist", "\(prefix)_\(finger)1"),
                ("\(prefix)_\(finger)1", "\(prefix)_\(finger)2"),
                ("\(prefix)_\(finger)2", "\(prefix)_\(finger)3")
            ]
        }
    }

    static func handJoints(
        prefix: String,
        wrist: CGPoint,
        handedness: CGFloat,
        spread: CGFloat,
        curl: CGFloat,
        rotation: CGFloat,
        depth: CGFloat
    ) -> [DemoPoseJoint] {
        let fingers: [(name: String, lateral: CGFloat, length: CGFloat)] = [
            ("thumb", -0.070 * handedness, 0.064),
            ("index", -0.034 * handedness, 0.098),
            ("middle", 0.000, 0.108),
            ("ring", 0.034 * handedness, 0.096),
            ("pinky", 0.064 * handedness, 0.076)
        ]

        var joints: [DemoPoseJoint] = [
            .init(id: "\(prefix)_wrist", position: wrist, depth: depth, isHand: true)
        ]

        for finger in fingers {
            let base = CGPoint(
                x: wrist.x + finger.lateral * spread,
                y: wrist.y - 0.018 - abs(finger.lateral) * 0.12
            ).rotated(around: wrist, by: rotation)

            for segment in 1...3 {
                let progress = CGFloat(segment) / 3
                let sideways = finger.lateral * 0.34 * progress * curl
                let reach = finger.length * progress * (1 - curl * 0.34)
                let point = CGPoint(
                    x: base.x + sideways,
                    y: base.y - reach
                ).rotated(around: wrist, by: rotation * progress)

                joints.append(
                    .init(
                        id: "\(prefix)_\(finger.name)\(segment)",
                        position: point,
                        depth: depth + progress * 0.05,
                        isHand: true
                    )
                )
            }
        }

        return joints
    }

    static func trailPoints(for point: CGPoint, phase: Double, handedness: CGFloat) -> [CGPoint] {
        Array((1...5).map { step in
            let distance = CGFloat(step) * 0.012
            let wave = CGFloat(sin(phase - Double(step) * 0.3))
            return CGPoint(
                x: point.x + handedness * distance,
                y: point.y + distance * 0.8 + wave * 0.008
            )
        }
        .reversed())
    }
}

private extension CGPoint {
    func offsetBy(dx: CGFloat, dy: CGFloat) -> CGPoint {
        CGPoint(x: x + dx, y: y + dy)
    }

    func rotated(around origin: CGPoint, by radians: CGFloat) -> CGPoint {
        let translatedX = x - origin.x
        let translatedY = y - origin.y
        let cosValue = cos(radians)
        let sinValue = sin(radians)

        return CGPoint(
            x: origin.x + translatedX * cosValue - translatedY * sinValue,
            y: origin.y + translatedX * sinValue + translatedY * cosValue
        )
    }
}
