import SwiftUI

struct AvatarCanvasView: View {
    let frame: DemoPoseFrame
    let isActive: Bool

    var body: some View {
        Canvas { context, size in
            let joints = Dictionary(uniqueKeysWithValues: frame.joints.map { joint in
                (joint.id, joint)
            })
            let points = joints.mapValues { $0.cgPoint(in: size) }

            drawHandTrails(frame.handTrails, in: &context, size: size)
            drawLegs(with: points, in: &context)
            drawTorso(with: points, in: &context)
            drawHead(with: points, in: &context)
            drawArms(with: points, in: &context)
            drawHands(with: points, in: &context)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(isActive ? "Human-like signing avatar animation active" : "Human-like signing avatar ready")
        .accessibilityValue(isActive ? "Hands and arms are animating" : "Ready to translate")
    }

    private func drawHandTrails(
        _ trails: [DemoMotionTrail],
        in context: inout GraphicsContext,
        size: CGSize
    ) {
        for trail in trails {
            let scaledPoints = trail.points.map { $0.cgPoint(in: size) }

            for index in 1..<scaledPoints.count {
                var path = Path()
                path.move(to: scaledPoints[index - 1])
                path.addLine(to: scaledPoints[index])

                context.stroke(path, with: .color(.cyan.opacity(Double(index) * 0.05)), style: .avatarLine(width: CGFloat(index) * 2))
            }
        }
    }

    private func drawTorso(with points: [String: CGPoint], in context: inout GraphicsContext) {
        guard
            let leftShoulder = points["leftShoulder"],
            let rightShoulder = points["rightShoulder"],
            let leftHip = points["leftHip"],
            let rightHip = points["rightHip"]
        else {
            return
        }

        let shoulderMidpoint = leftShoulder.midpoint(to: rightShoulder)
        let hipMidpoint = leftHip.midpoint(to: rightHip)
        let waistInset = abs(rightShoulder.x - leftShoulder.x) * 0.08

        var torso = Path()
        torso.move(to: leftShoulder)
        torso.addQuadCurve(
            to: rightShoulder,
            control: CGPoint(x: shoulderMidpoint.x, y: shoulderMidpoint.y - 12)
        )
        torso.addLine(to: CGPoint(x: rightHip.x - waistInset, y: rightHip.y))
        torso.addQuadCurve(
            to: CGPoint(x: leftHip.x + waistInset, y: leftHip.y),
            control: CGPoint(x: hipMidpoint.x, y: hipMidpoint.y + 16)
        )
        torso.closeSubpath()

        context.fill(torso, with: .color(.white.opacity(0.22)))
        context.stroke(
            torso,
            with: .color(.white.opacity(0.30)),
            style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round)
        )
    }

    private func drawHead(with points: [String: CGPoint], in context: inout GraphicsContext) {
        guard
            let head = points["head"],
            let leftShoulder = points["leftShoulder"],
            let rightShoulder = points["rightShoulder"]
        else {
            return
        }

        let shoulderWidth = leftShoulder.distance(to: rightShoulder)
        let headWidth = max(38, shoulderWidth * 0.34)
        let headHeight = headWidth * 1.16
        let headRect = CGRect(
            x: head.x - headWidth / 2,
            y: head.y - headHeight / 2,
            width: headWidth,
            height: headHeight
        )

        let headPath = Path(ellipseIn: headRect)
        context.fill(headPath, with: .color(.white.opacity(0.28)))
        context.stroke(headPath, with: .color(.white.opacity(0.36)), lineWidth: 2)

        if let leftEye = points["left_eye_smplhf"], let rightEye = points["right_eye_smplhf"] {
            drawDot(at: leftEye, radius: 2.5, color: .black.opacity(0.32), in: &context)
            drawDot(at: rightEye, radius: 2.5, color: .black.opacity(0.32), in: &context)
        }
    }

    private func drawLegs(with points: [String: CGPoint], in context: inout GraphicsContext) {
        drawLimb(["leftHip", "leftKnee", "leftAnkle"], width: 10, color: .white.opacity(0.16), points: points, in: &context)
        drawLimb(["rightHip", "rightKnee", "rightAnkle"], width: 10, color: .white.opacity(0.16), points: points, in: &context)
    }

    private func drawArms(with points: [String: CGPoint], in context: inout GraphicsContext) {
        drawLimb(["leftShoulder", "leftElbow", "left_body_wrist", "left_wrist"], width: 18, color: .white.opacity(0.38), points: points, in: &context)
        drawLimb(["rightShoulder", "rightElbow", "right_body_wrist", "right_wrist"], width: 18, color: .white.opacity(0.38), points: points, in: &context)
    }

    private func drawHands(with points: [String: CGPoint], in context: inout GraphicsContext) {
        drawHand(prefix: "left", points: points, in: &context)
        drawHand(prefix: "right", points: points, in: &context)
    }

    private func drawHand(prefix: String, points: [String: CGPoint], in context: inout GraphicsContext) {
        guard
            let wrist = points["\(prefix)_wrist"],
            let index = points["\(prefix)_index1"],
            let middle = points["\(prefix)_middle1"],
            let pinky = points["\(prefix)_pinky1"]
        else {
            return
        }

        drawPalm(wrist: wrist, index: index, middle: middle, pinky: pinky, in: &context)

        for finger in ["thumb", "index", "middle", "ring", "pinky"] {
            drawFinger(prefix: prefix, finger: finger, points: points, in: &context)
        }
    }

    private func drawPalm(
        wrist: CGPoint,
        index: CGPoint,
        middle: CGPoint,
        pinky: CGPoint,
        in context: inout GraphicsContext
    ) {
        let center = CGPoint(
            x: (wrist.x + index.x + middle.x + pinky.x) / 4,
            y: (wrist.y + index.y + middle.y + pinky.y) / 4
        )
        let palmWidth = max(34, index.distance(to: pinky) * 1.5)
        let palmHeight = max(42, wrist.distance(to: middle) * 1.55)
        let palmRect = CGRect(
            x: center.x - palmWidth / 2,
            y: center.y - palmHeight / 2,
            width: palmWidth,
            height: palmHeight
        )

        let palm = Path(roundedRect: palmRect, cornerRadius: palmWidth * 0.42)
        context.fill(palm, with: .color(.white.opacity(0.86)))
        context.stroke(palm, with: .color(.cyan.opacity(0.55)), lineWidth: 2)
    }

    private func drawFinger(
        prefix: String,
        finger: String,
        points: [String: CGPoint],
        in context: inout GraphicsContext
    ) {
        let names = [
            "\(prefix)_wrist",
            "\(prefix)_\(finger)1",
            "\(prefix)_\(finger)2",
            "\(prefix)_\(finger)3"
        ]

        guard let first = points[names[0]] else {
            return
        }

        var previous = first

        for (index, name) in names.dropFirst().enumerated() {
            guard let current = points[name] else {
                return
            }

            let width = max(7, 12 - CGFloat(index) * 1.8)
            drawCapsule(
                from: previous,
                to: current,
                width: width,
                fill: .white.opacity(0.88),
                outline: .cyan.opacity(finger == "thumb" ? 0.42 : 0.58),
                in: &context
            )
            previous = current
        }
    }

    private func drawLimb(
        _ names: [String],
        width: CGFloat,
        color: Color,
        points: [String: CGPoint],
        in context: inout GraphicsContext
    ) {
        for pair in zip(names, names.dropFirst()) {
            guard let start = points[pair.0], let end = points[pair.1] else {
                continue
            }

            drawCapsule(from: start, to: end, width: width, fill: color, outline: .white.opacity(0.18), in: &context)
        }
    }

    private func drawCapsule(
        from start: CGPoint,
        to end: CGPoint,
        width: CGFloat,
        fill: Color,
        outline: Color,
        in context: inout GraphicsContext
    ) {
        var path = Path()
        path.move(to: start)
        path.addLine(to: end)

        context.stroke(
            path,
            with: .color(fill),
            style: .avatarLine(width: width)
        )
        context.stroke(
            path,
            with: .color(outline),
            style: .avatarLine(width: 2)
        )
    }

    private func drawDot(at point: CGPoint, radius: CGFloat, color: Color, in context: inout GraphicsContext) {
        let rect = CGRect(x: point.x - radius, y: point.y - radius, width: radius * 2, height: radius * 2)
        context.fill(Circle().path(in: rect), with: .color(color))
    }
}

private extension DemoPoseJoint {
    func cgPoint(in size: CGSize) -> CGPoint {
        CGPoint(
            x: position.x * size.width,
            y: position.y * size.height
        )
    }
}

private extension CGPoint {
    func cgPoint(in size: CGSize) -> CGPoint {
        CGPoint(x: x * size.width, y: y * size.height)
    }

    func midpoint(to other: CGPoint) -> CGPoint {
        CGPoint(x: (x + other.x) / 2, y: (y + other.y) / 2)
    }

    func distance(to other: CGPoint) -> CGFloat {
        hypot(x - other.x, y - other.y)
    }
}

private extension StrokeStyle {
    static func avatarLine(width: CGFloat) -> StrokeStyle {
        StrokeStyle(lineWidth: width, lineCap: .round, lineJoin: .round)
    }
}

#Preview {
    AvatarCanvasView(frame: .sample(phase: 0.4), isActive: true)
        .padding()
}
