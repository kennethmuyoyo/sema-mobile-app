import SwiftUI

struct FloatingPiPView: View {
    let isActive: Bool
    let camera: CameraSessionController
    let bottomReservedHeight: CGFloat

    @State private var restingPosition: CGPoint?
    @GestureState private var dragTranslation: CGSize = .zero

    private let pipSize = CGSize(width: 124, height: 164)
    private let margin: CGFloat = 20
    private let topReservedHeight: CGFloat = 84

    var body: some View {
        GeometryReader { proxy in
            CameraPiPView(isActive: isActive, camera: camera)
                .position(position(in: proxy.size))
                .gesture(dragGesture(in: proxy.size))
                .accessibilityHint("Drag to move the camera preview.")
                .accessibilityAction(named: "Move camera preview to top right") {
                    restingPosition = defaultPosition(in: proxy.size)
                }
                .accessibilityAction(named: "Move camera preview to bottom left") {
                    restingPosition = clamped(
                        CGPoint(
                            x: pipSize.width / 2 + margin,
                            y: proxy.size.height - bottomReservedHeight - pipSize.height / 2
                        ),
                        in: proxy.size
                    )
                }
        }
        .ignoresSafeArea(.container, edges: .bottom)
    }
}

private extension FloatingPiPView {
    func dragGesture(in containerSize: CGSize) -> some Gesture {
        DragGesture()
            .updating($dragTranslation) { value, state, _ in
                state = value.translation
            }
            .onEnded { value in
                let start = restingPosition ?? defaultPosition(in: containerSize)
                let proposed = CGPoint(
                    x: start.x + value.translation.width,
                    y: start.y + value.translation.height
                )
                restingPosition = clamped(proposed, in: containerSize)
            }
    }

    func position(in containerSize: CGSize) -> CGPoint {
        let start = restingPosition ?? defaultPosition(in: containerSize)
        let proposed = CGPoint(
            x: start.x + dragTranslation.width,
            y: start.y + dragTranslation.height
        )
        return clamped(proposed, in: containerSize)
    }

    func defaultPosition(in containerSize: CGSize) -> CGPoint {
        CGPoint(
            x: containerSize.width - pipSize.width / 2 - margin,
            y: topReservedHeight + pipSize.height / 2
        )
    }

    func clamped(_ point: CGPoint, in containerSize: CGSize) -> CGPoint {
        let halfWidth = pipSize.width / 2
        let halfHeight = pipSize.height / 2
        let minX = halfWidth + margin
        let maxX = max(minX, containerSize.width - halfWidth - margin)
        let minY = topReservedHeight + halfHeight
        let maxY = max(minY, containerSize.height - bottomReservedHeight - halfHeight)

        return CGPoint(
            x: min(max(point.x, minX), maxX),
            y: min(max(point.y, minY), maxY)
        )
    }
}

#Preview {
    ZStack {
        Color.black.ignoresSafeArea()
        FloatingPiPView(
            isActive: true,
            camera: CameraSessionController(),
            bottomReservedHeight: 248
        )
    }
}
