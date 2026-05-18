import MediaPlayer
import SwiftUI

/// Placing an `MPVolumeView` in the visible hierarchy suppresses the system
/// volume HUD overlay when the user presses the hardware volume buttons.
/// Used in tandem with `VolumeShortcutDetector` so our double-press handler
/// can fire without the HUD flashing on screen.
struct HiddenVolumeHUDView: UIViewRepresentable {
    func makeUIView(context: Context) -> MPVolumeView {
        let view = MPVolumeView(frame: CGRect(x: -1000, y: -1000, width: 1, height: 1))
        view.alpha = 0.0001
        view.isUserInteractionEnabled = false
        return view
    }

    func updateUIView(_ uiView: MPVolumeView, context: Context) {}
}
