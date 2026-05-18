import AVFoundation
import Foundation

extension Notification.Name {
    /// Posted when the user double-presses a hardware volume button.
    static let shortcutStartConversation = Notification.Name("sema.shortcut.startConversation")
    /// Legacy names — both map to starting the unified conversation session.
    static let shortcutStartListening = Notification.Name("sema.shortcut.startListening")
    static let shortcutStartInterpreting = Notification.Name("sema.shortcut.startInterpreting")
}

/// Observes `AVAudioSession.outputVolume` to detect hardware volume-button presses.
/// Two same-direction presses within `window` seconds fire a notification.
///
/// Caveat: at min/max volume the system saturates and KVO won't fire for the
/// out-of-range press. A `HiddenVolumeHUDView` in the view tree suppresses the
/// system volume HUD but does not change saturation behavior.
final class VolumeShortcutDetector {
    private var observation: NSKeyValueObservation?
    private var lastUpAt: Date?
    private var lastDownAt: Date?
    private let window: TimeInterval = 0.7

    func start() {
        guard observation == nil else { return }
        let session = AVAudioSession.sharedInstance()
        observation = session.observe(\.outputVolume, options: [.new, .old]) { [weak self] _, change in
            guard let new = change.newValue, let old = change.oldValue else { return }
            let delta = new - old
            guard abs(delta) > 0.001 else { return }
            DispatchQueue.main.async { self?.handle(delta: delta) }
        }
    }

    func stop() {
        observation?.invalidate()
        observation = nil
        lastUpAt = nil
        lastDownAt = nil
    }

    private func handle(delta: Float) {
        let now = Date()
        if delta > 0 {
            if let last = lastUpAt, now.timeIntervalSince(last) < window {
                postStartConversation()
                lastUpAt = nil
            } else {
                lastUpAt = now
                lastDownAt = nil
            }
        } else {
            if let last = lastDownAt, now.timeIntervalSince(last) < window {
                postStartConversation()
                lastDownAt = nil
            } else {
                lastDownAt = now
                lastUpAt = nil
            }
        }
    }

    private func postStartConversation() {
        NotificationCenter.default.post(name: .shortcutStartConversation, object: nil)
        NotificationCenter.default.post(name: .shortcutStartListening, object: nil)
        NotificationCenter.default.post(name: .shortcutStartInterpreting, object: nil)
    }
}

