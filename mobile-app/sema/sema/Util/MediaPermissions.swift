import AVFoundation
import Foundation
import Speech

/// Snapshot of privacy permissions needed for a live conversation session.
struct MediaPermissionSnapshot: Equatable {
    var microphone: Status = .unknown
    var speechRecognition: Status = .unknown
    var camera: Status = .unknown

    enum Status: Equatable {
        case unknown
        case granted
        case denied
        case restricted
        case unavailable
    }

    var canListen: Bool { microphone == .granted && speechRecognition == .granted }
    var canRecognizeSigns: Bool { camera == .granted }

    @MainActor
    static func capture() -> MediaPermissionSnapshot {
        var snapshot = MediaPermissionSnapshot()
        snapshot.microphone = Self.microphoneStatus()
        snapshot.speechRecognition = Self.speechStatus()
        snapshot.camera = Self.cameraStatus()
        return snapshot
    }

    @MainActor
    private static func microphoneStatus() -> Status {
        if #available(iOS 17.0, *) {
            switch AVAudioApplication.shared.recordPermission {
            case .granted: return .granted
            case .denied: return .denied
            case .undetermined: return .unknown
            @unknown default: return .unavailable
            }
        }
        switch AVAudioSession.sharedInstance().recordPermission {
        case .granted: return .granted
        case .denied: return .denied
        case .undetermined: return .unknown
        @unknown default: return .unavailable
        }
    }

    @MainActor
    private static func speechStatus() -> Status {
        switch SFSpeechRecognizer.authorizationStatus() {
        case .authorized: return .granted
        case .denied: return .denied
        case .restricted: return .restricted
        case .notDetermined: return .unknown
        @unknown default: return .unavailable
        }
    }

    @MainActor
    private static func cameraStatus() -> Status {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return .granted
        case .denied: return .denied
        case .restricted: return .restricted
        case .notDetermined: return .unknown
        @unknown default: return .unavailable
        }
    }
}
