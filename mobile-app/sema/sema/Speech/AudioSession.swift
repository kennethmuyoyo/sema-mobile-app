import AVFoundation
import Foundation

/// Single shared point of `AVAudioSession` configuration. Per
/// `mobile-app/docs/path_b_avatar.md` and `generation/asr/contract.md`.
@MainActor
final class AudioSession {
    static let shared = AudioSession()

    private let session = AVAudioSession.sharedInstance()
    private(set) var isActive = false

    var hasRecordPermission: Bool {
        if #available(iOS 17.0, *) {
            return AVAudioApplication.shared.recordPermission == .granted
        }
        return session.recordPermission == .granted
    }

    @discardableResult
    func requestRecordPermission() async -> Bool {
        if #available(iOS 17.0, *) {
            switch AVAudioApplication.shared.recordPermission {
            case .granted:
                return true
            case .denied:
                return false
            case .undetermined:
                return await AVAudioApplication.requestRecordPermission()
            @unknown default:
                return false
            }
        }
        switch session.recordPermission {
        case .granted:
            return true
        case .denied:
            return false
        case .undetermined:
            return await withCheckedContinuation { continuation in
                session.requestRecordPermission { continuation.resume(returning: $0) }
            }
        @unknown default:
            return false
        }
    }

    func configure() throws {
        guard hasRecordPermission else {
            throw AudioSessionError.microphonePermissionDenied
        }

        // `.videoChat` keeps mic + front camera stable together (`.measurement` fights capture).
        try session.setCategory(
            .playAndRecord,
            mode: .videoChat,
            options: [.defaultToSpeaker, .allowBluetooth, .mixWithOthers]
        )
        // Match typical voice-chat hardware rate (device often reports 24 kHz).
        try session.setPreferredSampleRate(24_000)
        try session.setPreferredIOBufferDuration(0.02)
        try session.setActive(true, options: .notifyOthersOnDeactivation)
        isActive = true
    }

    /// Legacy playback-only category. Prefer `configure()` for the unified
    /// conversation screen; ASR pauses via `TTSGate` while TTS runs on the
    /// same `.playAndRecord` session. Kept for isolated TTS debugging.
    func configureForPlayback() throws {
        try session.setCategory(
            .playback,
            mode: .spokenAudio,
            options: [.duckOthers, .mixWithOthers]
        )
        try session.setActive(true, options: .notifyOthersOnDeactivation)
        isActive = true
    }

    func deactivate() {
        do {
            try session.setActive(false, options: .notifyOthersOnDeactivation)
            isActive = false
        } catch {
            // Best-effort; logging only.
        }
    }
}

enum AudioSessionError: LocalizedError {
    case microphonePermissionDenied

    var errorDescription: String? {
        switch self {
        case .microphonePermissionDenied:
            return "Microphone access is not allowed."
        }
    }
}
