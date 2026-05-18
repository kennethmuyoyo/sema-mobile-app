import AVFoundation

/// Single mirror policy for the front camera across capture, preview, and overlays.
/// Ignores the system "Mirror Front Camera" setting by disabling automatic adjustment.
enum FrontCameraMirroring {
    /// Selfie-style mirror — matches `AVCaptureConnection.isVideoMirrored` on the frame output.
    static let isEnabled = true

    /// Apply to a capture connection (video data output or preview layer).
    static func apply(to connection: AVCaptureConnection?) {
        guard let connection, connection.isVideoMirroringSupported else { return }
        connection.automaticallyAdjustsVideoMirroring = false
        connection.isVideoMirrored = isEnabled
    }

    /// Project normalized landmark x into PiP width (same mirrored space as capture + preview).
    static func landmarkX(jointX: Float, centerX: CGFloat, scale: CGFloat) -> CGFloat {
        let offset = CGFloat(jointX) * scale
        return isEnabled ? centerX + offset : centerX - offset
    }
}
