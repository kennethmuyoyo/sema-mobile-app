//
//  DebugCameraLandmarkPiP.swift
//  sema
//
//  Debug-only front-camera PiP with landmark skeleton overlay.
//  Delete this file and remove the overlay in ConversationScreenView when done.
//

import AVFoundation
import SwiftUI
import UIKit

enum DebugFeatures {
    static let cameraLandmarkPiP = true
    /// Reveal the gloss-picker sheet on a long-press of the LiveTopBar. Lets
    /// us play any token in the bundled PoseLibrary on the avatar without
    /// running ASR — the "controlled generation" surface.
    static let glossPlayer = true
}

struct DebugCameraLandmarkPiP: View {
    let camera: CameraSessionController
    let landmarkFrame: NormalizedFrame?
    /// When non-nil, the PiP becomes the activation surface: tapping it flips
    /// between listening (camera off) and watching (camera on). When nil, the
    /// PiP is purely informational (the legacy debug-overlay behaviour).
    var onTap: (() -> Void)? = nil
    /// True when the host has put the session into watching mode (camera
    /// active for sign capture). Drives the visual treatment and hint text.
    var isWatching: Bool = false

    private let pipSize = CGSize(width: 160, height: 220)

    var body: some View {
        ZStack(alignment: .bottom) {
            ZStack {
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(Color.black.opacity(0.55))

                if camera.status == .running {
                    DebugCameraPreviewView(session: camera.session)
                    DebugLandmarkSkeletonOverlay(frame: landmarkFrame)
                } else {
                    placeholder
                }
            }
            .frame(width: pipSize.width, height: pipSize.height)
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))

            statusStrip
        }
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(borderColor, lineWidth: isWatching ? 2 : 1)
        }
        .shadow(color: .black.opacity(0.35), radius: 12, y: 4)
        .contentShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .onTapGesture {
            onTap?()
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(isWatching ? "Camera active for sign capture" : "Camera preview")
        .accessibilityHint(onTap == nil
            ? ""
            : (isWatching ? "Double tap to stop signing and listen." : "Double tap to start signing."))
        .allowsHitTesting(onTap != nil)
    }

    private var borderColor: Color {
        isWatching ? Color.green.opacity(0.85) : Color.white.opacity(0.22)
    }

    private var placeholder: some View {
        VStack(spacing: 8) {
            Image(systemName: placeholderSymbol)
                .font(.title2)
                .foregroundStyle(.white.opacity(0.75))
            Text(placeholderText)
                .font(.callout.weight(.medium))
                .foregroundStyle(.white.opacity(0.78))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 10)
            if onTap != nil, isCameraAvailable {
                Text("Tap to sign")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(Color.white.opacity(0.18), in: Capsule())
            }
        }
    }

    /// True when the camera *could* be turned on (i.e. neither permanently
    /// denied nor in a failed state). `.failed` carries an associated
    /// `String`, so we pattern-match rather than `!= .failed`.
    private var isCameraAvailable: Bool {
        switch camera.status {
        case .denied, .failed:
            return false
        case .idle, .requestingPermission, .starting, .running:
            return true
        }
    }

    private var placeholderSymbol: String {
        switch camera.status {
        case .denied, .failed:
            return "exclamationmark.triangle.fill"
        default:
            return "video.slash"
        }
    }

    private var placeholderText: String {
        switch camera.status {
        case .idle:
            return "Camera off"
        case .requestingPermission:
            return "Allow camera"
        case .starting:
            return "Starting…"
        case .running:
            return "Live"
        case .denied:
            return "Denied"
        case .failed:
            return "Unavailable"
        }
    }

    private var statusStrip: some View {
        Text(statusLine)
            .font(.system(size: 9, weight: .semibold, design: .monospaced))
            .foregroundStyle(.white.opacity(0.92))
            .lineLimit(1)
            .minimumScaleFactor(0.7)
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .frame(maxWidth: pipSize.width)
            .background(Color.black.opacity(0.62))
    }

    private var statusLine: String {
        let cam = cameraStatusLabel
        guard let landmarkFrame else {
            return "\(cam) · no landmarks"
        }
        let age = Date().timeIntervalSince1970 - landmarkFrame.timestamp
        let landmarks = age < 0.5 ? "lm OK" : "lm stale"
        return "\(cam) · \(landmarks)"
    }

    private var cameraStatusLabel: String {
        switch camera.status {
        case .idle: return "cam idle"
        case .requestingPermission: return "cam perm"
        case .starting: return "cam …"
        case .running: return "cam run"
        case .denied: return "cam DENY"
        case .failed: return "cam ERR"
        }
    }
}

// MARK: - Camera preview

private struct DebugCameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> DebugPreviewUIView {
        let view = DebugPreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        FrontCameraMirroring.apply(to: view.previewLayer.connection)
        return view
    }

    func updateUIView(_ uiView: DebugPreviewUIView, context: Context) {
        if uiView.previewLayer.session !== session {
            uiView.previewLayer.session = session
        }
        FrontCameraMirroring.apply(to: uiView.previewLayer.connection)
    }
}

private final class DebugPreviewUIView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }
}

// The DebugLandmarkSkeletonOverlay + LandmarkSkeletonEdges live in
// DebugLandmarkSkeleton.swift so the MediaPipe template preview can reuse
// them without duplicating the projection / skeleton-edge code.
