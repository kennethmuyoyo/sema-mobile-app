import SwiftUI

struct CameraPiPView: View {
    let isActive: Bool
    let camera: CameraSessionController

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(Color(.secondarySystemBackground))

            if isActive, camera.status == .running {
                CameraPreviewView(session: camera.session)
            } else {
                placeholder
            }
        }
        .frame(width: 124, height: 164)
        .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .stroke(Color.white.opacity(0.18), lineWidth: 1)
        }
        .shadow(color: .black.opacity(0.28), radius: 18, y: 8)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityLabel)
        .task(id: isActive) {
            if isActive {
                camera.start()
            } else {
                camera.stop()
            }
        }
    }

    private var placeholder: some View {
        VStack(spacing: 10) {
            Image(systemName: placeholderSystemImage)
                .font(.title3)
                .foregroundStyle(.secondary)

            Text(placeholderText)
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 10)
        }
    }

    private var placeholderSystemImage: String {
        switch camera.status {
        case .denied, .failed:
            return "exclamationmark.triangle.fill"
        default:
            return "video.slash.fill"
        }
    }

    private var placeholderText: String {
        guard isActive else { return "Camera off" }

        switch camera.status {
        case .idle:
            return "Camera off"
        case .requestingPermission:
            return "Allow camera"
        case .starting:
            return "Starting"
        case .running:
            return "You"
        case .denied:
            return "Camera denied"
        case .failed:
            return "Camera unavailable"
        }
    }

    private var accessibilityLabel: String {
        if isActive, camera.status == .running {
            return "Camera preview on"
        }

        return "Camera preview. \(placeholderText)"
    }
}

#Preview {
    CameraPiPView(isActive: true, camera: CameraSessionController())
        .padding()
}
