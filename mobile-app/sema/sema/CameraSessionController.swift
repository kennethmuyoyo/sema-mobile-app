@preconcurrency import AVFoundation
import Foundation
import Observation

@MainActor
@Observable
final class CameraSessionController {
    enum Status: Equatable {
        case idle
        case requestingPermission
        case starting
        case running
        case denied
        case failed(String)
    }

    let session = AVCaptureSession()
    var status: Status = .idle

    @ObservationIgnored private var isConfigured = false

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndStart()
        case .notDetermined:
            requestPermission()
        case .denied, .restricted:
            status = .denied
        @unknown default:
            status = .failed("Camera permission is unavailable.")
        }
    }

    func stop() {
        if session.isRunning {
            session.stopRunning()
        }

        status = .idle
    }

    private func requestPermission() {
        status = .requestingPermission

        AVCaptureDevice.requestAccess(for: .video) { [weak self] isGranted in
            Task { @MainActor in
                guard let self else { return }

                if isGranted {
                    self.configureAndStart()
                } else {
                    self.status = .denied
                }
            }
        }
    }

    private func configureAndStart() {
        guard status != .running && status != .starting else { return }

        status = .starting

        do {
            if !isConfigured {
                try configureSession()
            }

            if !session.isRunning {
                session.startRunning()
            }

            status = .running
        } catch {
            status = .failed(error.localizedDescription)
        }
    }

    private func configureSession() throws {
        session.beginConfiguration()
        session.sessionPreset = .medium
        defer { session.commitConfiguration() }

        guard
            let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
        else {
            throw CameraSessionError.frontCameraUnavailable
        }

        let input = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(input) else {
            throw CameraSessionError.cannotAddInput
        }

        session.addInput(input)
        isConfigured = true
    }
}

private enum CameraSessionError: LocalizedError {
    case frontCameraUnavailable
    case cannotAddInput

    var errorDescription: String? {
        switch self {
        case .frontCameraUnavailable:
            return "Front camera is unavailable."
        case .cannotAddInput:
            return "Could not attach the front camera."
        }
    }
}
