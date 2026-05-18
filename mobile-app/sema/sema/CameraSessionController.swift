@preconcurrency import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import Observation

/// Delegates receive front-camera frames on a background queue. The pixel
/// buffer is BGRA8, matching the `kCVPixelFormatType_32BGRA` we configure on
/// the video data output.
protocol CameraFrameDelegate: AnyObject, Sendable {
    func camera(_ controller: CameraSessionController,
                didProduce pixelBuffer: CVPixelBuffer,
                at presentationTime: CMTime)
}

@MainActor
@Observable
final class CameraSessionController: NSObject {
    enum Status: Equatable {
        case idle
        case requestingPermission
        case starting
        case running
        case denied
        case failed(String)
    }

    @ObservationIgnored private lazy var captureSession = AVCaptureSession()
    var session: AVCaptureSession { captureSession }
    var status: Status = .idle

    /// Set this BEFORE calling `start()` to receive frames for processing
    /// (MediaPipe, etc.). Leaving it `nil` keeps the controller in
    /// preview-only mode.
    weak var frameDelegate: CameraFrameDelegate?

    /// Called on the main actor when capture reaches `.running` (including after
    /// the user grants camera access asynchronously).
    var onBecameRunning: (@MainActor () -> Void)?

    @ObservationIgnored private var isConfigured = false
    @ObservationIgnored private let videoOutput = AVCaptureVideoDataOutput()
    @ObservationIgnored private let videoQueue = DispatchQueue(
        label: "sema.camera.frames",
        qos: .userInitiated
    )
    @ObservationIgnored private let outputDelegate = VideoOutputDelegate()

    override init() {
        super.init()
        outputDelegate.owner = self
    }

    /// Requests camera access if needed, configures the session, and waits until
    /// capture is running or a terminal error/denial state is reached.
    @discardableResult
    func activate() async -> Status {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            await configureAndStartAsync()
        case .notDetermined:
            status = .requestingPermission
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            if granted {
                await configureAndStartAsync()
            } else {
                status = .denied
            }
        case .denied, .restricted:
            status = .denied
        @unknown default:
            status = .failed("Camera permission is unavailable.")
        }
        return status
    }

    func stop() {
        if session.isRunning {
            session.stopRunning()
        }
        status = .idle
    }

    private func configureAndStartAsync() async {
        guard status != .running else { return }
        status = .starting
        do {
            if !isConfigured {
                try configureSession()
            }
            let session = self.session
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                DispatchQueue.global(qos: .userInitiated).async {
                    if !session.isRunning {
                        session.startRunning()
                    }
                    continuation.resume()
                }
            }
            status = .running
            onBecameRunning?()
        } catch {
            status = .failed(error.localizedDescription)
        }
    }

    private func configureSession() throws {
        // Video-only capture: do not let AVFoundation reconfigure the shared
        // AVAudioSession (that breaks AVAudioEngine taps used by Path B ASR).
        session.automaticallyConfiguresApplicationAudioSession = false

        session.beginConfiguration()
        session.sessionPreset = .medium
        defer { session.commitConfiguration() }

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
            throw CameraSessionError.frontCameraUnavailable
        }
        let input = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(input) else {
            throw CameraSessionError.cannotAddInput
        }
        session.addInput(input)

        // 24 fps to match the recognizer's training cadence.
        try camera.lockForConfiguration()
        defer { camera.unlockForConfiguration() }
        let frameDuration = CMTime(value: 1, timescale: 24)
        camera.activeVideoMinFrameDuration = frameDuration
        camera.activeVideoMaxFrameDuration = frameDuration

        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.setSampleBufferDelegate(outputDelegate, queue: videoQueue)
        guard session.canAddOutput(videoOutput) else {
            throw CameraSessionError.cannotAddOutput
        }
        session.addOutput(videoOutput)

        if let connection = videoOutput.connection(with: .video) {
            FrontCameraMirroring.apply(to: connection)
            if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
        }

        isConfigured = true
    }

    fileprivate func forwardFrame(_ pixelBuffer: CVPixelBuffer, at time: CMTime) {
        // Hop off MainActor: delegate may be on any actor.
        guard let delegate = frameDelegate else { return }
        delegate.camera(self, didProduce: pixelBuffer, at: time)
    }
}

private final class VideoOutputDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    weak var owner: CameraSessionController?

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let owner = self.owner
        Task { @MainActor in
            owner?.forwardFrame(pixelBuffer, at: presentationTime)
        }
    }
}

private enum CameraSessionError: LocalizedError {
    case frontCameraUnavailable
    case cannotAddInput
    case cannotAddOutput

    var errorDescription: String? {
        switch self {
        case .frontCameraUnavailable:
            return "Front camera is unavailable."
        case .cannotAddInput:
            return "Could not attach the front camera."
        case .cannotAddOutput:
            return "Could not attach the video frame output."
        }
    }
}
