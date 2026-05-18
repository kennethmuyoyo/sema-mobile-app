import Foundation

/// Bounded FIFO ring buffer of NormalizedFrames. Used by Path A to keep the
/// last `capacity` frames of normalized landmark features so the gloss tagger
/// can re-run inference on a sliding window without re-allocating.
///
/// The buffer is filled before the first inference is allowed; until then
/// `snapshot()` returns nil.
actor FrameRing {
    let capacity: Int
    private var storage: [NormalizedFrame] = []
    private var writeIndex: Int = 0

    init(capacity: Int) {
        self.capacity = capacity
        storage.reserveCapacity(capacity)
    }

    var count: Int { storage.count }
    var isFull: Bool { storage.count >= capacity }

    func push(_ frame: NormalizedFrame) {
        if storage.count < capacity {
            storage.append(frame)
        } else {
            storage[writeIndex] = frame
            writeIndex = (writeIndex + 1) % capacity
        }
    }

    /// FIFO snapshot of the ring (oldest first). `nil` while the ring is
    /// still warming up. The returned array is a fresh copy.
    func snapshot() -> [NormalizedFrame]? {
        guard isFull else { return nil }
        if writeIndex == 0 { return storage }
        let tail = storage[writeIndex...]
        let head = storage[..<writeIndex]
        return Array(tail) + Array(head)
    }

    func reset() {
        storage.removeAll(keepingCapacity: true)
        writeIndex = 0
    }
}
