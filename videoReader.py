import cv2

class VideoReaderModule:
    """Read frames from a video and expose helper methods for windowed iteration.

    This class keeps frame-index state for sequential reads and also provides an
    independent sliding-window iterator used by scene-change detection.
    """

    def __init__(self, video_path):
        # Path is stored so we can reopen the stream for independent iterators.
        self.video_path = video_path
        # cap is the active cv2.VideoCapture used by sequential frame reads.
        self.cap = None
        # 1-based index for frames returned by get_next_frame.
        self.frame_index = 0

    def load_video(self):
        # Open the file and reset read position.
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.cap = None
            raise ValueError(f"Unable to open video at path: {self.video_path}")
        self.frame_index = 0

    def release_video(self):
        # Always release native resources when done.
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _require_loaded_video(self):
        # Guard helper so caller gets a clear error message.
        if self.cap is None:
            raise RuntimeError("Video is not loaded. Call load_video() first.")

    def get_next_frame(self):
        # Sequential frame read used by preview/testing workflows.
        self._require_loaded_video()
        success, frame = self.cap.read()
        if not success:
            return None

        self.frame_index += 1
        return frame

    def get_frame_window(self, window_size):
        """Return the next contiguous group of frames from the active stream.

        Output schema:
        - start_index: first 1-based frame index in this window
        - end_index: last 1-based frame index in this window
        - frames: list of BGR frames
        """
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer")

        frames = []
        # frame_index refers to the last emitted frame, so next starts at +1.
        start_index = self.frame_index + 1

        for _ in range(window_size):
            frame = self.get_next_frame()
            if frame is None:
                break
            frames.append(frame)

        if not frames:
            return None

        end_index = start_index + len(frames) - 1
        return {
            "start_index": start_index,
            "end_index": end_index,
            "frames": frames,
        }

    def iter_frame_windows(self, window_size=5, step=1):
        """Yield sliding windows across the full video without mutating main cap.

        A separate VideoCapture object is used so window extraction does not
        interfere with ongoing sequential reads from get_next_frame.
        """
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if step <= 0:
            raise ValueError("step must be a positive integer")

        self._require_loaded_video()
        local_cap = cv2.VideoCapture(self.video_path)
        if not local_cap.isOpened():
            raise ValueError(f"Unable to open video at path: {self.video_path}")

        # cursor is 0-based because CAP_PROP_POS_FRAMES is 0-based.
        cursor = 0
        total_frames = int(local_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        try:
            while cursor < total_frames:
                # Seek to the start of the next window.
                local_cap.set(cv2.CAP_PROP_POS_FRAMES, cursor)

                frames = []
                for _ in range(window_size):
                    success, frame = local_cap.read()
                    if not success:
                        break
                    frames.append(frame)

                if not frames:
                    break

                # Convert 0-based cursor to 1-based frame numbering for output.
                start_index = cursor + 1
                end_index = cursor + len(frames)
                yield {
                    "start_index": start_index,
                    "end_index": end_index,
                    "frames": frames,
                }

                # Move window start forward by step frames.
                cursor += step
        finally:
            local_cap.release()