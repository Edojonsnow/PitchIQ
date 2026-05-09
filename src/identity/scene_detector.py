import cv2
import numpy as np


def compute_frame_difference(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """
    Measure how visually different two consecutive frames are.
    Converts both to grayscale, computes absolute pixel difference,
    and returns the mean difference as a score (0.0 - 255.0).
    A high score means the frames look very different — likely a camera cut.
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_a, gray_b)
    return float(np.mean(diff))


def is_scene_cut(frame_a: np.ndarray, frame_b: np.ndarray, threshold: float = 50.0) -> bool:
    """
    Returns True if the difference between two frames exceeds the threshold.
    Raised to 50.0 — fast football gameplay scores 25–45, real cuts score 50+.
    """
    return compute_frame_difference(frame_a, frame_b) > threshold


class SceneTracker:
    """
    Maintains scene state across a stream of frames.
    Call update() on every frame — it tells you if a new scene just started
    and tracks a scene index so downstream code knows which scene it's in.

    min_scene_frames: minimum number of frames between two cuts.
    Prevents rapid-fire cut detection during fast action sequences
    where consecutive frames all score high — that's motion, not cuts.
    """

    def __init__(self, threshold: float = 50.0, min_scene_frames: int = 15):
        self.threshold = threshold
        self.min_scene_frames = min_scene_frames
        self.scene_index = 0
        self.prev_frame = None
        self.frames_since_last_cut = 0

    def update(self, frame: np.ndarray) -> tuple[bool, int]:
        """
        Returns (is_new_scene, scene_index).
        is_new_scene is True on the first frame after a genuine cut.
        """
        if self.prev_frame is None:
            self.prev_frame = frame
            return True, self.scene_index

        self.frames_since_last_cut += 1

        score = compute_frame_difference(self.prev_frame, frame)
        is_cut = (
            score > self.threshold
            and self.frames_since_last_cut >= self.min_scene_frames
        )

        if is_cut:
            self.scene_index += 1
            self.frames_since_last_cut = 0

        self.prev_frame = frame
        return is_cut, self.scene_index


if __name__ == "__main__":
    import os

    FRAMES_DIR = "outputs/frames"
    all_frames = sorted(os.listdir(FRAMES_DIR))[:200]

    tracker = SceneTracker(threshold=50.0, min_scene_frames=15)

    for filename in all_frames:
        frame = cv2.imread(os.path.join(FRAMES_DIR, filename))
        is_new, scene_idx = tracker.update(frame)
        if is_new:
            print(f"Scene {scene_idx:>3} starts at {filename}")

    print(f"\nTotal scenes detected in first 200 frames: {tracker.scene_index + 1}")
