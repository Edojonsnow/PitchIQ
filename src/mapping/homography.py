import cv2
import numpy as np
from src.mapping.pitch_model import KEYPOINTS, SCALE, metres_to_pixels


class HomographyEstimator:
    """
    Computes and applies a homography matrix that maps
    camera frame pixel coordinates → real pitch coordinates in metres.

    Usage:
        estimator = HomographyEstimator()
        estimator.calibrate(frame_points, pitch_keypoint_names)
        pitch_coords = estimator.to_pitch_coords(pixel_x, pixel_y)
    """

    def __init__(self):
        self._matrix: np.ndarray | None = None
        self._is_calibrated = False

    def calibrate(
        self,
        frame_points: list[tuple[int, int]],
        keypoint_names: list[str],
    ) -> bool:
        """
        Compute the homography matrix from point correspondences.

        frame_points   : pixel coordinates in the camera frame
                         e.g. [(234, 456), (891, 234), ...]
        keypoint_names : matching names from pitch_model.KEYPOINTS
                         e.g. ["right_box_top_left", "right_box_top_right", ...]

        Requires at least 4 point pairs. More = more accurate.
        Returns True if calibration succeeded.
        """
        if len(frame_points) < 4 or len(frame_points) != len(keypoint_names):
            print("Need at least 4 matching point pairs")
            return False

        # source: pixel coords in the camera frame
        src = np.array(frame_points, dtype=np.float32)

        # destination: pitch coords converted to template pixels
        dst = np.array(
            [metres_to_pixels(*KEYPOINTS[name]) for name in keypoint_names],
            dtype=np.float32,
        )

        # RANSAC makes it robust to a few bad point matches
        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        if matrix is None:
            print("Homography computation failed")
            return False

        self._matrix = matrix
        self._is_calibrated = True
        inliers = int(np.sum(mask))
        print(f"Homography calibrated — {inliers}/{len(frame_points)} inliers")
        return True

    def to_pitch_coords(self, pixel_x: int, pixel_y: int) -> tuple[float, float] | None:
        """
        Map a single pixel coordinate to pitch metres.
        Returns (x_metres, y_metres) or None if not calibrated.
        """
        if not self._is_calibrated:
            return None

        pt = np.array([[[float(pixel_x), float(pixel_y)]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._matrix)
        px, py = result[0][0]

        # convert template pixels back to metres
        return px / SCALE, py / SCALE

    def to_pitch_coords_bulk(self, points: list[tuple[int, int]]) -> list[tuple[float, float]]:
        """Map a list of pixel coordinates to pitch metres in one call."""
        if not self._is_calibrated or not points:
            return []

        pts = np.array([[list(p)] for p in points], dtype=np.float32)
        results = cv2.perspectiveTransform(pts, self._matrix)
        return [(r[0][0] / SCALE, r[0][1] / SCALE) for r in results]

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated


def calibrate_from_frame(frame: np.ndarray) -> HomographyEstimator:
    """
    Interactive calibration — shows the frame and lets you click
    corresponding pitch keypoints to build the homography.
    Press 'q' to finish selecting points.
    """
    print("\nInteractive calibration")
    print("You will click points on the frame that match known pitch locations.")
    print()

    available = list(KEYPOINTS.keys())
    for i, name in enumerate(available):
        coords = KEYPOINTS[name]
        print(f"  [{i:>2}] {name:<35} ({coords[0]}m, {coords[1]}m)")

    print("\nEnter the indices of the keypoints you will click (comma-separated):")
    print("Recommended: pick 6-8 clearly visible points from the current camera angle")
    indices_input = input("Keypoint indices: ").strip()
    selected_indices = [int(x.strip()) for x in indices_input.split(",")]
    selected_names   = [available[i] for i in selected_indices]

    print(f"\nNow click each of these points ON THE FRAME in this order:")
    for i, name in enumerate(selected_names):
        print(f"  {i+1}. {name}")
    print("\nClose the window when done (or press 'q')")

    clicked_points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < len(selected_names):
            clicked_points.append((x, y))
            label = selected_names[len(clicked_points) - 1]
            print(f"  Clicked {label}: ({x}, {y})")
            cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display_frame, str(len(clicked_points)),
                        (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Calibration", display_frame)

    display_frame = frame.copy()
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_click)
    cv2.imshow("Calibration", display_frame)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(clicked_points) == len(selected_names):
            break

    cv2.destroyAllWindows()

    estimator = HomographyEstimator()
    if len(clicked_points) >= 4:
        estimator.calibrate(clicked_points, selected_names[:len(clicked_points)])
    else:
        print("Not enough points clicked — need at least 4")

    return estimator


if __name__ == "__main__":
    import os
    from src.mapping.pitch_model import draw_pitch_template, is_outside_box

    # --- test with synthetic known points ---
    # simulate a camera looking at the right half of the pitch
    # these would normally come from clicking on a real frame
    print("=== Homography self-test with synthetic points ===\n")

    estimator = HomographyEstimator()

    # fake camera pixel coords → real pitch keypoints
    # in a real run these come from clicking on the frame
    synthetic_frame_points = [
        (320, 180),   # right_box_top_left
        (960, 180),   # right_box_top_right
        (320, 540),   # right_box_bottom_left
        (960, 540),   # right_box_bottom_right
        (640, 360),   # right_penalty_spot
    ]
    synthetic_keypoints = [
        "right_box_top_left",
        "right_box_top_right",
        "right_box_bottom_left",
        "right_box_bottom_right",
        "right_penalty_spot",
    ]

    success = estimator.calibrate(synthetic_frame_points, synthetic_keypoints)

    if success:
        # test: penalty spot pixel → should map to ~(94m, 34m)
        result = estimator.to_pitch_coords(640, 360)
        print(f"\nPenalty spot pixel (640, 360) → {result[0]:.1f}m, {result[1]:.1f}m")
        print(f"Expected: ({94.0}m, {34.0}m)")

        # test outside box detection
        x_m, y_m = result
        print(f"Is outside box? {is_outside_box(x_m, y_m, 'right')}")

    # render and save pitch template
    template = draw_pitch_template()
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/pitch_template.jpg", template)
    print("\nSaved pitch template to outputs/pitch_template.jpg")
    print("Run: open outputs/pitch_template.jpg")
