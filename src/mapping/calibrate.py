"""
Interactive homography calibration.
Opens a frame, lets you click on visible pitch landmarks,
then tests the calibration by mapping player positions to pitch metres.
"""
import cv2
import numpy as np
import json
import os

from src.mapping.homography import HomographyEstimator
from src.mapping.pitch_model import KEYPOINTS, draw_pitch_template, is_outside_box, SCALE


# keypoints visible in a wide right-end shot like frame_007000.jpg
# listed in the order you will click them
CLICK_ORDER = [
    "right_box_top_left",       # 1. far-left corner of box, top side
    "right_box_top_right",      # 2. goal line corner of box, top side
    "right_box_bottom_right",   # 3. goal line corner of box, bottom side
    "right_box_bottom_left",    # 4. far-left corner of box, bottom side
    "right_penalty_spot",       # 5. small dot in centre of box
]


def run_calibration(frame_path: str, save_path: str = "outputs/homography.json") -> HomographyEstimator:
    frame = cv2.imread(frame_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load: {frame_path}")

    print("\n=== PitchIQ Homography Calibration ===")
    print(f"Frame: {frame_path}")
    print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
    print("\nClick these points IN ORDER on the frame that opens:")
    for i, name in enumerate(CLICK_ORDER, 1):
        coords = KEYPOINTS[name]
        print(f"  {i}. {name:<35} ({coords[0]}m, {coords[1]}m)")
    print("\nLeft-click to mark each point. Press 'q' or 'ESC' to finish early.")
    print("Opening frame...\n")

    clicked_points = []
    display_frame  = frame.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = len(clicked_points)
            if idx >= len(CLICK_ORDER):
                return

            clicked_points.append((x, y))
            name = CLICK_ORDER[idx]

            # draw red dot + number label
            cv2.circle(display_frame, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(display_frame, str(idx + 1), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Calibration — click landmarks in order", display_frame)

            print(f"  ✓ {idx + 1}. {name}: pixel ({x}, {y})")

            if len(clicked_points) == len(CLICK_ORDER):
                print("\nAll points collected. Press any key to continue.")

    cv2.namedWindow("Calibration — click landmarks in order", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration — click landmarks in order", 1280, 720)
    cv2.setMouseCallback("Calibration — click landmarks in order", on_click)
    cv2.imshow("Calibration — click landmarks in order", display_frame)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27) or len(clicked_points) == len(CLICK_ORDER):
            break

    cv2.destroyAllWindows()

    if len(clicked_points) < 4:
        raise ValueError(f"Need at least 4 points, only got {len(clicked_points)}")

    # calibrate
    estimator = HomographyEstimator()
    names_used = CLICK_ORDER[:len(clicked_points)]
    estimator.calibrate(clicked_points, names_used)

    # save clicked points for reuse
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({
            "frame":   frame_path,
            "points":  clicked_points,
            "names":   names_used,
            "matrix":  estimator._matrix.tolist() if estimator._matrix is not None else None,
        }, f, indent=2)
    print(f"\nCalibration saved to {save_path}")

    return estimator


def test_calibration(estimator: HomographyEstimator, frame_path: str) -> None:
    """
    Draw player positions mapped to the pitch template so you can
    visually verify the homography looks correct.
    """
    from src.detection.detector import load_model, detect_objects, is_play_frame, filter_pitch_zone
    from src.tracking.tracker import build_tracker, detections_to_sv, sv_to_tracks

    frame = cv2.imread(frame_path)
    model = load_model("yolov8n.pt")
    tracker = build_tracker()

    detections = detect_objects(model, frame)
    if not is_play_frame(detections):
        print("Frame filtered as non-play — try a different frame")
        return

    detections = filter_pitch_zone(detections, frame.shape[0])
    sv_dets    = detections_to_sv(detections)
    sv_tracked = tracker.update_with_detections(sv_dets)

    from src.tracking.tracker import sv_to_tracks
    tracks = sv_to_tracks(sv_tracked, model.names)

    template = draw_pitch_template()
    pad = 20

    print("\n=== Player positions on pitch ===")
    for track in tracks:
        if track.class_name != "person":
            continue

        # use feet position (bottom-center of bounding box)
        feet_x = (track.x1 + track.x2) // 2
        feet_y = track.y2

        pitch_coords = estimator.to_pitch_coords(feet_x, feet_y)
        if pitch_coords is None:
            continue

        x_m, y_m = pitch_coords
        outside   = is_outside_box(x_m, y_m, attacking_end="right")

        print(f"  track={track.track_id} | feet=({feet_x},{feet_y}) → ({x_m:.1f}m, {y_m:.1f}m) | outside box: {outside}")

        # plot on template
        tx = int(x_m * SCALE) + pad
        ty = int(y_m * SCALE) + pad
        if 0 <= tx < template.shape[1] and 0 <= ty < template.shape[0]:
            cv2.circle(template, (tx, ty), 5, (0, 255, 255), -1)
            cv2.putText(template, str(track.track_id), (tx + 6, ty - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/pitch_positions.jpg", template)
    print("\nSaved outputs/pitch_positions.jpg — player dots on pitch map")
    print("Run: open outputs/pitch_positions.jpg")


if __name__ == "__main__":
    FRAME = "outputs/frames/frame_012000.jpg"

    estimator = run_calibration(FRAME)
    test_calibration(estimator, FRAME)
