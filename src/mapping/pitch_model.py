import numpy as np
import cv2

# Standard football pitch dimensions in metres (FIFA regulations)
PITCH_LENGTH = 105.0   # x-axis: goal line to goal line
PITCH_WIDTH  = 68.0    # y-axis: touchline to touchline

# Penalty box dimensions
BOX_LENGTH   = 16.5    # depth from goal line into pitch
BOX_WIDTH    = 40.32   # width of the box (centred on goal)
BOX_Y_START  = (PITCH_WIDTH - BOX_WIDTH) / 2   # = 13.84m
BOX_Y_END    = BOX_Y_START + BOX_WIDTH          # = 54.16m

# Penalty spots (distance from goal line)
PENALTY_SPOT_DIST = 11.0

# Pixels per metre when rendering the top-down template
SCALE = 10  # 1m = 10px → template is 1050x680px


def metres_to_pixels(x_m: float, y_m: float) -> tuple[int, int]:
    """Convert pitch coordinates in metres to template pixel coordinates."""
    return int(x_m * SCALE), int(y_m * SCALE)


def pixels_to_metres(px: int, py: int) -> tuple[float, float]:
    """Convert template pixel coordinates back to pitch metres."""
    return px / SCALE, py / SCALE


def is_outside_box(x_m: float, y_m: float, attacking_end: str = "right") -> bool:
    """
    Returns True if the coordinate is outside the penalty box at the attacking end.
    attacking_end: "right" means the goal being attacked is at x=105m.
                   "left"  means the goal being attacked is at x=0m.
    """
    in_y_range = BOX_Y_START <= y_m <= BOX_Y_END

    if attacking_end == "right":
        in_box = (x_m >= PITCH_LENGTH - BOX_LENGTH) and in_y_range
    else:
        in_box = (x_m <= BOX_LENGTH) and in_y_range

    return not in_box


def draw_pitch_template(
    highlight_point: tuple[float, float] | None = None
) -> np.ndarray:
    """
    Draw a top-down pitch template for visualisation.
    Optionally highlight a specific point (in metres).
    Returns a BGR image.
    """
    h = int(PITCH_WIDTH  * SCALE) + 40   # padding
    w = int(PITCH_LENGTH * SCALE) + 40
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (34, 139, 34)   # grass green

    pad = 20  # pixel padding so lines aren't at the edge

    def p(x_m, y_m):
        return (int(x_m * SCALE) + pad, int(y_m * SCALE) + pad)

    # --- pitch outline ---
    cv2.rectangle(img, p(0, 0), p(PITCH_LENGTH, PITCH_WIDTH), (255, 255, 255), 2)

    # --- centre line ---
    cv2.line(img, p(PITCH_LENGTH / 2, 0), p(PITCH_LENGTH / 2, PITCH_WIDTH), (255, 255, 255), 1)

    # --- centre circle ---
    centre = p(PITCH_LENGTH / 2, PITCH_WIDTH / 2)
    cv2.circle(img, centre, int(9.15 * SCALE), (255, 255, 255), 1)
    cv2.circle(img, centre, 3, (255, 255, 255), -1)

    # --- left penalty box ---
    cv2.rectangle(img, p(0, BOX_Y_START), p(BOX_LENGTH, BOX_Y_END), (255, 255, 255), 1)

    # --- right penalty box ---
    cv2.rectangle(img,
        p(PITCH_LENGTH - BOX_LENGTH, BOX_Y_START),
        p(PITCH_LENGTH, BOX_Y_END),
        (255, 255, 255), 1
    )

    # --- penalty spots ---
    cv2.circle(img, p(PENALTY_SPOT_DIST, PITCH_WIDTH / 2), 3, (255, 255, 255), -1)
    cv2.circle(img, p(PITCH_LENGTH - PENALTY_SPOT_DIST, PITCH_WIDTH / 2), 3, (255, 255, 255), -1)

    # --- highlight a specific point ---
    if highlight_point:
        x_m, y_m = highlight_point
        cv2.circle(img, p(x_m, y_m), 6, (0, 0, 255), -1)

    return img


# Named keypoints on the pitch — used to establish homography correspondences
# Each entry: name → (x_metres, y_metres)
KEYPOINTS = {
    "top_left_corner":              (0.0,                    0.0),
    "top_right_corner":             (PITCH_LENGTH,           0.0),
    "bottom_left_corner":           (0.0,                    PITCH_WIDTH),
    "bottom_right_corner":          (PITCH_LENGTH,           PITCH_WIDTH),
    "centre_spot":                  (PITCH_LENGTH / 2,       PITCH_WIDTH / 2),
    "left_box_top_left":            (0.0,                    BOX_Y_START),
    "left_box_top_right":           (BOX_LENGTH,             BOX_Y_START),
    "left_box_bottom_left":         (0.0,                    BOX_Y_END),
    "left_box_bottom_right":        (BOX_LENGTH,             BOX_Y_END),
    "right_box_top_left":           (PITCH_LENGTH - BOX_LENGTH, BOX_Y_START),
    "right_box_top_right":          (PITCH_LENGTH,           BOX_Y_START),
    "right_box_bottom_left":        (PITCH_LENGTH - BOX_LENGTH, BOX_Y_END),
    "right_box_bottom_right":       (PITCH_LENGTH,           BOX_Y_END),
    "left_penalty_spot":            (PENALTY_SPOT_DIST,      PITCH_WIDTH / 2),
    "right_penalty_spot":           (PITCH_LENGTH - PENALTY_SPOT_DIST, PITCH_WIDTH / 2),
}


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    template = draw_pitch_template(highlight_point=(94.0, 34.0))
    cv2.imwrite("outputs/pitch_template.jpg", template)
    print("Saved outputs/pitch_template.jpg")
    print(f"\nPitch: {PITCH_LENGTH}m x {PITCH_WIDTH}m")
    print(f"Box zone (right end): x >= {PITCH_LENGTH - BOX_LENGTH}m, y between {BOX_Y_START:.1f}m and {BOX_Y_END:.1f}m")
    print(f"\nIs (94m, 34m) outside box? {is_outside_box(94.0, 34.0, 'right')}")
    print(f"Is (80m, 34m) outside box? {is_outside_box(80.0, 34.0, 'right')}")
