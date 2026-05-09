"""
Debug script — saves torso crops to disk and prints every raw OCR result.
Run this to understand what EasyOCR is actually seeing before it filters.
"""
import cv2
import os
import easyocr
import numpy as np

from src.detection.detector import load_model, detect_objects, is_play_frame, filter_pitch_zone
from src.tracking.tracker import build_tracker, detections_to_sv, sv_to_tracks
from src.identity.jersey_ocr import crop_torso, preprocess_crop


FRAMES_DIR = "outputs/frames"
CROPS_DIR  = "outputs/debug_crops"
os.makedirs(CROPS_DIR, exist_ok=True)

reader = easyocr.Reader(["en"], gpu=False)
model  = load_model("yolov8n.pt")
tracker = build_tracker()

# use frames we already know have good player detections
sample = [
    "frame_000000.jpg",
    "frame_000030.jpg",
    "frame_000190.jpg",
    "frame_000355.jpg",
    "frame_000420.jpg",
    "frame_009510.jpg",
    "frame_014265.jpg",
]

crop_count = 0

for filename in sample:
    frame = cv2.imread(os.path.join(FRAMES_DIR, filename))
    detections = detect_objects(model, frame)

    if not is_play_frame(detections):
        continue

    detections = filter_pitch_zone(detections, frame.shape[0])
    sv_dets    = detections_to_sv(detections)
    sv_tracked = tracker.update_with_detections(sv_dets)
    tracks     = sv_to_tracks(sv_tracked, model.names)

    for track in tracks:
        if track.class_name != "person":
            continue

        # save the raw torso crop
        raw_crop = crop_torso(frame, track.x1, track.y1, track.x2, track.y2)
        if raw_crop.size == 0:
            continue

        # save the preprocessed version too
        processed = preprocess_crop(raw_crop)

        crop_name = f"{filename[:-4]}_id{track.track_id}"
        cv2.imwrite(os.path.join(CROPS_DIR, f"{crop_name}_raw.jpg"), raw_crop)
        cv2.imwrite(os.path.join(CROPS_DIR, f"{crop_name}_proc.jpg"), processed)

        # upscale the raw colour crop — try OCR on colour directly
        # EasyOCR was designed for natural images, grayscale may hurt it
        upscaled_color = cv2.resize(raw_crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        # run OCR on colour upscaled (no preprocessing)
        color_results = reader.readtext(upscaled_color, detail=0)
        color_digits  = reader.readtext(upscaled_color, allowlist="0123456789", detail=0)

        # also try on preprocessed grayscale for comparison
        processed      = preprocess_crop(raw_crop)
        gray_results   = reader.readtext(processed, detail=0)
        gray_digits    = reader.readtext(processed, allowlist="0123456789", detail=0)

        h, w = raw_crop.shape[:2]
        print(f"{filename} | track={track.track_id} | crop_size={w}x{h}")
        print(f"  colour OCR : {color_results}  digits: {color_digits}")
        print(f"  gray OCR   : {gray_results}   digits: {gray_digits}")

        crop_count += 1

print(f"\nSaved {crop_count} crops to {CROPS_DIR}/")
print("Open the crops directory to inspect what OCR is working with:")
print(f"  open {CROPS_DIR}")
