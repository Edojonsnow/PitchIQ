import cv2
import os

from src.detection.detector import load_model, detect_objects, is_play_frame, filter_pitch_zone
from src.tracking.tracker import build_tracker, detections_to_sv, sv_to_tracks, draw_tracks
from src.identity.scene_detector import SceneTracker
from src.identity.jersey_ocr import PlayerIdentityMap, read_jersey_number, is_mancity_player


def run_identity_pipeline(frames_dir: str, output_dir: str, max_frames: int = 200) -> None:
    os.makedirs(output_dir, exist_ok=True)

    model        = load_model("yolov8n.pt")
    tracker      = build_tracker()
    scene_tracker = SceneTracker(threshold=50.0, min_scene_frames=15)
    identity_map  = PlayerIdentityMap()

    all_frames = sorted(os.listdir(frames_dir))[:max_frames]

    for filename in all_frames:
        frame_path = os.path.join(frames_dir, filename)
        frame = cv2.imread(frame_path)

        # --- scene cut detection ---
        is_new_scene, scene_idx = scene_tracker.update(frame)
        if is_new_scene:
            tracker = build_tracker()
            identity_map.reset_for_scene()
            print(f"\n--- Scene {scene_idx} | {filename} ---")

        # --- detection + filtering ---
        detections = detect_objects(model, frame)
        if not is_play_frame(detections):
            continue
        detections = filter_pitch_zone(detections, frame.shape[0])

        # --- tracking ---
        sv_dets    = detections_to_sv(detections)
        sv_tracked = tracker.update_with_detections(sv_dets)
        tracks     = sv_to_tracks(sv_tracked, model.names)

        # --- identity signals on each player track ---
        for track in tracks:
            if track.class_name != "person":
                continue

            already_resolved  = track.track_id in identity_map.get_all_resolved()
            already_mancity   = identity_map.is_mancity_track(track.track_id)

            # skip OCR entirely for tracks we've already fully identified
            if already_resolved:
                continue

            # if already confirmed Man City, only run cheap colour check — skip OCR
            if already_mancity:
                mancity = is_mancity_player(frame, track.x1, track.y1, track.x2, track.y2)
                identity_map.add_mancity_signal(track.track_id, mancity)
            else:
                # new track — run full OCR + colour check to establish identity
                jersey  = read_jersey_number(frame, track.x1, track.y1, track.x2, track.y2)
                identity_map.add_jersey_reading(track.track_id, jersey)

                mancity = is_mancity_player(frame, track.x1, track.y1, track.x2, track.y2)
                identity_map.add_mancity_signal(track.track_id, mancity)

                if mancity:
                    print(f"  {filename} | track_id={track.track_id} → Man City player detected")

            player_name = identity_map.resolve(track.track_id)
            if player_name:
                print(f"  {filename} | track_id={track.track_id} → ✓ {player_name}")

        # --- annotate frame ---
        annotated = draw_tracks(frame, tracks)
        resolved  = identity_map.get_all_resolved()

        for track in tracks:
            name = resolved.get(track.track_id)
            if name:
                cv2.putText(
                    annotated, name,
                    (track.x1, track.y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
                )
            elif identity_map.is_mancity_track(track.track_id):
                cv2.putText(
                    annotated, "Man City",
                    (track.x1, track.y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1
                )

        cv2.imwrite(os.path.join(output_dir, filename), annotated)

    print("\n=== Final resolved identities ===")
    for track_id, name in identity_map.get_all_resolved().items():
        print(f"  track_id={track_id} → {name}")

    print("\n=== Man City players detected ===")
    for track_id, votes in identity_map._mancity_votes.items():
        if votes >= 2:
            print(f"  track_id={track_id} → {votes} Man City signals")


if __name__ == "__main__":
    run_identity_pipeline(
        frames_dir="outputs/frames",
        output_dir="outputs/annotated",
        max_frames=100,
    )
