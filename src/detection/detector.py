import cv2
from ultralytics import YOLO
from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


def load_model(weights: str = "yolov8n.pt") -> YOLO:
    # ultralytics auto-downloads the weights on first run if not found locally
    return YOLO(weights)


def detect_objects(model: YOLO, frame: np.ndarray, confidence: float = 0.3) -> list[Detection]:
    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < confidence:
            continue

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append(Detection(
            class_id=class_id,
            class_name=class_name,
            confidence=conf,
            x1=x1, y1=y1, x2=x2, y2=y2,
        ))

    return detections


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = frame.copy()

    for det in detections:
        color = _class_color(det.class_name)
        cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), color, 2)

        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(annotated, label, (det.x1, det.y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return annotated


def _class_color(class_name: str) -> tuple[int, int, int]:
    colors = {
        "person": (0, 255, 0),
        "sports ball": (0, 0, 255),
    }
    # default color for anything else
    return colors.get(class_name, (255, 255, 0))


if __name__ == "__main__":
    import os

    FRAMES_DIR = "outputs/frames"
    OUTPUT_DIR = "outputs/annotated"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model("yolov8n.pt")

    # run detection on 10 sample frames spread across the video
    all_frames = sorted(os.listdir(FRAMES_DIR))
    sample = all_frames[::len(all_frames) // 10][:10]

    for filename in sample:
        frame_path = os.path.join(FRAMES_DIR, filename)
        frame = cv2.imread(frame_path)

        detections = detect_objects(model, frame)
        annotated = draw_detections(frame, detections)

        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, annotated)

        persons = [d for d in detections if d.class_name == "person"]
        balls = [d for d in detections if d.class_name == "sports ball"]
        print(f"{filename} → {len(persons)} persons, {len(balls)} balls")
