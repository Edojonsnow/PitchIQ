import cv2
import numpy as np
import easyocr
import re
from collections import Counter


# initialise once — loading the OCR model is expensive, don't do it per frame
_reader = None

def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def crop_torso(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Crop the torso region of a player bounding box.
    Jersey numbers sit in the upper body — we take the top 60% of the box,
    starting just below the head (top 15%).
    """
    height = y2 - y1
    torso_y1 = y1 + int(height * 0.15)
    torso_y2 = y1 + int(height * 0.75)
    return frame[torso_y1:torso_y2, x1:x2]


def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    upscaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def read_jersey_number(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str | None:
    """
    Attempt to read a jersey number from a player bounding box.
    Returns a 1-2 digit string e.g. "9" or None if nothing valid found.
    """
    crop = crop_torso(frame, x1, y1, x2, y2)
    if crop.size == 0:
        return None

    upscaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    reader = get_reader()
    results = reader.readtext(upscaled, allowlist="0123456789", detail=0)

    for text in results:
        text = text.strip()
        if re.fullmatch(r"\d{1,2}", text):
            return text
    return None


def detect_sponsor_text(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> list[str]:
    """
    Run OCR on the full player crop and return all text found.
    Used to detect sponsor text like 'ETIHAD' which identifies team membership.
    """
    crop = crop_torso(frame, x1, y1, x2, y2)
    if crop.size == 0:
        return []

    upscaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    reader = get_reader()
    results = reader.readtext(upscaled, detail=0)
    return [r.upper().strip() for r in results]


def is_mancity_player(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    """
    Detect if a player belongs to Man City using two signals.
    Runs cheap colour check first — only calls OCR if colour is inconclusive.
    1. Team colour  — sky blue (home) or yellow (away kit)  [cheap — pure NumPy]
    2. Sponsor text — 'ETIHAD' on jersey                    [expensive — OCR]
    """
    # --- Signal 1: team colour (cheap, run first) ---
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    sky_blue_mask = cv2.inRange(hsv,
        np.array([90, 80, 100]),
        np.array([110, 255, 255])
    )
    yellow_mask = cv2.inRange(hsv,
        np.array([25, 100, 100]),
        np.array([40, 255, 255])
    )

    total_pixels   = crop.shape[0] * crop.shape[1]
    sky_blue_ratio = np.sum(sky_blue_mask > 0) / total_pixels
    yellow_ratio   = np.sum(yellow_mask > 0) / total_pixels

    if sky_blue_ratio > 0.10 or yellow_ratio > 0.10:
        return True  # colour confirmed — skip OCR entirely

    # --- Signal 2: sponsor text OCR (expensive, only if colour inconclusive) ---
    texts = detect_sponsor_text(frame, x1, y1, x2, y2)
    return any("ETIHAD" in t for t in texts)


class PlayerIdentityMap:
    """
    Maps track_id → player name using three signals:
    1. Jersey number OCR  ("9" → Haaland)
    2. Sponsor text       ("ETIHAD" → Man City player)
    3. Team colour        (sky blue / yellow → Man City player)

    Haaland is confirmed when a Man City player is closest to the ball
    at a goal event, OR when jersey number "9" is directly read.
    """

    JERSEY_ROSTER = {
        "9": "Erling Haaland",
    }

    def __init__(self):
        self._jersey_readings: dict[int, list[str]] = {}
        self._mancity_votes: dict[int, int] = {}   # track_id → count of Man City signals
        self._resolved: dict[int, str] = {}

    def add_jersey_reading(self, track_id: int, jersey_number: str | None) -> None:
        if jersey_number is None:
            return
        self._jersey_readings.setdefault(track_id, []).append(jersey_number)

    def add_mancity_signal(self, track_id: int, is_mancity: bool) -> None:
        if is_mancity:
            self._mancity_votes[track_id] = self._mancity_votes.get(track_id, 0) + 1

    def is_mancity_track(self, track_id: int) -> bool:
        return self._mancity_votes.get(track_id, 0) >= 2

    def resolve(self, track_id: int) -> str | None:
        if track_id in self._resolved:
            return self._resolved[track_id]

        readings = self._jersey_readings.get(track_id, [])
        if len(readings) >= 2:
            most_common, _ = Counter(readings).most_common(1)[0]
            player_name = self.JERSEY_ROSTER.get(most_common)
            if player_name:
                self._resolved[track_id] = player_name
                return player_name

        return None

    def resolve_haaland_by_proximity(self, track_id: int) -> None:
        """
        Called when a Man City player is closest to the ball at a goal moment.
        Tags them as Haaland if no other identity is already resolved.
        """
        if track_id not in self._resolved and self.is_mancity_track(track_id):
            self._resolved[track_id] = "Erling Haaland"

    def reset_for_scene(self) -> None:
        self._jersey_readings.clear()
        self._mancity_votes.clear()

    def get_all_resolved(self) -> dict[int, str]:
        return dict(self._resolved)
