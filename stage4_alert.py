"""
Stage 4: real-time fire and smoke alert.

Watches a webcam or a video file, draws the YOLO boxes live, and shows a red ALERT
only when fire or smoke is seen in most of the last few frames, so one odd frame
does not set it off. Each time an alert starts, a snapshot is saved in alerts/.

Run from the project folder, with .venv-yolo active:
    python stage4_alert.py                     # webcam
    python stage4_alert.py --source my_video.mp4
    python stage4_alert.py --source my_video.mp4 --display-height 900   # bigger window
Press Q in the video window to stop. The window can also be resized with the mouse.
"""
import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_PATH = 'models/yolo_stage2/best.pt'
CONF = 0.34          # confidence threshold; 0.34 gave the best F1 on the test photos
WINDOW = 8           # look at the last 8 frames ...
NEEDED = 5           # ... and raise the alert if at least 5 of them had fire or smoke
SNAPSHOT_DIR = Path('alerts')
DISPLAY_HEIGHT = 720 # the window is shrunk to at most this tall (and 1280 wide) so tall videos fit the screen
WINDOW_NAME = 'Fire and smoke alert (press Q to quit)'


class AlertTracker:
    """Remembers which recent frames had a detection and decides when to alert."""

    def __init__(self, window=WINDOW, needed=NEEDED):
        self.recent = deque(maxlen=window)   # True/False for each recent frame
        self.needed = needed
        self.active = False

    def update(self, detected):
        """Add one frame. Returns True only on the frame where a new alert starts."""
        self.recent.append(bool(detected))
        was_active = self.active
        self.active = sum(self.recent) >= self.needed
        return self.active and not was_active


def fit_for_screen(frame, max_height, max_width=1280):
    """Shrink a frame (keeping its shape) so it fits on screen. Only the display is shrunk, not what the model sees."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_height / h, max_width / w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def draw_banner(frame, text, colour):
    font = min(1.0, frame.shape[1] / 700)      # smaller text on narrow (vertical) videos
    height = int(20 + 30 * font)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], height), colour, -1)
    cv2.putText(frame, text, (10, int(height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description='Real-time fire and smoke alert')
    parser.add_argument('--source', default='0', help='0 for the webcam, or the path of a video file')
    parser.add_argument('--display-height', type=int, default=DISPLAY_HEIGHT, help='largest window height in pixels')
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(MODEL_PATH)
    video = cv2.VideoCapture(source)
    if not video.isOpened():
        raise SystemExit(f'Could not open video source: {args.source}')

    SNAPSHOT_DIR.mkdir(exist_ok=True)
    tracker = AlertTracker()
    last_time = time.time()
    last_labels = set()   # what was seen most recently, kept for the banner
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)   # resizable, keeps the shape

    while True:
        ok, frame = video.read()
        if not ok:
            print('Video finished.')
            break

        result = model(frame, conf=CONF, verbose=False)[0]
        found = [model.names[int(c)] for c in result.boxes.cls]   # e.g. ['fire', 'smoke']
        shown = result.plot()                                        # frame with boxes drawn
        if found:
            last_labels = set(found)

        if tracker.update(len(found) > 0):
            stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            cv2.imwrite(str(SNAPSHOT_DIR / f'alert_{stamp}.jpg'), shown)   # full size, before shrinking
            print(f'ALERT at {stamp}: {", ".join(sorted(last_labels))}')

        shown = fit_for_screen(shown, args.display_height)

        if tracker.active:
            draw_banner(shown, 'ALERT: ' + ' + '.join(sorted(last_labels)).upper() + ' DETECTED', (0, 0, 220))
        else:
            draw_banner(shown, 'Monitoring...', (60, 60, 60))

        now = time.time()
        fps = 1 / max(now - last_time, 1e-6)
        last_time = now
        cv2.putText(shown, f'{fps:.1f} FPS', (10, shown.shape[0] - 15),       # bottom-left corner
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, shown)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
