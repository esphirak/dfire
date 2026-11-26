"""
Make a test video for stage4_alert.py from real D-Fire test photos.

The video cycles through none -> smoke -> fire -> both, showing one photo for a few
seconds each, with the TRUE answer written under the photo. The photos come from
DFire/test, which the model never trained on, and are picked at random.

Run from the project folder, with .venv-yolo active:
    python make_test_video.py
    python make_test_video.py --seconds 60 --per-photo 5 --seed 7
Then test the alert on it:
    python stage4_alert.py --source test_video.mp4
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

TEST_DIR = Path('DFire/test')
ORDER = ['none', 'smoke', 'fire', 'both']
WIDTH, HEIGHT = 1280, 720
TOP_SPACE = 60       # kept empty so stage4_alert.py's banner does not cover the photo
CAPTION_SPACE = 50   # bar at the bottom with the true answer
FPS = 15


def kind_of(label_file):
    """none / smoke / fire / both, from a D-Fire label file (0 = smoke, 1 = fire)."""
    ids = set()
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            if line.split():
                ids.add(int(float(line.split()[0])))
    return {frozenset(): 'none', frozenset({0}): 'smoke',
            frozenset({1}): 'fire', frozenset({0, 1}): 'both'}.get(frozenset(ids), 'other')


def make_frame(photo_path, true_kind):
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img = cv2.imread(str(photo_path))
    box_h = HEIGHT - TOP_SPACE - CAPTION_SPACE
    scale = min(WIDTH / img.shape[1], box_h / img.shape[0])
    w, h = int(img.shape[1] * scale), int(img.shape[0] * scale)
    x, y = (WIDTH - w) // 2, TOP_SPACE + (box_h - h) // 2
    canvas[y:y + h, x:x + w] = cv2.resize(img, (w, h))
    text = f'TRUE ANSWER: {true_kind.upper()}   ({photo_path.name})'
    cv2.putText(canvas, text, (20, HEIGHT - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def main():
    parser = argparse.ArgumentParser(description='Build a none/smoke/fire/both test video from D-Fire test photos')
    parser.add_argument('--seconds', type=int, default=60, help='length of the video')
    parser.add_argument('--per-photo', type=int, default=5, help='seconds each photo stays on screen')
    parser.add_argument('--seed', type=int, default=None, help='same number = same photos again')
    parser.add_argument('--out', default='test_video.mp4')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    groups = {k: [] for k in ORDER}
    for img in sorted((TEST_DIR / 'images').iterdir()):
        if img.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            k = kind_of(TEST_DIR / 'labels' / (img.stem + '.txt'))
            if k in groups:
                groups[k].append(img)
    for k in ORDER:
        if not groups[k]:
            raise SystemExit(f'No "{k}" photos found in {TEST_DIR}. Is DFire/test in this folder?')
        rng.shuffle(groups[k])

    n_photos = max(1, args.seconds // args.per_photo)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (WIDTH, HEIGHT))
    print(f'Making {args.out}: {n_photos} photos x {args.per_photo} s')
    for i in range(n_photos):
        k = ORDER[i % len(ORDER)]
        photo = groups[k][i // len(ORDER) % len(groups[k])]
        frame = make_frame(photo, k)
        for _ in range(FPS * args.per_photo):
            writer.write(frame)
        print(f'{i * args.per_photo:>3}-{(i + 1) * args.per_photo:>3} s  {k:<6} {photo.name}')
    writer.release()
    print(f'Done: {args.out}')


if __name__ == '__main__':
    main()
