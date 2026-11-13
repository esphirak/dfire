# VERSION 4
"""
Stage 1 helper: turn D-Fire (YOLO box labels) into an image-level classification dataset.

Each image becomes one of four classes, decided from its label .txt file:
    none  = empty/missing label file
    smoke = only smoke boxes
    fire  = only fire boxes
    both  = fire and smoke boxes

Two layouts are supported (it detects which one you have):
  A) DFire/ already contains train/ and test/ folders, each with images/ and labels/.
     -> used directly. You do NOT need DFire_splits/.
  B) DFire/ is a flat images + labels download. Then it uses the official lists
     dfire_train.txt and dfire_test.txt, which must be inside DFire_splits/.

Validation images are taken from the training images (15%).

Run from your project folder:   python build_fire_dataset.py
"""
import random
import shutil
import sys
from pathlib import Path

# ---------------- settings you may change ----------------
SRC = Path('DFire')            # the D-Fire download (unzipped)
SPLITS = Path('DFire_splits')  # only used for layout B
OUT = Path('Dataset_fire')     # output folder (created by this script)

SMOKE_ID = 0                   # class id of smoke in the label files
FIRE_ID = 1                    # class id of fire in the label files
# ^ I took this mapping from a third-party repo, NOT from the D-Fire authors.
#   The notebook shows sample images per class. If "fire" shows smoke (or the
#   reverse), swap these two numbers, delete Dataset_fire/, and run again.

CAP_TRAIN = 1800     # max images per class from the training images, before validation is carved out (None = all)
CAP_TEST = 400       # max images per class from the test images (None = all)
VAL_FRACTION = 0.15  # share of the training images used for validation
SEED = 42
# ----------------------------------------------------------

CLASSES = ['none', 'smoke', 'fire', 'both']
IMG_EXT = {'.jpg', '.jpeg', '.png'}
LABEL_NAMES = {'labels', 'det_labels'}
random.seed(SEED)


def categorize(label_file):
    if label_file is None or not label_file.exists():
        return 'none'
    ids = set()
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if parts:
            ids.add(int(float(parts[0])))
    has_fire, has_smoke = FIRE_ID in ids, SMOKE_ID in ids
    if has_fire and has_smoke:
        return 'both'
    if has_fire:
        return 'fire'
    if has_smoke:
        return 'smoke'
    return 'none'


def all_dirs(root):
    return [root] + sorted(p for p in root.rglob('*') if p.is_dir())


def find_split_root(src):
    """Layout A: a folder that has train/images and test/images."""
    for p in all_dirs(src):
        if (p / 'train' / 'images').is_dir() and (p / 'test' / 'images').is_dir():
            return p
    return None


def labels_folder(split_dir):
    for name in LABEL_NAMES:
        if (split_dir / name).is_dir():
            return split_dir / name
    return None


def groups_from_folders(root):
    """Layout A."""
    groups = {}
    for split in ('train', 'test'):
        lab = labels_folder(root / split)
        if lab is None:
            sys.exit(f'No labels folder found inside {root / split}. '
                     f'Contents: {[p.name for p in (root / split).iterdir()]}')
        groups[split] = {c: [] for c in CLASSES}
        for img in sorted((root / split / 'images').rglob('*')):
            if img.suffix.lower() in IMG_EXT:
                groups[split][categorize(lab / (img.stem + '.txt'))].append(img)
    return groups


def groups_from_lists():
    """Layout B."""
    if not SPLITS.exists():
        sys.exit(f'Your {SRC} folder has no train/ and test/ folders, so I need the official lists, '
                 f'but the folder "{SPLITS}" does not exist.')
    img_dirs = [d for d in all_dirs(SRC) if d.name.lower() == 'images'] or [SRC]
    lab_dirs = [d for d in all_dirs(SRC) if d.name.lower() in LABEL_NAMES] or [SRC]
    img_by_stem = {p.stem: p for d in img_dirs for p in d.rglob('*') if p.suffix.lower() in IMG_EXT}
    lab_by_stem = {p.stem: p for d in lab_dirs for p in d.rglob('*.txt')}
    print(f'Found {len(img_by_stem)} images and {len(lab_by_stem)} label files.')
    groups = {}
    for split, filename in (('train', 'dfire_train.txt'), ('test', 'dfire_test.txt')):
        hits = [SPLITS / filename] if (SPLITS / filename).exists() else sorted(SPLITS.rglob(filename))
        if not hits:
            sys.exit(f'{filename} not found inside {SPLITS}.')
        names = [l.strip() for l in hits[0].read_text().splitlines() if l.strip()]
        groups[split] = {c: [] for c in CLASSES}
        missing = 0
        for name in names:
            stem = Path(name.replace('\\', '/')).stem
            img = img_by_stem.get(stem)
            if img is None:
                missing += 1
                continue
            groups[split][categorize(lab_by_stem.get(stem))].append(img)
        print(f'{split}: {filename} has {len(names)} entries, {missing} not found in the images')
        if names and missing == len(names):
            sys.exit(f'None of the entries in {filename} matched an image. First line: "{names[0]}"\n'
                     f'Example image I found: "{next(iter(img_by_stem.values())).name}"\n'
                     'Tell me both and I will fix the matching.')
    return groups


def take(items, n):
    items = items[:]
    random.shuffle(items)
    return items if n is None else items[:n]


def copy_all(items, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for f in items:
        shutil.copy(f, dest / f.name)


# ----------------------------- main -----------------------------
if OUT.exists():
    sys.exit(f'{OUT} already exists. Delete that folder first, then run the script again.')
if not SRC.exists():
    sys.exit(f'Folder "{SRC}" not found in {Path.cwd()}.')

root = find_split_root(SRC)
if root is not None:
    print('Layout A: using the train/ and test/ folders in', root)
    groups = groups_from_folders(root)
else:
    print('Layout B: no train/ and test/ folders found, using the official lists instead.')
    groups = groups_from_lists()

print('\nAvailable (before limiting):')
print(f"{'class':<8}{'train':>8}{'test':>8}")
for c in CLASSES:
    print(f"{c:<8}{len(groups['train'][c]):>8}{len(groups['test'][c]):>8}")

for c in CLASSES:
    tr = take(groups['train'][c], CAP_TRAIN)
    n_val = int(len(tr) * VAL_FRACTION)
    copy_all(tr[n_val:], OUT / 'train' / c)
    copy_all(tr[:n_val], OUT / 'validation' / c)
    copy_all(take(groups['test'][c], CAP_TEST), OUT / 'test' / c)

print('\nCreated in', OUT)
print(f"{'class':<8}{'train':>8}{'val':>8}{'test':>8}")
for c in CLASSES:
    n = [len(list((OUT / s / c).iterdir())) for s in ('train', 'validation', 'test')]
    print(f'{c:<8}{n[0]:>8}{n[1]:>8}{n[2]:>8}')
