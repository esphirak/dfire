# Fire and Smoke Detection project: handoff notes

These notes carry over an earlier conversation with Claude so a new Claude Code
session in VS Code can continue where it left off.

## How to work with me
- I'm a student. Go **one small step at a time**, in plain language. Too much at once overwhelms me.
- Before giving a new step, wait for me to report the result of the last one.
- I work in VS Code on Windows (PowerShell), Lenovo ThinkPad X1 Carbon, Intel Core Ultra 7 256V, no NVIDIA GPU (CPU only).
- Remind me which stage we're in when we move on.

## Project (from my proposal)
Fire and Smoke Detection Using Computer Vision.
Objective: detect early fire hazards from images/videos. Techniques: YOLO, CNN, segmentation.
Dataset: D-Fire (FireNet optional). Extension: real-time emergency alert system.

## Stages
| Stage | What | Status |
|---|---|---|
| 1 | CNN classifier: whole photo is none / smoke / fire / both | **Done** |
| 2 | YOLO detection: boxes around fire and smoke | **In progress** |
| 3 | Segmentation (optional) | Later |
| 4 | Real-time alert from webcam/video (OpenCV + YOLO, alert after several frames in a row) | Later |

## Project folder: `D:\Y3 S1\Project Practicum\dfire`
- `DFire/` original D-Fire download, already split: `train/images`, `train/labels`, `test/images`, `test/labels`. Do not modify.
- `DFire_splits/` official lists `dfire_train.txt`, `dfire_test.txt` (may be missing; optional).
- `Dataset_fire/` built by `build_fire_dataset.py` (version 5) for Stage 1: `train` / `validation` / `test`, each with `none`, `smoke`, `fire`, `both`.
  Counts: train 1530/1530/803/1530, val 270/270/141/270, test 400/400/220/400 (none/smoke/fire/both).
- `Fire_Stage1_CNN.ipynb` teacher's CNN adapted to 4 classes (200x200, 10 epochs, class weights). Saves `models/fire_cnn_stage1.h5`.
- `Fire_Stage1_Improved.ipynb` MobileNetV2 transfer learning (224x224, augmentation, early stopping). Saves `models/fire_mobilenet_stageB.keras`.
- `dfire.yaml` YOLO data config (see Stage 2).

## Key facts
- D-Fire label files are YOLO format: `class cx cy w h`, normalized 0-1. Empty file = no fire/smoke.
- **Class IDs: 0 = smoke, 1 = fire.** Confirmed by matching published counts (1,164 fire-only, 5,867 smoke-only, 4,658 both) and by checking label files against images.
- Validation for Stage 1 was a random 15% of the training images (D-Fire has no official validation list).
- The "Test accuracy" printed right after training in the baseline notebook is measured on training data. Always report the "Real test accuracy".

## Stage 1 results (1,420 test images)
| Model | Real test accuracy |
|---|---|
| Baseline CNN (teacher's) | 70.14% (overfits: train 97.7%, val ~70%, val loss lowest at epoch 2) |
| MobileNetV2 fine-tuned | 78.87% (much less overfitting) |

Baseline weakest: smoke recall 0.59; often guesses "both". MobileNetV2: smoke recall 0.74, but more smoke photos called "none" (83 vs 49).
Remaining errors: tiny flames at night, faint smoke, ambiguous hazy scenes, possible label noise.
All numbers are also saved in a claude.ai Sheet called "Fire and Smoke Stage 1 Results".

**Project log doc:** "Fire and Smoke Detection Project Log" (https://claude.ai/artifact/FceifSJdXeN6pX3cwn3L8h).
Update it every time a stage finishes: the stage table's status, the stage's own section, and the presentation outline.

## Environments
- TensorFlow work (Stage 1): the environment used for the notebooks (the one with TensorFlow installed).
- YOLO work (Stage 2+): `.venv-yolo` in the project folder, Python 3.14.7, torch 2.15.0+cpu, ultralytics 8.4.211.
  Activate with `.venv-yolo\Scripts\activate`.
- `yolo predict model=yolo26n.pt source=https://ultralytics.com/images/bus.jpg` worked: 4 persons + 1 bus, ~115 ms per image on CPU.

## Where we stopped (Stage 2, step 2-3)
1. Create `dfire.yaml` in the project folder (absolute path needed, since Ultralytics resolves relative paths against its own datasets folder):
   ```yaml
   path: D:/Y3 S1/Project Practicum/dfire/DFire
   train: train/images
   val: test/images
   names:
     0: smoke
     1: fire
   ```
   Note: this uses the test images for validation, so the final test score isn't fully independent. Tighten this before the real training run (e.g. carve a validation set from train).
2. Tiny test run to check everything connects and time one epoch:
   ```
   yolo detect train data=dfire.yaml model=yolo26n.pt epochs=1 imgsz=320 fraction=0.05 batch=16
   ```
   **Done 2026-11-13** (`runs/detect/tiny_test`): pipeline works. 861 train images took ~1 min (~1.1 s per batch of 16), validating 4,302 test images ~1.5 min.
   Scores ~0 as expected (the 5% slice was the first files alphabetically, 859 of 861 empty). Estimate: full epoch ~20 min at imgsz=320, ~80 min at 640 on this CPU.
   YOLO flagged 4 test labels as out-of-bounds (ignored) and re-saved 12 slightly corrupt test JPEGs in place (WEB11243-WEB11323).
3. Decided: full training is too slow on this CPU, so it runs on **Kaggle (free GPU)**. Kaggle already hosts D-Fire, so no upload.
4. **Next: `Stage2_YOLO_Kaggle.ipynb`** (in this repo). Run it on Kaggle: add the D-Fire dataset as input, GPU on, Internet on.
   First with `QUICK = True` (~5 min check), then `QUICK = False` via Save Version → Save & Run All.
   It checks 0 = smoke / 1 = fire against the published counts, keeps 15% of train aside for validation in blocks of 50
   consecutive file names (neighbouring video frames are near-copies), trains yolo26n at 640, tests on the official test set,
   and collects `best.pt` + plots in `/kaggle/working/to_download`. Download `best.pt` to the laptop project folder (git-ignored).
5. **Kaggle QUICK run done 2026-11-21.** Kaggle copy = sayedgamal99/smoke-fire-detection-yolo (GAIA's D-Fire):
   train 14,122 / val 3,099 / test 4,306 (its own val folder is used). Label check matched exactly
   (both 4,658, none 9,838, smoke-only 5,867, fire-only 1,164). 1 epoch on 1,500 images: 41 s on a Tesla T4, mAP50 0.03 (expected for 1 epoch).
   Found: the Kaggle input drive is slow (~8 MB/s) and read-only, so epochs wait on photo reads, and YOLO skipped
   ~0.2% slightly corrupt JPEGs it couldn't repair in place. Added cell 3b: in the full run the dataset is copied
   to `/tmp/dfire_local` first (~10 min), which fixes both.
6. **Full Kaggle run done 2026-11-22** (QUICK = False, with cell 3b). Test results on the 4,306 official test photos:
   | Measure | Smoke | Fire | Overall |
   |---|---|---|---|
   | Precision | 0.841 | 0.726 | 0.783 |
   | Recall | 0.797 | 0.664 | 0.731 |
   | mAP50 | 0.843 | 0.715 | 0.779 |
   | mAP50-95 | 0.520 | 0.376 | 0.448 |
   Fire is harder than smoke (fire boxes are often tiny, e.g. small flames at night). Number of epochs trained: not yet reported.
7. **Next:** download `best.pt` + plots from the version's Output tab (`to_download/`) into the laptop project folder,
   then try the model on a few photos/videos on the laptop. That finishes Stage 2; then Stage 4.
