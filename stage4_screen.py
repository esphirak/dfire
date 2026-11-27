"""
Stage 4 (screen version): watch your SCREEN for fire and smoke, and draw the boxes
and the alert in a see-through layer on top of whatever you are looking at
(a YouTube video, a photo, a CCTV viewer...). Windows only.

You can keep using your computer normally: mouse clicks go straight through the layer.

Run from the project folder, with .venv-yolo active:
    pip install mss                  (one time only)
    python stage4_screen.py
    python stage4_screen.py --monitor 2      # a second screen

Stop: press Ctrl+Shift+Q (works from any window), or Ctrl+C in the terminal.
"""
import argparse
import ctypes
import time
import tkinter as tk
from datetime import datetime

import cv2
import mss
import numpy as np
from ultralytics import YOLO

from stage4_alert import CONF, MODEL_PATH, SNAPSHOT_DIR, AlertTracker

KEY_COLOUR = '#010101'      # this colour becomes see-through; near-black so text edges look clean
BOX_COLOURS = {'fire': '#ff3b30', 'smoke': '#3b82f6'}
user32 = ctypes.windll.user32


def make_dpi_aware():
    """Use real screen pixels, so the boxes line up with the screen on scaled (125%, 150%) displays."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        user32.SetProcessDPIAware()


def make_click_through(root):
    """Let mouse clicks pass through the layer, hide it from the taskbar, and hide it from screenshots."""
    root.update()
    hwnd = user32.GetParent(root.winfo_id())
    GWL_EXSTYLE, WS_EX_LAYERED, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW = -20, 0x80000, 0x20, 0x80
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW)
    # Keep the layer out of our own screenshots, or the model would see its own boxes.
    WDA_EXCLUDEFROMCAPTURE = 0x11
    return bool(user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))


def stop_keys_pressed():
    """True while Ctrl+Shift+Q is held, whichever window is active."""
    down = lambda vk: user32.GetAsyncKeyState(vk) & 0x8000
    return down(0x11) and down(0x10) and down(ord('Q'))


def draw_box(canvas, x1, y1, x2, y2, label, colour):
    canvas.create_rectangle(x1, y1, x2, y2, outline=colour, width=4)
    ty = y1 - 26 if y1 >= 26 else y1          # label above the box, or inside if at the top edge
    canvas.create_rectangle(x1, ty, x1 + 12 + 10 * len(label), ty + 26, fill=colour, outline=colour)
    canvas.create_text(x1 + 6, ty + 13, text=label, fill='white', anchor='w', font=('Segoe UI', 12, 'bold'))


def draw_status(canvas, width, alert_text, fps):
    if alert_text:
        canvas.create_rectangle(width // 2 - 330, 10, width // 2 + 330, 70, fill='#dc0000', outline='#dc0000')
        canvas.create_text(width // 2, 40, text=alert_text, fill='white', font=('Segoe UI', 22, 'bold'))
    canvas.create_rectangle(10, 10, 300, 40, fill='#333333', outline='#333333')
    canvas.create_text(20, 25, anchor='w', fill='white', font=('Segoe UI', 11),
                       text=f'Fire watch ON  {fps:.1f} FPS  (Ctrl+Shift+Q stops)')


def main():
    parser = argparse.ArgumentParser(description='Fire and smoke alert drawn over your screen')
    parser.add_argument('--monitor', type=int, default=1, help='1 = main screen, 2 = second screen')
    args = parser.parse_args()

    make_dpi_aware()
    model = YOLO(MODEL_PATH)
    sct = mss.mss()
    if args.monitor >= len(sct.monitors):
        raise SystemExit(f'There is no screen number {args.monitor}. Screens found: {len(sct.monitors) - 1}')
    mon = sct.monitors[args.monitor]

    root = tk.Tk()
    root.overrideredirect(True)                       # no title bar or border
    root.geometry(f"{mon['width']}x{mon['height']}+{mon['left']}+{mon['top']}")
    root.attributes('-topmost', True)
    root.attributes('-transparentcolor', KEY_COLOUR)
    root.configure(bg=KEY_COLOUR)
    canvas = tk.Canvas(root, bg=KEY_COLOUR, highlightthickness=0)
    canvas.pack(fill='both', expand=True)
    if not make_click_through(root):
        print('Note: this Windows version cannot hide the layer from screenshots, so the model may see its own boxes.')

    SNAPSHOT_DIR.mkdir(exist_ok=True)
    tracker = AlertTracker()
    last_labels = set()
    last_time = time.time()
    print('Watching the screen. Press Ctrl+Shift+Q to stop.')

    try:
        while not stop_keys_pressed():
            frame = np.ascontiguousarray(np.array(sct.grab(mon))[:, :, :3])   # screenshot, BGRA -> BGR
            result = model(frame, conf=CONF, verbose=False)[0]

            canvas.delete('all')
            found = []
            for (x1, y1, x2, y2), c, p in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                              result.boxes.conf.tolist()):
                name = model.names[int(c)]
                found.append(name)
                draw_box(canvas, x1, y1, x2, y2, f'{name} {p:.2f}', BOX_COLOURS.get(name, '#ffcc00'))
            if found:
                last_labels = set(found)

            if tracker.update(len(found) > 0):
                stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                cv2.imwrite(str(SNAPSHOT_DIR / f'screen_alert_{stamp}.jpg'), result.plot())
                print(f'ALERT at {stamp}: {", ".join(sorted(last_labels))}')

            now = time.time()
            fps = 1 / max(now - last_time, 1e-6)
            last_time = now
            alert_text = ('ALERT: ' + ' + '.join(sorted(last_labels)).upper() + ' DETECTED') if tracker.active else ''
            draw_status(canvas, mon['width'], alert_text, fps)
            root.update()
    except KeyboardInterrupt:
        pass
    finally:
        root.destroy()
        print('Stopped.')


if __name__ == '__main__':
    main()
