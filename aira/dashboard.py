"""Step 10 - local live dashboard + alert sound (OpenCV window, no network).

Designed to be understood at a glance by someone who has never seen it:
  * every card says in plain words what is happening and what happens next;
  * a 3-step tracker shows where the system is: Find -> Learn -> Watch;
  * the "time since last breath" bar shows exactly when PAUSE / PRE-ALERT / ALARM happen;
  * the event list is written in plain English.

Controls:
    click a card : select it and ACKNOWLEDGE its alarm
    1-9          : select a card
    c            : full camera view (to adjust the camera)
    r            : re-learn normal breathing for the selected card
    v            : mark selected cot empty / occupied again
    f            : full screen on/off
    Stop button  : close this page
"""
import re
import sys
import threading
import time

import cv2
import numpy as np

from . import theme as T
from .monitor import TIER_ALARM, TIER_PAUSE, TIER_PREALERT

BG = T.bgr("bg")
_t = np.linspace(0, 2 * np.pi, 60)
HEART_SHAPE = np.stack([16 * np.sin(_t) ** 3, -(13 * np.cos(_t) - 5 * np.cos(2 * _t) - 2 * np.cos(3 * _t) - np.cos(4 * _t))], 1) / 17.0
HEADER_H, LOG_H, FOOT_H, GAP = 86, 150, 44, 26
BAR_MAX_S = 25.0
LEARNING = ("CALIBRATING", "STARTING", "WEAK SIGNAL")


def _screen_size():
    try:
        import ctypes
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def _work_area():
    """Usable screen area (without the taskbar): x, y, w, h."""
    try:
        import ctypes
        from ctypes import wintypes
        r = wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(r), 0)
        return r.left, r.top, r.right - r.left, r.bottom - r.top
    except Exception:
        w, h = _screen_size()
        return 0, 0, w, h


def _fit_window(title, W=None, H=None):
    """Open the page centred on the screen at a comfortable size (it can be resized and
    maximised; the layout re-arranges itself to use whatever space the window has)."""
    ax, ay, aw, ah = _work_area()
    w, h = int(aw * 0.82), int(ah * 0.86)
    cv2.resizeWindow(title, w, h)
    cv2.moveWindow(title, ax + (aw - w) // 2, ay + max(0, (ah - h) // 2 - 20))


def _ecg(phase):
    """Stylised one-beat heartbeat shape (P wave, QRS spike, T wave) for phase in [0, 1)."""
    g = lambda c, w, a: a * np.exp(-((phase - c) / w) ** 2)
    return g(0.18, 0.035, 0.12) - g(0.30, 0.012, 0.15) + g(0.33, 0.014, 1.0) - g(0.36, 0.012, 0.3) + g(0.58, 0.06, 0.28)


def _wrap(P, text, size, width):
    words, line, out = text.split(), "", []
    for w in words:
        if line and P.width(line + " " + w, size) > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


def _paste(canvas, arr_rgb, x, y):
    h, w = arr_rgb.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = arr_rgb[y0 - y:y1 - y, x0 - x:x1 - x, ::-1]


def _raised(canvas, x, y, w, h, r=24, face=None, d=7, blur=9):
    """Soft raised panel. Blended as a difference from the background, so neighbouring
    panels' shadows add up instead of one panel's margin cutting off the other's shadow."""
    a, m = T.raised(int(w), int(h), r, d, blur, face=face)
    x, y = int(x) - m, int(y) - m
    hh, ww = a.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + ww), min(H, y + hh)
    if x1 <= x0 or y1 <= y0:
        return
    delta = a[y0 - y:y1 - y, x0 - x:x1 - x, ::-1].astype(np.int16) - np.array(BG, np.int16)
    canvas[y0:y1, x0:x1] = np.clip(canvas[y0:y1, x0:x1].astype(np.int16) + delta, 0, 255).astype(np.uint8)


def _well(canvas, x, y, w, h, r=16, face=None, outer=None):
    _paste(canvas, T.pressed(int(w), int(h), r, face=face, outer=outer), int(x), int(y))


def _fit(img, w, h):
    """Letterbox into w x h on a dark backdrop, rounded corners."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    out = np.full((h, w, 3), T.bgr("video"), np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, (x, y, s)


def _paste_rounded(canvas, img, x, y, r):
    h, w = img.shape[:2]
    m = T.round_mask(w, h, r)
    corners = [(0, 0), (0, w - r), (h - r, 0), (h - r, w - r)]
    keep = [canvas[y + a:y + a + r, x + b:x + b + r].astype(np.float32) for a, b in corners]
    canvas[y:y + h, x:x + w] = img
    for (a, b), old in zip(corners, keep):
        mm = m[a:a + r, b:b + r]
        canvas[y + a:y + a + r, x + b:x + b + r] = (img[a:a + r, b:b + r] * mm + old * (1 - mm)).astype(np.uint8)


def _pill(canvas, P, x, y, text, color, size=15, anchor="l", fg="#FFFFFF", weight="semibold"):
    w = int(P.width(text, size, weight)) + 26
    h = size + 16
    if anchor == "r":
        x -= w
    elif anchor == "c":
        x -= w // 2
    cv2.rectangle(canvas, (x + h // 2, y), (x + w - h // 2, y + h), T.bgr(color), -1)
    cv2.circle(canvas, (x + h // 2, y + h // 2), h // 2, T.bgr(color), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x + w - h // 2, y + h // 2), h // 2, T.bgr(color), -1, cv2.LINE_AA)
    P.text(text, (x + w // 2, y + h // 2), size, fg, weight, "mm")
    return w


def plain_event(kind, msg):
    """Turn internal log lines into sentences anyone can read."""
    who, _, what = msg.partition(": ")
    m = re.search(r"rate~(\d+)", what)
    if what.startswith("calibrated"):
        return f"{who}: learned normal breathing" + (f" (about {m.group(1)} breaths/min)" if m else "")
    if what.startswith("cot drawn"):
        return f"{who}: cot drawn by hand"
    if what.startswith("auto-detected"):
        return f"{who}: breathing area found automatically"
    m = re.search(r"no breath for (\d+)s", what)
    if m and kind == "alarm":
        return f"{who}: ALARM - no breath for {m.group(1)} s"
    if m:
        return f"{who}: warning - no breath for {m.group(1)} s"
    if "acknowledged" in what:
        return f"{who}: alarm acknowledged by staff"
    if "POOR SIGNAL" in what:
        return f"{who}: picture too poor to see breathing - check camera"
    if "quality restored" in what:
        return f"{who}: picture good again"
    if "not visible" in what:
        return f"{who}: can't see the baby (covered or dark)"
    if "SIMULATED APNEA" in what:
        return f"{who}: simulated apnea started (video frozen for the demo)"
    if "recalibrating" in what:
        return f"{who}: re-learning normal breathing"
    if "vacated" in what:
        return f"{who}: cot marked empty"
    if "trending low" in what:
        return f"{who}: breaths getting shallower than usual - review"
    return msg


EVENT_COLOR = {"alarm": "alarm", "pre-alert": "pre", "technical": "pre", "warn": "watch", "ack": "ok", "info": "muted"}


class Sounder:
    """Non-blocking alert sound. Loud repeating tone for alarms, soft beep for pre-alerts."""

    def __init__(self):
        self.level = 0
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _beep(self, f, ms):
        if sys.platform == "win32":
            import winsound
            winsound.Beep(f, ms)
        else:
            print("\a", end="", flush=True)
            time.sleep(ms / 1000)

    def _loop(self):
        while not self._stop:
            if self.level >= TIER_ALARM:
                self._beep(1400, 250)
                time.sleep(0.15)
            elif self.level == TIER_PREALERT:
                self._beep(800, 120)
                time.sleep(3.0)
            else:
                time.sleep(0.1)

    def stop(self):
        self._stop = True


class Dashboard:
    def __init__(self, monitors, title="AIRA - ward monitor"):
        self.monitors = monitors
        self.title = title
        self.selected = 0
        self.sounder = Sounder()
        # resizable + maximisable; we draw at the window's real size ourselves, so never stretched
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        self._cols = None
        self._placed = False
        self._view = (1.0, 0, 0)
        self._fullscreen = False
        self._stop_rect = (-1, -1, -1, -1)
        self._quit = False
        self.allow_draw = False   # live camera pages: the user may draw the cots by hand (D)
        self.drawing = False      # draw-cots mode is on
        self.draw_boxes = []      # boxes drawn so far (camera-frame pixels)
        self.on_zones = None      # callback(list of boxes) whenever a cot is drawn / cleared
        self.on_auto = None       # callback() when "Draw cots myself" is switched off (back to automatic)
        self.manual = False       # "Draw cots myself" switch
        self._drag = None         # (x0, y0, x1, y1) box being dragged, camera-frame pixels
        self._cam_map = None      # layout -> camera-frame mapping of the camera view
        self._pressed = None      # button currently held down (drawn pressed in)
        self._clicked = []        # buttons released over themselves, waiting to run
        self._buttons = {}
        cv2.setMouseCallback(title, self._on_mouse)
        self.t = 0.0
        self.message = ""
        self.footer = ""          # extra help text (e.g. demo legend)
        self.on_key = None        # optional callback(key, selected_index, t)
        self.subtitle = "Contactless apnea monitor"
        self.rects = []
        self._mode = None
        self.camera = None        # full camera frame (set by the caller each frame)
        self.zones = []           # zone boxes in camera-frame pixels
        self.skin_box = None      # where the pulse is read from (heart-rate module)
        self.show_camera = False  # `c` toggles the full camera view
        self.full_view = False    # cards show the WHOLE camera picture with the body mesh drawn on it
        self.mesh = ((), ())      # (body points per person, face mesh points per face) from the tracker
        self._annotated = None
        self.scan_progress = None  # (elapsed, total) while searching for breathing

    # ------------------------------------------------------------------ layout
    @property
    def cols(self):
        if self._cols:
            return self._cols
        n = len(self.monitors)
        return 1 if n <= 1 else 2 if n in (2, 4) else 3

    def _card_w(self):
        return {1: 900, 2: 660, 3: 560}[self.cols]

    def _window_size(self):
        try:
            _, _, ww, wh = cv2.getWindowImageRect(self.title)
            return (ww, wh) if ww > 200 and wh > 200 else None
        except Exception:
            return None

    def _canvas_size(self):
        """Draw at the window's real size (no scaling, no wasted space)."""
        win = self._window_size() if self._placed else None
        if win is None:
            ax, ay, aw, ah = _work_area()
            win = (int(aw * 0.82), int(ah * 0.86))
        return max(640, win[0]), max(480, win[1])

    def _grid(self, Aw, Ah):
        """Pick columns x rows so the cards are as big as possible in the available area."""
        n = max(1, len(self.monitors))
        best = None
        for c in range(1, n + 1):
            r = -(-n // c)
            cw = (Aw - (c - 1) * GAP) / c
            ch = (Ah - (r - 1) * GAP) / r
            score = min(cw, ch * 1.45)     # a card is comfortable when it is not too squashed either way
            if best is None or score > best[0] + 1:
                best = (score, c, r, int(cw), int(ch))
        _, c, r, cw, ch = best
        self._cols = c
        return c, r, cw, ch

    def _present(self, canvas):
        """Show `canvas`; if it does not match the window (e.g. while resizing) fit it in."""
        H, W = canvas.shape[:2]
        if not self._placed:
            _fit_window(self.title)
            self._placed = True
            self._raise_frames = 3
        if getattr(self, "_raise_frames", 0) > 0:
            # bring the new page to the front (it can otherwise open behind other windows),
            # then let it behave like a normal window again
            self._raise_frames -= 1
            try:
                cv2.setWindowProperty(self.title, cv2.WND_PROP_TOPMOST, 1 if self._raise_frames > 0 else 0)
            except Exception:
                pass
        win = self._window_size()
        if win is None or win == (W, H):
            self._view = (1.0, 0, 0)
            cv2.imshow(self.title, canvas)
            return
        ww, wh = win
        sc = min(ww / W, wh / H)
        nw, nh = max(1, int(W * sc)), max(1, int(H * sc))
        img = cv2.resize(canvas, (nw, nh), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
        out = np.full((wh, ww, 3), BG, np.uint8)
        ox, oy = (ww - nw) // 2, (wh - nh) // 2
        out[oy:oy + nh, ox:ox + nw] = img
        self._view = (sc, ox, oy)
        cv2.imshow(self.title, out)

    def toggle_fullscreen(self):
        self._fullscreen = not self._fullscreen
        cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if self._fullscreen else cv2.WINDOW_NORMAL)

    def _true_mouse(self, x, y):
        return x, y

    def _target(self, x, y):
        for name, (x0, y0, x1, y1) in self._buttons.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
        if not self.show_camera and self.monitors:
            for i, (x0, y0, x1, y1) in enumerate(self.rects):
                if x0 <= x <= x1 and y0 <= y <= y1:
                    return f"card:{i}"
        return None

    def _on_mouse(self, ev, x, y, *_):
        """Buttons press in on mouse-down and act on release (like normal buttons)."""
        x, y = self._true_mouse(x, y)
        sc, ox, oy = self._view
        lx, ly = (x - ox) / sc, (y - oy) / sc  # window pixels -> layout pixels
        if self.drawing and self._cam_map is not None and self._draw_mouse(ev, lx, ly):
            return
        if ev not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP):
            return
        name = self._target(lx, ly)
        if ev == cv2.EVENT_LBUTTONDOWN:
            self._pressed = name
        else:
            if name is not None and name == self._pressed:
                self._clicked.append(name)
            self._pressed = None
        self._last_draw = 0.0  # redraw right away so the press is visible

    def _draw_mouse(self, ev, lx, ly):
        """Drag on the camera image to draw a cot box. Returns True if the event was used."""
        vx, vy, vw, vh, ox, oy, s, fw, fh = self._cam_map
        fx = min(max((lx - vx - ox) / s, 0), fw - 1)
        fy = min(max((ly - vy - oy) / s, 0), fh - 1)
        inside = vx + ox <= lx <= vx + ox + fw * s and vy + oy <= ly <= vy + oy + fh * s
        if ev == cv2.EVENT_LBUTTONDOWN and inside and self._target(lx, ly) is None:
            self._drag = (fx, fy, fx, fy)
        elif ev == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            self._drag = (self._drag[0], self._drag[1], fx, fy)
        elif ev == cv2.EVENT_LBUTTONUP and self._drag is not None:
            x0, y0, x1, y1 = self._drag
            self._drag = None
            x, y, w, h = int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0))
            if w >= 24 and h >= 24:           # ignore accidental clicks
                self.draw_boxes.append((x, y, w, h))
                if self.on_zones is not None:
                    self.on_zones(list(self.draw_boxes))   # the new cot starts monitoring right away
        else:
            return False
        self._last_draw = 0.0
        return True

    def start_drawing(self):
        self.drawing, self._drag, self.show_camera = True, None, True

    def set_manual(self, on):
        """The "Draw cots myself" switch."""
        if on and not self.manual:
            self.manual, self.draw_boxes = True, []
            if self.on_zones is not None:
                self.on_zones([])                # drop automatic cots; the user draws their own
            self.start_drawing()
        elif not on and self.manual:
            self.manual, self.drawing, self._drag, self.draw_boxes = False, False, None, []
            self.show_camera = True              # back to automatic search, shown in the camera view
            if self.on_auto is not None:
                self.on_auto()

    def _do_clicks(self, t):
        """Run clicked buttons (from the render loop, not the mouse callback). False = close page."""
        while self._clicked:
            name = self._clicked.pop(0)
            if name == "stop":
                self._quit = True
            elif name.startswith("key:"):
                if not self._handle_key(ord(name[4:]), t):
                    return False
            elif name == "toggle:manual":
                self.set_manual(not self.manual)
            elif name == "draw:clear":
                self.draw_boxes = []
                if self.on_zones is not None:
                    self.on_zones([])
            elif name == "draw:show":
                self.drawing, self._drag, self.show_camera = False, None, False
            elif name.startswith("card:"):
                i = int(name[5:])
                if i < len(self.monitors):
                    self.selected = i
                    self.monitors[i].acknowledge(t)
        return True

    # ------------------------------------------------------------------ words
    def _who(self, mon):
        return getattr(mon, "subject", "the baby")

    def _story(self, mon, snap):
        """(headline, colour, explanation) in plain language."""
        st, tier, p = snap["status"], snap["tier"], snap["pause_s"]
        who = self._who(mon)
        c = mon.cfg
        if snap["latched"] and tier < TIER_ALARM and st != "NOT VISIBLE":
            return "Alarm - please acknowledge", "alarm", "Breathing has restarted, but the alarm stays until someone confirms they checked. Click this card."
        if st == "NOT VISIBLE":
            if tier >= TIER_ALARM:
                return "Check now - can't see " + who, "alarm", "The picture is dark or covered. Click this card once checked."
            return "Can't see " + who, "watch", "The picture is dark or covered. An alarm follows if it stays like this."
        if tier >= TIER_ALARM:
            return f"No breath for {p:.0f} s - check now", "alarm", f"This is an apnea alarm. Go to {who}. Click this card once checked."
        if st == "POOR SIGNAL":
            return "Picture too poor to see breathing", "pre", "Add light or move the camera closer. Alarms still work, but may be less reliable."
        if tier == TIER_PREALERT:
            return f"No breath for {p:.0f} s", "pre", f"Early warning - look at {who} now. The alarm sounds at {c.alarm_s:.0f} s."
        if tier == TIER_PAUSE:
            return "Short breathing pause", "watch", f"Pauses up to ~10 s are normal in babies. Warning at {c.prealert_s:.0f} s, alarm at {c.alarm_s:.0f} s."
        if st == "MOVING":
            return "Moving", "watch", "Breathing can't be read during big movements. It resumes automatically."
        if st == "VACATED":
            return "Cot marked empty", "muted", "No alarms for this cot. Press V when the baby is back."
        if st in LEARNING:
            total = c.warmup_s + c.calibration_s
            el = 0.0 if mon.t0 is None else min(total, self.t - mon.t0)
            extra = " Breathing is faint - check the camera view (C)." if st == "WEAK SIGNAL" else ""
            return f"Learning normal breathing  {el:.0f}/{total:.0f} s", "accent", f"Every baby breathes differently, so AIRA learns this one first. Alarms already work.{extra}"
        return "Breathing normally", "ok", f"Watching the breathing movement. Last breath {p:.1f} s ago."

    def _stage(self, mon, snap):
        if snap["status"] in LEARNING:
            return 1
        return 2

    # ------------------------------------------------------------------ pieces
    def _stepper(self, canvas, P, x_right, y, stage, scale=1.0):
        """One-line progress tracker, right-aligned at x_right:  (1) Find - (2) Learn - (3) Watch"""
        names = ["Find", "Learn", "Watch"]
        fs = int(14 * scale)
        widths = [int(P.width(n, fs, "semibold")) for n in names]
        seg = int(22 * scale)
        total = sum(w + 22 for w in widths) + seg * 2
        x = x_right - total
        for i, n in enumerate(names):
            done, cur = i < stage, i == stage
            col = T.bgr("accent") if (done or cur) else T.bgr("faint")
            cv2.circle(canvas, (x + 7, y), 7, col, -1 if (done or cur) else 2, cv2.LINE_AA)
            P.text(n, (x + 20, y), fs, "text" if cur else ("muted" if done else "faint"), "semibold" if cur else "regular", "lm")
            x += widths[i] + 22
            if i < 2:
                cv2.line(canvas, (x + 2, y), (x + seg - 4, y), T.bgr("accent") if i < stage else T.bgr("faint"), 2, cv2.LINE_AA)
                x += seg

    def _card(self, canvas, P, x0, y0, W, H, mon, img, snap, sel, idx=0):
        """One cot. Lays itself out for its size: stacked when tall, video-left when wide."""
        wide = W / H > 1.25
        sc = max(0.72, min(1.45, W / 1150, H / 520)) if wide else max(0.72, min(1.45, W / 660, H / 760))
        # not enough room for the full card -> compact card (video + headline + one summary line)
        compact = (H < 430 * sc) if wide else (H < 520 * sc or W < 400)
        pad = int(22 * sc)
        tier = snap["tier"]
        alarm = snap["latched"] or tier == TIER_ALARM
        card_face = self._face_for(idx, snap)

        # title row
        P.text(mon.name, (x0 + pad, y0 + pad - 2), int(22 * sc), "text", "semibold")
        self._stepper(canvas, P, x0 + W - pad, y0 + pad + int(14 * sc), self._stage(mon, snap), min(sc, 1.1))
        top = y0 + pad + int(40 * sc)
        inner_w, inner_h = W - 2 * pad, y0 + H - pad - top
        story_h, metric_h = int(100 * sc), int(118 * sc)

        if compact:
            strip = int(58 * sc)
            vh = max(40, inner_h - strip - int(10 * sc))
            self._video(canvas, P, x0 + pad, top, inner_w, vh, img, snap, mon, sc, alarm)
            y = top + vh + int(12 * sc)
            head, hcol, _ = self._story(mon, snap)
            rate = snap["rate_bpm"]
            bits = [f"{rate:.0f}/min" if rate else "--/min", f"last breath {snap['pause_s']:.1f} s"]
            if getattr(mon, "heart_bpm", None):
                bits.append(f"heart {mon.heart_bpm:.0f}")
            line = "   ·   ".join(bits)
            hs = self._fit_label(P, head, int(19 * sc), inner_w - int(26 * sc))
            cv2.circle(canvas, (x0 + pad + 7, y + int(11 * sc)), int(7 * sc), T.bgr(hcol), -1, cv2.LINE_AA)
            P.text(head, (x0 + pad + int(22 * sc), y), hs, hcol if hcol != "muted" else "text", "semibold")
            P.text(line, (x0 + pad, y + int(30 * sc)), self._fit_label(P, line, int(14 * sc), inner_w, "regular"), "muted")
        elif wide:
            lw = int(inner_w * 0.56)
            self._video(canvas, P, x0 + pad, top, lw, inner_h, img, snap, mon, sc, alarm)
            rx, rw = x0 + pad + lw + pad, inner_w - lw - pad
            heart_row = rw < 560 * sc
            y = top
            y += self._story_block(canvas, P, rx, y, rw, story_h, mon, snap, sc, card_face, heart_inline=not heart_row)
            if heart_row and (getattr(mon, "heart_bpm", None) or getattr(mon, "heart_note", None)):
                hh = int(70 * sc)
                self._heart_widget(canvas, P, rx, y, min(rw, int(300 * sc)), hh, mon, sc, card_face)
                y += hh + int(14 * sc)
            stacked = rw < 470 * sc
            mh = metric_h * (2 if stacked else 1) - (0 if not stacked else int(10 * sc))
            self._metrics(canvas, P, rx, y, rw, mh, mon, snap, sc, card_face, stacked)
            y += mh + int(14 * sc)
            self._wave(canvas, P, rx, y, rw, max(int(70 * sc), y0 + H - pad - y), mon, snap, sc, card_face)
        else:
            wave_h = int(92 * sc)
            ih = inner_h - story_h - metric_h - wave_h - int(18 * sc)
            ih = max(int(90 * sc), ih)
            self._video(canvas, P, x0 + pad, top, inner_w, ih, img, snap, mon, sc, alarm)
            y = top + ih + int(18 * sc)
            y += self._story_block(canvas, P, x0 + pad, y, inner_w, story_h, mon, snap, sc, card_face, heart_inline=True)
            self._metrics(canvas, P, x0 + pad, y, inner_w, metric_h - int(16 * sc), mon, snap, sc, card_face, False)
            y += metric_h
            self._wave(canvas, P, x0 + pad, y, inner_w, max(int(60 * sc), y0 + H - pad - y), mon, snap, sc, card_face)

        if sel and len(self.monitors) > 1 and not alarm:
            nw = int(P.width(mon.name, int(22 * sc), "semibold"))
            yy = y0 + pad + int(32 * sc)
            cv2.line(canvas, (x0 + pad, yy), (x0 + pad + nw, yy), T.bgr("accent"), 3, cv2.LINE_AA)

    def _video(self, canvas, P, x, y, w, h, img, snap, mon, sc, alarm):
        vid, _ = _fit(img, int(w), int(h))
        if snap["status"] in ("NOT VISIBLE", "VACATED"):
            vid = (vid * 0.45).astype(np.uint8)
        if alarm:
            red = np.zeros_like(vid)
            red[:] = T.bgr("alarm")
            vid = cv2.addWeighted(vid, 0.72, red, 0.28, 0)
        _paste_rounded(canvas, vid, int(x), int(y), 18)
        banner = getattr(mon, "demo_banner", None)
        if banner:
            _pill(canvas, P, int(x + w // 2), int(y + 14), banner, "accent", int(15 * sc), "c")

    def _story_block(self, canvas, P, x, y, w, h, mon, snap, sc, face, heart_inline=True):
        """Headline + plain explanation (+ heart widget on the right). Returns height used."""
        head, hcol, expl = self._story(mon, snap)
        text_w = w
        if heart_inline and (getattr(mon, "heart_bpm", None) or getattr(mon, "heart_note", None)):
            hw, hh = int(250 * sc), int(70 * sc)
            self._heart_widget(canvas, P, x + w - hw, y + (h - hh) // 2 - int(10 * sc), hw, hh, mon, sc, face)
            text_w = w - hw - int(28 * sc)       # clear space between text and widget
        size = int(24 * sc)
        while size > 13 and P.width(head, size, "semibold") > text_w - int(26 * sc):
            size -= 1                              # shrink the headline rather than overlap
        cv2.circle(canvas, (x + 8, y + int(15 * sc)), int(8 * sc), T.bgr(hcol), -1, cv2.LINE_AA)
        P.text(head, (x + int(26 * sc), y + int((24 * sc - size) / 2)), size, hcol if hcol != "muted" else "text", "semibold")
        for i, ln in enumerate(_wrap(P, expl, int(15 * sc), text_w)[:2]):
            P.text(ln, (x, y + int((40 + 20 * i) * sc)), int(15 * sc), "muted")
        return h

    def _fit_label(self, P, text, size, width, weight="semibold"):
        while size > 9 and P.width(text, size, weight) > width:
            size -= 1
        return size

    def _metrics(self, canvas, P, x, y, w, h, mon, snap, sc, face, stacked):
        """Breathing rate / time since last breath / picture quality, in pressed wells."""
        gap = int(14 * sc)
        tier = snap["tier"]
        if stacked:     # 2 rows: [rate | picture] then [time since last breath]
            rh = (h - gap) // 2
            cells = {"rate": (x, y, (w - gap) // 2, rh), "pic": (x + (w + gap) // 2, y, (w - gap) // 2, rh),
                     "time": (x, y + rh + gap, w, rh)}
        else:
            ws = [int((w - 2 * gap) * f) for f in (0.27, 0.46, 0.27)]
            cells = {"rate": (x, y, ws[0], h), "time": (x + ws[0] + gap, y, ws[1], h), "pic": (x + ws[0] + ws[1] + 2 * gap, y, ws[2], h)}
        for c in cells.values():
            _well(canvas, c[0], c[1], c[2], c[3], 18, outer=face)
        lab = int(13 * sc)
        # rate
        cx, cy, cw_, ch_ = cells["rate"]
        P.text("BREATHING RATE", (cx + 16, cy + 12), self._fit_label(P, "BREATHING RATE", lab, cw_ - 28), "muted", "semibold")
        rate = snap["rate_bpm"]
        num = f"{rate:.0f}" if rate else "–"
        big = min(int(44 * sc), int(ch_ * 0.45))
        P.text(num, (cx + 14, cy + ch_ - 12), big, "text", "semibold", "ls")
        P.text("per min", (cx + 22 + P.width(num, big, "semibold"), cy + ch_ - 16), lab, "muted", "regular", "ls")
        # time since last breath
        cx, cy, cw_, ch_ = cells["time"]
        p = 0.0 if snap["status"] == "VACATED" else snap["pause_s"]
        val = f"{p:.1f} s"
        vs = int(20 * sc)
        P.text(val, (cx + cw_ - 16, cy + 8), vs, T.TIER_HEX[tier] if tier else "text", "semibold", "ra")
        room = cw_ - 44 - P.width(val, vs, "semibold")
        for label in ("TIME SINCE LAST BREATH", "SINCE LAST BREATH", "LAST BREATH"):
            if P.width(label, max(10, lab - 2), "semibold") <= room:
                break
        P.text(label, (cx + 16, cy + 12), self._fit_label(P, label, lab, room), "muted", "semibold")
        bx0, bx1 = cx + 18, cx + cw_ - 18
        by = cy + int(ch_ * 0.48)
        cv2.line(canvas, (bx0, by), (bx1, by), T.bgr("faint"), max(4, int(8 * sc)), cv2.LINE_AA)
        fill = bx0 + int((bx1 - bx0) * min(p, BAR_MAX_S) / BAR_MAX_S)
        if fill > bx0:
            cv2.line(canvas, (bx0, by), (fill, by), T.bgr(T.TIER_HEX[tier] if tier else "ok"), max(4, int(8 * sc)), cv2.LINE_AA)
        c = mon.cfg
        tick_gap = (bx1 - bx0) * 5 / BAR_MAX_S          # pixels between the 10/15/20 s marks
        for lim, col, name in ((c.pause_notice_s, "watch", "pause"), (c.prealert_s, "pre", "warn"), (c.alarm_s, "alarm", "alarm")):
            tx = bx0 + int((bx1 - bx0) * lim / BAR_MAX_S)
            cv2.line(canvas, (tx, by - int(9 * sc)), (tx, by + int(9 * sc)), T.bgr(col), 2, cv2.LINE_AA)
            P.text(f"{lim:.0f}s", (tx, by + int(13 * sc)), max(10, int(12 * sc)), col, "semibold", "ma")
            if tick_gap >= 40:                              # only when the words cannot collide
                P.text(name, (tx, by + int(28 * sc)), max(9, int(11 * sc)), col, "regular", "ma")
        # picture quality
        cx, cy, cw_, ch_ = cells["pic"]
        P.text("PICTURE", (cx + 16, cy + 12), lab, "muted", "semibold")
        q, qc = self._quality(snap)
        cv2.circle(canvas, (cx + 24, cy + int(ch_ * 0.5)), int(7 * sc), T.bgr(qc), -1, cv2.LINE_AA)
        P.text(q, (cx + 38, cy + int(ch_ * 0.5)), int(17 * sc), "text", "semibold", "lm")
        if snap["amp_trend_low"]:
            P.text("breaths shallower", (cx + 16, cy + ch_ - 12), max(10, int(12 * sc)), "pre", "regular", "ls")

    def _wave(self, canvas, P, x, y, w, h, mon, snap, sc, face):
        tier = snap["tier"]
        lab = int(13 * sc)
        _well(canvas, x, y, w, h, 18, outer=face)
        P.text("BREATHING MOVEMENT  -  last 15 s", (x + 16, y + 10), lab, "muted", "semibold")
        if getattr(mon, "demo_gt", []) and w > 520 * sc:
            lx = x + w - 16
            P.text("detected by AIRA", (lx, y + 10), lab, "muted", "regular", "ra")
            lx -= int(P.width("detected by AIRA", lab)) + 10
            cv2.line(canvas, (lx, y + 12), (lx, y + 24), T.bgr("text"), 3)
            lx -= 22
            P.text("marked by annotators", (lx, y + 10), lab, "muted", "regular", "ra")
            lx -= int(P.width("marked by annotators", lab)) + 10
            cv2.line(canvas, (lx, y + 12), (lx, y + 24), T.bgr("ok"), 3)
        sig = np.asarray(mon.signal_hist)
        top, bot = y + 36, y + h - 10
        mid = (top + bot) // 2
        if len(sig) > 2 and bot - top > 10:
            s_ = max(np.abs(sig).max(), 3 * mon.h, 1e-9)
            ys = (mid - sig / s_ * ((bot - top) / 2 - 6)).astype(int)
            xs = np.linspace(x + 16, x + w - 16, len(sig)).astype(int)
            col = T.bgr("accent") if tier == 0 else T.bgr(T.TIER_HEX[tier])
            cv2.polylines(canvas, [np.stack([xs, ys], 1)], False, col, 2, cv2.LINE_AA)
        win = 15.0
        tx = lambda bt: int(x + 16 + (bt - (self.t - win)) / win * (w - 32))
        for bt in getattr(mon, "demo_gt", []):
            if self.t - win <= bt <= self.t:
                cv2.line(canvas, (tx(bt), top - 6), (tx(bt), top + 4), T.bgr("ok"), 3)
        for bt in mon.breath_times:
            if self.t - win <= bt <= self.t:
                cv2.line(canvas, (tx(bt), bot - 8), (tx(bt), bot + 2), T.bgr("text"), 3)

    def _heart_widget(self, canvas, P, x, y, w, h, mon, sc, outer=None):
        """Neumorphic mini-card: raised panel, heart in a pressed-in round badge,
        rate in the middle, heartbeat line in its own pressed-in strip.
        The line is a stylised ECG shape drawn at the MEASURED rate (the camera measures
        a pulse rate from skin colour, not an electrical ECG)."""
        _raised(canvas, x, y, w, h, r=int(20 * sc), face=outer, d=max(3, int(5 * sc)), blur=max(4, int(7 * sc)))
        bpm = getattr(mon, "heart_bpm", None)
        note = getattr(mon, "heart_note", None)
        now = time.time()
        inner = 12 * sc
        # 1) round pressed badge with the beating heart
        bd = int(h - 2 * inner)
        bx, by = int(x + inner), int(y + inner)
        _well(canvas, bx, by, bd, bd, bd // 2, outer=outer)
        hx, hy = bx + bd // 2, by + bd // 2 + int(1 * sc)
        if bpm:
            phase = (now * bpm / 60.0) % 1.0
            size = bd * 0.26 * (1 + 0.28 * np.exp(-phase * 7.0))
            col = T.bgr("alarm")
        else:
            size, col = bd * 0.24, T.bgr("faint")
        cv2.fillPoly(canvas, [(HEART_SHAPE * size + (hx, hy)).astype(np.int32)], col, cv2.LINE_AA)
        # 2) numbers
        tx = bx + bd + int(12 * sc)
        cy = y + h // 2
        if bpm:
            num = f"{bpm:.0f}"
            big = max(20, int(24 * sc))
            P.text(num, (tx, cy + int(4 * sc)), big, "text", "semibold", "ls")
            P.text("/min", (tx + int(P.width(num, big, "semibold")) + 3, cy + int(2 * sc)), max(11, int(11 * sc)), "muted", "regular", "ls")
            P.text("locked on skin" if getattr(mon, "heart_locked", False) else "heart rate (beta)",
                   (tx, cy + int(20 * sc)), max(11, int(11 * sc)), "muted", "regular", "ls")
        else:
            short = "no colour cam" if note and "colour" in note else ("paused" if note and "Paused" in note else
                    ("finding pulse" if note and "pulse" in note.lower() else "measuring"))
            P.text("Heart rate", (tx, cy - int(2 * sc)), max(12, int(13 * sc)), "text", "semibold", "ls")
            P.text(short, (tx, cy + int(17 * sc)), max(11, int(12 * sc)), "muted", "regular", "ls")
        # 3) heartbeat line in a pressed strip (last 3 s, scrolling) - only when there is a rate
        if not bpm:
            return
        sx0 = int(x + w * 0.56)
        sw, sh = int(x + w - inner - sx0), int(h - 2 * inner)
        sy0 = int(y + inner)
        _well(canvas, sx0, sy0, sw, sh, int(10 * sc), outer=outer)
        ex0, ex1 = sx0 + int(8 * sc), sx0 + sw - int(8 * sc)
        ey = sy0 + int(sh * 0.62)
        n = max(8, ex1 - ex0)
        xs = np.linspace(ex0, ex1, n).astype(int)
        if bpm:
            tt = now - (1 - np.linspace(0, 1, n)) * 3.0
            ys = (ey - _ecg((tt * bpm / 60.0) % 1.0) * sh * 0.48).astype(int)
            lc = T.bgr("alarm")
        else:
            ys = np.full(n, ey)
            lc = T.bgr("faint")
        cv2.polylines(canvas, [np.stack([xs, ys], 1)], False, lc, max(1, int(2 * sc)), cv2.LINE_AA)
        if bpm:
            cv2.circle(canvas, (xs[-1], ys[-1]), max(2, int(3 * sc)), lc, -1, cv2.LINE_AA)

    def _face_for(self, idx, snap):
        """Card colour: darker while being clicked, pink/red when in alarm."""
        if self._pressed == f"card:{idx}":
            return "#DCD6CE"
        return self._card_face(snap)

    @staticmethod
    def _card_face(snap):
        if not (snap["latched"] or snap["tier"] == TIER_ALARM):
            return None
        return "#F2CFCB" if int(time.time() * 2.5) % 2 == 0 else "#EDDCD8"

    def _background(self, W, H, rects, faces, log_rect):
        """Panels + soft shadows never change between frames - draw them once per layout/alarm state."""
        key = (W, H, tuple(rects), faces, log_rect)
        cache = self.__dict__.setdefault("_bg_cache", {})
        if key not in cache:
            if len(cache) > 16:
                cache.clear()
            bg = np.full((H, W, 3), BG, np.uint8)
            for (x0, y0, x1, y1), face in zip(rects, faces):
                _raised(bg, x0, y0, x1 - x0, y1 - y0, r=28, face=face)
            lx, ly, lw, lh = log_rect
            _raised(bg, lx, ly, lw, lh, r=24)
            cache[key] = bg
        return cache[key].copy()

    def _quality(self, snap):
        st = snap["status"]
        if st == "POOR SIGNAL":
            return "Poor", "alarm"
        if st == "NOT VISIBLE":
            return "No view", "alarm"
        if st == "MOVING":
            return "Movement", "watch"
        if st in LEARNING:
            return "Learning", "accent"
        if st == "VACATED":
            return "-", "faint"
        return "Good", "ok"

    def _header(self, canvas, P, W, snaps):
        P.text("AIRA", (GAP + 6, 22), 34, "accent", "bold")
        alarms = [m.name for m, s in zip(self.monitors, snaps) if s["latched"] or s["tier"] == TIER_ALARM]
        warn = [m.name for m, s in zip(self.monitors, snaps) if s["tier"] in (TIER_PAUSE, TIER_PREALERT) and m.name not in alarms]
        if not snaps:
            txt, col = "Setting up", "accent"
        elif alarms:
            txt, col = "ALARM - " + ", ".join(alarms), "alarm"
        elif warn:
            txt, col = "Watch - " + ", ".join(warn), "pre"
        elif any(s["status"] in LEARNING for s in snaps):
            txt, col = "Learning normal breathing", "accent"
        else:
            txt, col = "Everyone breathing normally", "ok"
        clock = time.strftime("%H:%M")
        cw = int(P.width(clock, 22, "semibold"))
        P.text(clock, (W - GAP - 6, 30), 22, "text", "semibold", "ra")
        pw = _pill(canvas, P, W - GAP - cw - 24, 24, txt, col, 16, "r")
        # subtitle gets whatever room is left: smaller font, then trimmed, never under the pill
        room = (W - GAP - cw - 24 - pw - 24) - (GAP + 112)
        sub, size = self.subtitle, 17
        while size > 12 and P.width(sub, size) > room:
            size -= 1
        while len(sub) > 8 and P.width(sub, size) > room:
            sub = sub[:-2].rstrip() + "…" if not sub.endswith("…") else sub[:-3].rstrip() + "…"
        if room > 60:
            P.text(sub, (GAP + 112, 36 + (17 - size) // 2), size, "muted")

    def _action_button(self, canvas, P, x, y, label, name, face="bg", fg="text", bw=None, bh=38):
        bw = bw or int(P.width(label, 16, "semibold")) + 44
        down = self._pressed == name
        if down:
            _well(canvas, x, y + 1, bw, bh, bh // 2, face="well" if face == "bg" else face)
        else:
            _raised(canvas, x, y, bw, bh, r=bh // 2, face=None if face == "bg" else face, d=4, blur=5)
        P.text(label, (x + bw // 2, y + bh // 2 + (1 if down else 0)), 16, fg, "semibold", "mm")
        self._buttons[name] = (x, y, x + bw, y + bh)
        return bw

    def _switch(self, canvas, P, x, y, on, name, label):
        """Neumorphic on/off switch + label (clickable as a whole). Returns its width."""
        held = self._pressed == name
        _well(canvas, x, y, 52, 28, 14, face="accent" if on else "well")
        kx = x + (28 if on else 4) + ((-5 if on else 5) if held else 0)
        _raised(canvas, kx, y + 4, 20, 20, r=10, face="well" if held else "light", d=1, blur=2)
        P.text(label, (x + 64, y + 14), 15, "text", "regular", "lm")
        w = 64 + int(P.width(label, 15))
        self._buttons[name] = (x - 4, y - 6, x + w + 4, y + 34)
        return w

    def _stop_button(self, canvas, P, W, H):
        """Red Stop button, bottom right: closes this page. Returns its left edge."""
        bw, bh = 104, 38
        bx, by = W - GAP - 6 - bw, H - FOOT_H - 6
        down = self._pressed == "stop"
        if down:
            _well(canvas, bx, by + 1, bw, bh, 19, face="#B83A35")
        else:
            _raised(canvas, bx, by, bw, bh, r=19, face="alarm")
        P.text("Stop", (bx + bw // 2, by + bh // 2 + (1 if down else 0)), 16, "#FFFFFF", "semibold", "mm")
        self._stop_rect = (bx, by, bx + bw, by + bh)
        self._buttons["stop"] = self._stop_rect
        return bx

    def _log(self, canvas, P, x, y, W, Hh=130):
        P.text("WHAT HAPPENED", (x + 24, y + 18), 13, "muted", "semibold")
        keep = ("calibrated", "auto-detected", "SIMULATED", "recalibrating", "vacated")
        evs = [e for m in self.monitors for e in m.events if e[1] != "info" or any(k in e[2] for k in keep)]
        evs = sorted(evs, key=lambda e: e[0] or 0)[-max(1, (Hh - 44) // 20):][::-1]
        for i, (t, kind, msg) in enumerate(evs):
            yy = y + 44 + i * 20
            ts = "--:--" if t is None else f"{int(t // 60):02d}:{int(t % 60):02d}"
            cv2.circle(canvas, (x + 30, yy + 8), 5, T.bgr(EVENT_COLOR.get(kind, "muted")), -1, cv2.LINE_AA)
            P.text(ts, (x + 46, yy), 14, "muted")
            P.text(plain_event(kind, msg), (x + 100, yy), 15, "text" if kind != "info" else "muted")
        if not evs:
            P.text("Nothing yet. Alerts and changes will be listed here.", (x + 24, y + 50), 15, "muted")

    def _keys(self, canvas, P, x, y, items, compact=False):
        """Key hints that are also clickable buttons (press animation while held)."""
        for key, what in items:
            if compact and len(key) > 1:
                continue
            kw = int(P.width(key, 13, "semibold")) + 18
            name = "key:" + key.lower()
            clickable = len(key) == 1
            if not clickable:
                _well(canvas, x, y, kw, 26, 8)
            elif self._pressed == name:
                _well(canvas, x, y + 1, kw, 26, 8)
            else:
                _raised(canvas, x, y, kw, 26, r=8, d=2, blur=3)
            P.text(key, (x + kw // 2, y + 13 + (1 if self._pressed == name else 0)), 13, "text", "semibold", "mm")
            label = what.split()[0] if compact else what
            P.text(label, (x + kw + 8, y + 13), 14, "muted", "regular", "lm")
            tw = kw + 8 + int(P.width(label, 14))
            if clickable:
                self._buttons[name] = (x, y - 4, x + tw, y + 30)
            x += tw + (12 if compact else 16)
        return x

    def _keys_width(self, P, items, compact=False):
        w = 0
        for key, what in items:
            if compact and len(key) > 1:
                continue
            label = what.split()[0] if compact else what
            w += int(P.width(key, 13, "semibold")) + 18 + 8 + int(P.width(label, 14)) + (12 if compact else 16)
        return w

    def _footer(self, canvas, P, W, H, keys, right_extra=0):
        """Bottom bar: keys (+ 'Draw cots myself' switch) on the left, Stop on the right.
        Labels shorten if the window is narrow, so nothing overlaps."""
        sw = (64 + int(P.width("Draw cots myself", 15)) + 20) if self.allow_draw else 0
        room = W - 2 * GAP - 130 - right_extra - sw
        compact = self._keys_width(P, keys) > room
        x = self._keys(canvas, P, GAP + 6, H - FOOT_H, keys, compact)
        if self.allow_draw:
            self._switch(canvas, P, x + 6, H - FOOT_H - 1, self.manual, "toggle:manual", "Draw cots myself")
        return self._stop_button(canvas, P, W, H)

    # ------------------------------------------------------------------ camera view
    def _render_camera(self, snaps, t):
        W, H = self._canvas_size()
        PH = H - HEADER_H - GAP - FOOT_H - 10
        VH = max(160, PH - 150)
        canvas = np.full((H, W, 3), BG, np.uint8)
        P = T.Painter()
        self._header(canvas, P, W, snaps)
        px, py = GAP, HEADER_H
        _raised(canvas, px, py, W - 2 * GAP, PH, r=28)
        scanning = not self.monitors
        stage = 0 if scanning else 1
        P.text("Camera view", (px + 26, py + 22), 22, "text", "semibold")
        self._stepper(canvas, P, W - GAP - 26, py + 36, stage)
        if self.drawing:
            info = "Drag a box around each cot (the chest / belly of each baby). Each box starts monitoring right away."
        elif scanning:
            info = "Step 1 - finding the breathing area. Keep still: AIRA looks for something that rises and falls like breathing."
        else:
            info = "Red box = where breathing is measured. Green box = where the pulse is read. Adjust the camera so the red box sits on the chest or belly."
        for i, ln in enumerate(_wrap(P, info, 15, W - 2 * GAP - 52)[:2]):
            P.text(ln, (px + 26, py + 58 + 20 * i), 15, "muted")
        vx, vy, vw = px + 26, py + 104, W - 2 * GAP - 52
        if self.camera is not None:
            cam = self.camera.copy() if self.camera.ndim == 3 else cv2.cvtColor(self.camera, cv2.COLOR_GRAY2BGR)
            view, (ox, oy, s) = _fit(cam, vw, VH)
            self._cam_map = (vx, vy, vw, VH, ox, oy, s, cam.shape[1], cam.shape[0])
            for i, (x, y, w, h) in enumerate(self.draw_boxes if self.drawing else []):
                p0 = (int(ox + x * s), int(oy + y * s))
                p1 = (int(ox + (x + w) * s), int(oy + (y + h) * s))
                cv2.rectangle(view, p0, p1, T.bgr("accent"), 3, cv2.LINE_AA)
                P.text(f"Cot {i + 1}", (vx + p0[0] + 8, vy + p0[1] + 6), 15, "#FFFFFF", "semibold")
            if self.drawing and self._drag is not None:
                x0, y0, x1, y1 = self._drag
                p0 = (int(ox + min(x0, x1) * s), int(oy + min(y0, y1) * s))
                p1 = (int(ox + max(x0, x1) * s), int(oy + max(y0, y1) * s))
                cv2.rectangle(view, p0, p1, (255, 255, 255), 2, cv2.LINE_AA)
            for i, (x, y, w, h) in enumerate([] if self.drawing else self.zones):
                p0 = (int(ox + x * s), int(oy + y * s))
                p1 = (int(ox + (x + w) * s), int(oy + (y + h) * s))
                cv2.rectangle(view, p0, p1, T.bgr("alarm"), 3, cv2.LINE_AA)
                name = self.monitors[i].name if i < len(self.monitors) else f"zone {i + 1}"
                P.text(name, (vx + p0[0] + 8, vy + p0[1] + 6), 15, "#FFFFFF", "semibold")
            if self.skin_box is not None:
                x, y, w, h = self.skin_box
                p0 = (int(ox + x * s), int(oy + y * s))
                p1 = (int(ox + (x + w) * s), int(oy + (y + h) * s))
                cv2.rectangle(view, p0, p1, T.bgr("ok"), 2, cv2.LINE_AA)
                P.text("pulse read here", (vx + p0[0] + 6, vy + p1[1] - 22), 14, "#FFFFFF", "semibold")
        else:
            view = np.full((VH, vw, 3), T.bgr("video"), np.uint8)
            P.text("Waiting for camera...", (vx + vw // 2, vy + VH // 2), 18, "#FFFFFF", "regular", "mm")
        _paste_rounded(canvas, view, vx, vy, 18)
        if scanning and self.scan_progress and not self.drawing:
            el, tot = self.scan_progress
            bw, bh = 380, 64
            bx, by = vx + vw // 2 - bw // 2, vy + VH - bh - 18
            ov = canvas[by:by + bh, bx:bx + bw].astype(np.float32)
            m = T.round_mask(bw, bh, 16)
            canvas[by:by + bh, bx:bx + bw] = (ov * (1 - 0.75 * m) + np.array(T.bgr("text"), np.float32) * 0.75 * m).astype(np.uint8)
            frac = 0 if tot <= 0 else min(1.0, el / tot)
            cv2.line(canvas, (bx + 20, by + 44), (bx + bw - 20, by + 44), (90, 90, 90), 6, cv2.LINE_AA)
            cv2.line(canvas, (bx + 20, by + 44), (bx + 20 + int((bw - 40) * frac), by + 44), T.bgr("accent"), 6, cv2.LINE_AA)
            P.text(f"Finding the breathing area   {el:.0f} / {tot:.0f} s", (bx + bw // 2, by + 20), 15, "#FFFFFF", "semibold", "mm")
        elif self.message and scanning:
            P.text(self.message, (vx + vw // 2, vy + VH - 30), 16, "#FFFFFF", "semibold", "mm")
        keys = ([("C", "back to monitor")] if self.monitors and not self.drawing else []) + [("F", "full screen")]
        n = len(self.draw_boxes)
        extra = (int(P.width(f"Show monitor ({n})", 16, "semibold")) + 44 + 14 if n else 0) + (int(P.width("Clear", 16, "semibold")) + 58) if self.drawing else 0
        sx = self._footer(canvas, P, W, H, keys, extra)
        if self.drawing:
            by = H - FOOT_H - 6
            x = sx - 24
            n = len(self.draw_boxes)
            buttons = [("Clear", "draw:clear", "bg", "text" if n else "faint")]
            if n:
                buttons.insert(0, (f"Show monitor ({n})", "draw:show", "accent", "#FFFFFF"))
            for label, name, face, fg in buttons:
                bw = int(P.width(label, 16, "semibold")) + 44
                x -= bw
                self._action_button(canvas, P, x, by, label, name, face, fg, bw)
                x -= 14
        canvas = P.flush(canvas)
        self._present(canvas)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("f"):
            self.toggle_fullscreen()
        if k == ord("c") and self.monitors and not self.drawing:
            self.show_camera = False
        if k == ord("d") and self.allow_draw:
            self.set_manual(not self.manual)
        if self.monitors:
            self.sounder.level = max([TIER_ALARM if s["latched"] else s["tier"] for s in snaps] + [0])
        return True

    # ------------------------------------------------------------------ main
    def render(self, crops, snaps, t):
        """Returns False when the user quits."""
        self.t = t
        if not self._do_clicks(t):
            return False
        if self._quit:  # Stop button was clicked
            self.sounder.stop()
            return False
        now = time.time()
        if now - getattr(self, "_last_draw", 0.0) < 1 / 12:
            # drawing is skipped this frame (the analysis is not); still listen for keys
            self.sounder.level = max([TIER_ALARM if s["latched"] else s["tier"] for s in snaps] + [0])
            return self._handle_key(cv2.waitKey(1) & 0xFF, t)
        self._last_draw = now
        self._buttons = {}
        if not self.monitors or self.show_camera:
            return self._render_camera(snaps, t)
        W, H = self._canvas_size()
        log_h = LOG_H if H >= 760 else 112
        Aw = W - 2 * GAP - 10
        Ah = H - HEADER_H - log_h - FOOT_H - GAP - 10
        cols, rows, cw, ch = self._grid(Aw, Ah)
        rects = []
        for i in range(len(self.monitors)):
            r, c = divmod(i, cols)
            x0 = GAP + 5 + c * (cw + GAP)
            y0 = HEADER_H + r * (ch + GAP)
            rects.append((x0, y0, x0 + cw, y0 + ch))
        ly = HEADER_H + rows * (ch + GAP) + 4
        log_rect = (GAP + 5, ly, W - 2 * GAP - 10, log_h - 20)
        canvas = self._background(W, H, rects, tuple(self._face_for(i, s) for i, s in enumerate(snaps)), log_rect)
        P = T.Painter()
        self._header(canvas, P, W, snaps)
        self.rects = rects
        self._annotated = None
        if self.full_view and self.camera is not None:
            from . import mesh_draw
            breath = []
            for m in self.monitors:
                sig = np.asarray(m.signal_hist)
                scale = max(np.abs(sig).max(), 1e-9) if len(sig) else 1.0
                breath.append(float(sig[-1] / scale) if len(sig) else 0.0)
            hr = next((m.heart_bpm for m in self.monitors if getattr(m, "heart_bpm", None)), None)
            poses, faces = self.mesh
            self._annotated = mesh_draw.draw(self.camera, poses, faces, self.zones, breath, hr)
        for i, (m, c, s) in enumerate(zip(self.monitors, crops, snaps)):
            x0, y0, x1, y1 = rects[i]
            self._card(canvas, P, x0, y0, x1 - x0, y1 - y0, m, self._annotated if self._annotated is not None else c, s,
                       i == self.selected, i)
        self._log(canvas, P, log_rect[0], log_rect[1], log_rect[2], log_rect[3])
        keys = [("click", "confirm alarm"), ("C", "camera view"), ("F", "full screen"), ("R", "re-learn"), ("V", "empty cot")]
        self._footer(canvas, P, W, H, keys)
        canvas = P.flush(canvas)

        self.sounder.level = max([TIER_ALARM if s["latched"] else s["tier"] for s in snaps] + [0])
        self._present(canvas)
        return self._handle_key(cv2.waitKey(1) & 0xFF, t)

    def _handle_key(self, k, t):
        if k == ord("d") and self.allow_draw:
            self.set_manual(not self.manual)
            return True
        if k == ord("f"):
            self.toggle_fullscreen()
        if not self.monitors:
            return True
        if self.show_camera:
            if k == ord("c"):
                self.show_camera = False
            return True
        mon = self.monitors[self.selected]
        if k == ord("c") and self.camera is not None:
            self.show_camera = True
            if self.manual:
                self.drawing = True   # in "draw cots myself" mode the camera view is where you draw
        if k == ord("r"):
            mon.recalibrate(t)
        if k == ord("v"):
            mon.unvacate(t) if mon.vacated else mon.mark_vacated(t)
        if ord("1") <= k <= ord("9") and k - ord("1") < len(self.monitors):
            self.selected = k - ord("1")
        if k != 255 and self.on_key is not None:
            self.on_key(k, self.selected, t)
        return True

    def _card_height(self):
        W = self._card_w()
        sc = W / 660
        pad = int(24 * sc)
        return pad + 40 + int(W * (0.40 if self.cols == 1 else 0.48)) + int(96 * sc) + int(118 * sc) + int(92 * sc) + pad + 6
