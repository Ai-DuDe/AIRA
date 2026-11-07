"""AIRA launcher: buttons instead of terminal commands.

Double-click AIRA.bat (or run `python launcher.py`). Each tile starts a mode in its own
window. Every page has a Stop button that closes that page; Stop on the launcher
closes AIRA completely (the launcher and any open page). Same soft theme as the dashboard.

The whole launcher is drawn as one picture at the window's current size, so it stays
sharp when resized or maximised (F = full screen).
"""
import math
import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from aira import theme as T

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "data", "air125", "AIR_125")
BASE_W, BASE_H = 960, 800          # design size; everything scales from this
TILE_W, TILE_H = 412, 132
HOVER = "#EEEBE6"   # face colour of a button under the mouse


def build_command(mode, camera=0, heart=True, manual_zones=False, fullscreen=False, video=None):
    """The exact command each tile runs (kept separate so it can be tested)."""
    py = sys.executable
    if mode == "me":
        cmd = [py, "run_live.py", "--source", str(camera), "--auto-zones", "--adult"]
    elif mode == "babies":
        cmd = [py, "run_live.py", "--source", str(camera), "--select-zones" if manual_zones else "--auto-zones"]
    elif mode == "demo":
        cmd = [py, "demo.py"] + (["--fullscreen"] if fullscreen else [])
    elif mode == "synthetic":
        cmd = [py, "run_live.py", "--source", "synthetic", "--auto-zones"]
    elif mode == "video":
        cmd = [py, "run_live.py", "--source", video, "--auto-zones"]
    else:
        raise ValueError(mode)
    if not heart and mode in ("me", "babies", "video"):
        cmd.append("--no-heart")
    return cmd


TILES = [
    ("me", "Monitor me", "Live camera on yourself. Breathe normally,\nthen hold your breath for 20 s."),
    ("demo", "Demo: real infant videos", "Three real recordings as a ward,\nwith one simulated apnea."),
    ("babies", "Monitor babies", "Live camera on one or more cots. Found\nautomatically, or draw the cots yourself."),
    ("synthetic", "Synthetic ward", "Computer-made babies with planned\napneas. No camera needed."),
    ("video", "Play a video file", "Run AIRA on any recording\n(mp4, avi, mov) from this computer."),
]
TILE_NAMES = {k: t for k, t, _ in TILES}
TOGGLES = [("heart", "Heart rate")]


class Scene:
    """Draws the launcher at scale `s` into an RGB picture and records the click regions."""

    def __init__(self, s, state):
        self.s = s
        self.st = state
        self.W, self.H = int(BASE_W * s), int(BASE_H * s)
        self.img = np.full((self.H, self.W, 3), T.rgb("bg"), np.uint8)
        self.texts, self.icons = [], []
        self.hits = []  # (name, x0, y0, x1, y1) in picture pixels

    def S(self, v):
        return int(round(v * self.s))

    # -- primitives -------------------------------------------------------------------
    def raised(self, x, y, w, h, r, face=None, d=7, blur=9):
        a, m = T.raised(self.S(w), self.S(h), self.S(r), max(1, self.S(d)), max(1, self.S(blur)), face=face)
        x, y = self.S(x) - m, self.S(y) - m
        hh, ww = a.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(self.W, x + ww), min(self.H, y + hh)
        if x1 > x0 and y1 > y0:  # blend as a difference from the background so shadows add up
            delta = a[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.int16) - np.array(T.rgb("bg"), np.int16)
            self.img[y0:y1, x0:x1] = np.clip(self.img[y0:y1, x0:x1].astype(np.int16) + delta, 0, 255).astype(np.uint8)

    def pressed(self, x, y, w, h, r, face="bg", d=6, blur=8, outer=None):
        a = T.pressed(self.S(w), self.S(h), self.S(r), max(1, self.S(d)), max(1, self.S(blur)), face=face, outer=outer)
        X, Y = self.S(x), self.S(y)
        self.img[Y:Y + a.shape[0], X:X + a.shape[1]] = a[:max(0, self.H - Y), :max(0, self.W - X)]

    def text(self, s, x, y, size, color="text", weight="regular", anchor="la"):
        self.texts.append((s, (self.S(x), self.S(y)), max(6, self.S(size)), color, weight, anchor))

    def hit(self, name, x, y, w, h):
        self.hits.append((name, self.S(x), self.S(y), self.S(x + w), self.S(y + h)))

    def icon(self, kind, cx, cy, enabled):
        self.icons.append((kind, cx, cy, enabled))

    def finish(self):
        im = Image.fromarray(self.img)
        dr = ImageDraw.Draw(im)
        for s, xy, size, color, weight, anchor in self.texts:
            dr.multiline_text(xy, s, font=T.font(size, weight), fill=T.rgb(color), anchor=anchor, spacing=self.S(5))
        S, w = self.S, max(2, self.S(3))
        for kind, cx, cy, en in self.icons:
            col = T.rgb("accent" if en else "faint")
            cx, cy = S(cx), S(cy)
            if kind == "me":
                dr.ellipse((cx - S(9), cy - S(20), cx + S(9), cy - S(2)), outline=col, width=w)
                dr.arc((cx - S(20), cy + S(2), cx + S(20), cy + S(40)), 200, 340, fill=col, width=w)
            elif kind == "babies":
                for dx in (-20, 0, 20):
                    dr.rounded_rectangle((cx + S(dx - 8), cy - S(10), cx + S(dx + 8), cy + S(12)), radius=S(4), outline=col, width=w)
                    dr.ellipse((cx + S(dx - 3), cy - S(6), cx + S(dx + 3), cy), fill=col)
            elif kind == "demo":
                dr.rounded_rectangle((cx - S(20), cy - S(15), cx + S(20), cy + S(15)), radius=S(6), outline=col, width=w)
                dr.polygon([(cx - S(6), cy - S(8)), (cx - S(6), cy + S(8)), (cx + S(9), cy)], fill=col)
            elif kind == "synthetic":
                pts = [(cx - S(22) + S(i), cy - S(10) * math.sin(i / 44 * 2 * math.pi * 1.5)) for i in range(45)]
                dr.line(pts, fill=col, width=w, joint="curve")
            elif kind == "video":
                dr.rounded_rectangle((cx - S(20), cy - S(12), cx + S(20), cy + S(16)), radius=S(5), outline=col, width=w)
                dr.rounded_rectangle((cx - S(20), cy - S(18), cx - S(4), cy - S(10)), radius=S(3), fill=col)
        return im

    # -- the page ---------------------------------------------------------------------
    def build(self):
        st = self.st
        self.text("AIRA", 40, 58, 40, "accent", "bold", "lm")
        self.text("Contactless infant apnea monitor  -  choose what to run", 158, 62, 16, "muted", "regular", "lm")
        for i, (kind, title, desc) in enumerate(TILES):
            x, y = 48 + (i % 2) * (TILE_W + 40), 112 + (i // 2) * (TILE_H + 34)
            enabled = kind != "demo" or st["has_data"]
            if not enabled:
                desc = "Needs the AIR-125 download\nin data/air125 (see README)."
            if st["active"] == kind and enabled:
                self.pressed(x, y, TILE_W, TILE_H, 26)      # pressed in while the mouse button is held
            elif st["hover"] == kind and enabled:
                self.raised(x, y, TILE_W, TILE_H, 26, face=HOVER, d=9, blur=11)   # lifts a little
            else:
                self.raised(x, y, TILE_W, TILE_H, 26)
            hovered = st["hover"] == kind and enabled and st["active"] != kind
            self.pressed(x + 24, y + 34, 64, 64, 32, outer=HOVER if hovered else None)
            self.icon(kind, x + 56, y + 66, enabled)
            dy = 1 if st["active"] == kind else 0
            self.text(title, x + 112, y + 28 + dy, 21, "text" if enabled else "faint", "semibold")
            self.text(desc, x + 112, y + 64 + dy, 14, "muted" if enabled else "faint")
            if enabled:
                self.hit(kind, x, y, TILE_W, TILE_H)
        # how it works (not clickable)
        x, y = 48 + TILE_W + 40, 112 + 2 * (TILE_H + 34)
        self.pressed(x, y, TILE_W, TILE_H, 26)
        self.text("HOW IT WORKS", x + 24, y + 18, 13, "muted", "semibold")
        for i, (a, b) in enumerate([("Find", "spots what rises and falls like breathing"),
                                    ("Learn", "learns this baby's normal breathing (~25 s)"),
                                    ("Watch", "warns at 15 s, alarms at 20 s without a breath")]):
            yy = y + 46 + i * 26
            self.raised(x + 24, yy + 2, 16, 16, 8, face="accent", d=1, blur=2)
            self.text(str(i + 1), x + 32, yy + 10, 11, "#FFFFFF", "bold", "mm")
            self.text(a, x + 52, yy, 15, "text", "semibold")
            self.text(b, x + 52 + T.font(15, "semibold").getlength(a) + 8, yy + 1, 14, "muted")
        # options bar
        oy = 112 + 3 * (TILE_H + 34) + 4
        self.raised(48, oy, BASE_W - 96, 64, 22)
        cy = oy + 32
        self.text("Camera", 74, cy, 15, "muted", "regular", "lm")
        for name, cx, lab in (("cam-", 160, "-"), ("cam+", 232, "+")):
            if st["active"] == name:
                self.pressed(cx - 15, cy - 15, 30, 30, 15, d=3, blur=4)
            else:
                self.raised(cx - 15, cy - 15, 30, 30, 15, face=HOVER if st["hover"] == name else None, d=3, blur=4)
            self.text(lab, cx, cy, 18, "text", "semibold", "mm")
            self.hit(name, cx - 18, cy - 18, 36, 36)
        self.text(str(st["camera"]), 196, cy, 18, "text", "semibold", "mm")
        ox = 292
        for key, label in TOGGLES:
            on = st["opts"][key]
            self.pressed(ox, cy - 14, 52, 28, 14, face="accent" if on else "well", d=3, blur=4)
            held = st["active"] == "opt:" + key
            kx = ox + (30 if on else 4) + ((-6 if on else 6) if held else 0)   # knob starts to slide while pressed
            self.raised(kx, cy - 10, 20, 20, 10, face="well" if held else "light", d=1, blur=2)
            self.text(label, ox + 64, cy, 15, "text", "regular", "lm")
            lw = T.font(15).getlength(label)
            self.hit("opt:" + key, ox - 4, cy - 18, 72 + lw, 36)
            ox += 72 + int(lw) + 34
        # status + stop
        sy = BASE_H - 44
        self.text(st["status"], 48, sy, 14, st["status_color"], "regular", "lm")
        sx = BASE_W - 30 - 110
        if st["active"] == "stop":
            self.pressed(sx, sy - 20, 110, 40, 20, face="#B83A35", d=4, blur=5)
        else:
            self.raised(sx, sy - 20, 110, 40, 20, face="#DA5A54" if st["hover"] == "stop" else "alarm", d=4, blur=5)
        self.text("Stop", sx + 55, sy + (1 if st["active"] == "stop" else 0), 16, "#FFFFFF", "semibold", "mm")
        self.hit("stop", sx, sy - 20, 110, 40)
        return self.finish()


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AIRA")
        self.configure(bg=T.HEX["bg"])
        self.minsize(560, 470)
        self.proc = None
        self.running_name = ""
        self.state = dict(camera=0, opts={"heart": True}, hover=None, active=None,
                          status="Pick a tile. Each page has its own Stop button. Stop here closes AIRA and every page.",
                          status_color="muted", running=False, has_data=os.path.isdir(DATASET))
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        s0 = min(1.0, 0.9 * sw / BASE_W, 0.85 * sh / BASE_H)
        self.geometry(f"{int(BASE_W * s0)}x{int(BASE_H * s0)}")
        self.cv = tk.Canvas(self, bg=T.HEX["bg"], highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.img_id = self.cv.create_image(0, 0, anchor="nw")
        self.hits, self.off, self.photo = [], (0, 0), None
        self._pending = None
        self.cv.bind("<Configure>", lambda e: self.redraw(delay=60))
        self.cv.bind("<Motion>", self.on_move)
        self.cv.bind("<ButtonPress-1>", self.on_press)
        self.cv.bind("<ButtonRelease-1>", self.on_click)
        toggle = lambda e: self.attributes("-fullscreen", not self.attributes("-fullscreen"))
        self.bind("<F11>", toggle)
        self.bind("f", toggle)   # same key as the other pages
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll_id = self.after(400, self.poll)

    # ---- drawing --------------------------------------------------------------------
    def redraw(self, delay=0):
        """Re-draw at the current window size (debounced while the user drags the edge)."""
        if self._pending is not None:
            self.after_cancel(self._pending)
        self._pending = self.after(delay, self._draw)

    def _draw(self):
        self._pending = None
        w, h = max(200, self.cv.winfo_width()), max(200, self.cv.winfo_height())
        sc = Scene(min(w / BASE_W, h / BASE_H), self.state)
        self.photo = ImageTk.PhotoImage(sc.build())
        ox, oy = (w - sc.W) // 2, (h - sc.H) // 2
        self.cv.coords(self.img_id, ox, oy)
        self.cv.itemconfigure(self.img_id, image=self.photo)
        self.hits, self.off = sc.hits, (ox, oy)

    def _hit_at(self, x, y):
        x, y = x - self.off[0], y - self.off[1]
        for name, x0, y0, x1, y1 in self.hits:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
        return None

    def on_move(self, e):
        name = self._hit_at(e.x, e.y)
        self.cv.configure(cursor="hand2" if name else "")
        hov = name if (name in TILE_NAMES or name in ("cam-", "cam+", "stop") or (name or "").startswith("opt:")) else None
        if hov != self.state["hover"]:
            self.state["hover"] = hov
            self.redraw()

    def on_press(self, e):
        """Mouse down: show the button pressed in. The action runs on release."""
        self.state["active"] = self._hit_at(e.x, e.y)
        self._draw()

    def on_click(self, e):
        name = self._hit_at(e.x, e.y)
        was = self.state["active"]
        self.state["active"] = None
        if name is None or name != was:   # released somewhere else: cancel, like normal buttons
            self.redraw()
            return
        if name == "cam-":
            self.state["camera"] = max(0, self.state["camera"] - 1)
        elif name == "cam+":
            self.state["camera"] = min(5, self.state["camera"] + 1)
        elif name.startswith("opt:"):
            k = name[4:]
            self.state["opts"][k] = not self.state["opts"][k]
        elif name == "stop":
            self.on_close()   # Stop on the launcher closes AIRA: the launcher and any open page
            return
        else:
            self.launch(name, TILE_NAMES[name])
        self.redraw()

    def _status(self, text, color="muted"):
        self.state["status"], self.state["status_color"] = text, color

    # ---- actions --------------------------------------------------------------------
    def launch(self, kind, name):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showinfo("AIRA", f"'{self.running_name}' is still open.\nClose it with the Stop button on that page first.")
            return
        video = None
        if kind == "video":
            video = filedialog.askopenfilename(title="Choose a video", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
            if not video:
                return
        o = self.state["opts"]
        cmd = build_command(kind, self.state["camera"], o["heart"], False, False, video)
        # pages watch this PID and close themselves if the launcher goes away (even if it crashes)
        env = dict(os.environ, AIRA_PARENT_PID=str(os.getpid()))
        self.proc = subprocess.Popen(cmd, cwd=HERE, env=env)
        self.running_name = name
        self.state["running"] = True
        self._status(f"Running: {name}.  Use Stop on that page to close it.", "accent")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()

    def poll(self):
        if self.proc is not None and self.proc.poll() is not None:
            code = self.proc.returncode
            ok = code in (0, 1, -15, 15)
            self._status("Page closed. Pick another tile." if ok else f"'{self.running_name}' closed with an error (code {code}). See the terminal.",
                         "muted" if ok else "alarm")
            self.state["running"] = False
            self.proc = None
            self.redraw()
        self._poll_id = self.after(400, self.poll)

    def on_close(self):
        self.stop()
        if self.proc is not None:
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
        for job in (getattr(self, "_poll_id", None), self._pending):
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
        self.destroy()   # always close the launcher (bug: this used to run only when a page was open)


def _sharp_on_windows():
    """Crisp text on scaled (125%/150%) Windows displays instead of blurry upscaling."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


if __name__ == "__main__":
    _sharp_on_windows()
    Launcher().mainloop()
