"""Download the three small Google MediaPipe models AIRA uses (about 9 MB, Apache-2.0).

    python get_models.py

Each file comes from Google's official model storage over HTTPS and is checked against a
pinned SHA-256 checksum before it is saved. A file that does not match is rejected.
AIRA still runs without them (motion-only breathing area, pulse-map heart rate).
"""
import hashlib
import os
import urllib.request

BASE = "https://storage.googleapis.com/mediapipe-models/"
MODELS = {
    "blaze_face_short_range.tflite": (
        BASE + "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"),
    "pose_landmarker_lite.task": (
        BASE + "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"),
    "face_landmarker.task": (
        BASE + "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
}
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def main():
    os.makedirs(DEST, exist_ok=True)
    for name, (url, sha) in MODELS.items():
        path = os.path.join(DEST, name)
        if os.path.exists(path) and hashlib.sha256(open(path, "rb").read()).hexdigest() == sha:
            print(f"ok       {name}")
            continue
        assert url.startswith("https://storage.googleapis.com/")
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read(50 * 1024 * 1024)                       # hard size cap
        if hashlib.sha256(data).hexdigest() != sha:
            print(f"REJECTED {name}: checksum mismatch (file not saved)")
            continue
        with open(path, "wb") as f:
            f.write(data)
        print(f"saved    {name}")


if __name__ == "__main__":
    main()
