# Security and privacy

AIRA watches babies, so it is built to keep video private by design:

- **Nothing leaves the computer.** AIRA makes no network connections while running. The only
  network code is `get_models.py`, which you run once by hand; it downloads from Google's
  official MediaPipe storage over HTTPS only and rejects any file whose SHA-256 checksum
  does not match the pinned value.
- **No video or pictures are saved.** Frames are processed in memory and dropped.
  The only file written is `aira_events.csv` (alert times and cot names), on your machine.
- **No accounts, passwords, keys or tracking** anywhere in the code.
- **No shell commands.** The launcher starts its own pages with a fixed argument list (no shell).
- **No unsafe loading.** Settings are read as plain JSON; no pickle/eval of outside files.
- **`.gitignore` blocks recordings, datasets, logs and zone files** from being committed by accident.

Please do not upload real patient video to issues or pull requests.

To report a security problem, open a GitHub issue *without* sensitive details and ask for a
private contact, or use GitHub's private vulnerability reporting.
