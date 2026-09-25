from pathlib import Path
import csv
import re
import subprocess

ROOT = Path(r"D:\01_Projects\HuaweiCup-2026\02_Drafts\e\q1-manual-review-normalized-v1.0")
FFMPEG = Path(r"D:\06_Apps\ffmpeg\bin\ffmpeg.exe")
FFPROBE = Path(r"D:\06_Apps\ffmpeg\bin\ffprobe.exe")
OUT = Path(r"D:\01_Projects\HuaweiCup-2026\02_Drafts\e\2026-09-25_q1-review-video-audit_v1.0.csv")


def probe(path: Path) -> dict:
    cmd = [str(FFPROBE), "-v", "error", "-count_frames", "-select_streams", "v:0",
           "-show_entries", "stream=nb_frames,nb_read_frames,avg_frame_rate,r_frame_rate,duration",
           "-of", "default=noprint_wrappers=1", str(path)]
    text = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace")
    result = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            result[k] = v
    return result


def ffmpeg_decoded_frames(path: Path) -> int:
    cmd = [str(FFMPEG), "-nostdin", "-v", "info", "-i", str(path),
           "-map", "0:v:0", "-f", "null", "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"FFmpeg failed for {path.name}: {p.stderr[-500:]}")
    matches = re.findall(r"frame=\s*(\d+)", p.stderr)
    if not matches:
        raise RuntimeError(f"No FFmpeg frame count for {path.name}")
    return int(matches[-1])


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    files = sorted(ROOT.glob("sample_*__normalized.mp4"))
    if len(files) != 100:
        raise RuntimeError(f"Expected 100 normalized videos, found {len(files)}")
    rows = []
    for i, path in enumerate(files, 1):
        p = probe(path)
        ff = ffmpeg_decoded_frames(path)
        declared = int(p["nb_frames"]) if p.get("nb_frames", "").isdigit() else None
        read = int(p["nb_read_frames"]) if p.get("nb_read_frames", "").isdigit() else None
        fps_text = p.get("avg_frame_rate", "")
        if "/" in fps_text:
            a, b = fps_text.split("/", 1)
            fps = float(a) / float(b) if float(b) else None
        else:
            fps = num(fps_text)
        duration = num(p.get("duration"))
        expected = round(duration * fps) if duration is not None and fps else None
        status = "ok" if declared == read == ff else "review"
        rows.append({"sample": path.stem.replace("__normalized", ""),
                     "file": str(path), "declared_frames": declared,
                     "ffprobe_read_frames": read, "ffmpeg_decoded_frames": ff,
                     "fps": fps, "duration_s": duration,
                     "duration_x_fps_rounded": expected,
                     "status": status})
        if i % 10 == 0:
            print(f"Audited: {i} / {len(files)}")
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"Saved: {OUT}")
    print(f"Audit OK: {len(rows) - len(bad)} / {len(rows)}")
    print(f"Needs review: {len(bad)}")
    for row in bad:
        print(row["sample"], row["declared_frames"], row["ffprobe_read_frames"], row["ffmpeg_decoded_frames"])


if __name__ == "__main__":
    main()
