from pathlib import Path
import csv
import subprocess
import cv2

SRC = Path(r"D:\01_Projects\HuaweiCup-2026\02_Drafts\e\mosei-stage-v1.0")
OUT = Path(r"D:\01_Projects\HuaweiCup-2026\02_Drafts\e\q1-manual-review-normalized-v1.0")
FFMPEG = Path(r"D:\06_Apps\ffmpeg\bin\ffmpeg.exe")
FFPROBE = Path(r"D:\06_Apps\ffmpeg\bin\ffprobe.exe")

def probe(path):
    out = subprocess.check_output([str(FFPROBE), "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,nb_read_frames,avg_frame_rate,duration", "-of", "default=noprint_wrappers=1", str(path)], text=True)
    data = {}
    for line in out.splitlines():
        key, value = line.split("=", 1)
        data[key] = value
    return data

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in sorted(SRC.glob("*.mp4")):
        dst = OUT / f"{src.stem}__normalized.mp4"
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS)
        decoded = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            decoded += 1
        cap.release()
        if not fps or decoded <= 0:
            raise RuntimeError(f"Cannot decode {src}")
        duration = decoded / fps
        subprocess.run([str(FFMPEG), "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
                        "-vf", f"fps={fps:.8f},setpts=PTS-STARTPTS", "-af", f"atrim=duration={duration:.6f},asetpts=PTS-STARTPTS",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-ar", "44100", "-ac", "2", "-movflags", "+faststart", str(dst)], check=True)
        before, after = probe(src), probe(dst)
        rows.append({"sample": src.stem, "source": str(src), "normalized": str(dst),
                     "source_frames": before.get("nb_frames", ""), "decoded_frames": str(decoded),
                     "normalized_frames": after.get("nb_frames", ""), "normalized_decoded_frames": after.get("nb_read_frames", ""),
                     "source_duration": before.get("duration", ""), "normalized_duration": after.get("duration", "")})
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"Normalized: {len(rows)}")

if __name__ == "__main__":
    main()
