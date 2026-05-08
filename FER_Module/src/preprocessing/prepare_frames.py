from pathlib import Path
import cv2
import pandas as pd
import numpy as np

CSV_FILES = [
    "data/metadata/train_multilabel.csv",
    "data/metadata/val_multilabel.csv",
    "data/metadata/test_multilabel.csv",
]

RAW_VIDEO_ROOT = Path("data/raw")
FRAMES_ROOT = Path("data/frames")
NUM_FRAMES = 16


def load_all_clip_ids():
    clip_ids = []
    for csv_path in CSV_FILES:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        clip_ids.extend(df["ClipID"].tolist())
    return sorted(set(clip_ids))


def find_video_file(clip_id: str):
    clip_stem = Path(clip_id).stem  # 5000441001 from 5000441001.avi

    # first try exact file name
    matches = list(RAW_VIDEO_ROOT.rglob(clip_id))
    if matches:
        return matches[0]

    # then try any video file inside a folder named like the clip id
    folder_matches = list(RAW_VIDEO_ROOT.rglob(clip_stem))
    for folder in folder_matches:
        if folder.is_dir():
            for ext in ["*.avi", "*.mp4", "*.mov", "*.mkv"]:
                vids = list(folder.glob(ext))
                if vids:
                    return vids[0]

    return None


def extract_evenly_spaced_frames(video_path: Path, output_dir: Path, num_frames: int = 16):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return False, f"Could not read frame count: {video_path}"

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    saved = 0
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        out_path = output_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()

    if saved == num_frames:
        return True, f"Saved {saved} frames"
    return False, f"Saved only {saved}/{num_frames} frames"


def main():
    clip_ids = load_all_clip_ids()
    print(f"Total unique clips to process: {len(clip_ids)}")

    missing_videos = []
    failed = []
    done = 0

    for clip_id in clip_ids:
        video_path = find_video_file(clip_id)

        if video_path is None:
            missing_videos.append(clip_id)
            print(f"[MISSING] {clip_id}")
            continue

        clip_name = Path(clip_id).stem
        output_dir = FRAMES_ROOT / clip_name
        # skip if already extracted
        if output_dir.exists() and len(list(output_dir.glob("*.jpg"))) == NUM_FRAMES:
            print(f"[SKIP] {clip_id} already has {NUM_FRAMES} frames")
            done += 1
            continue
        ok, msg = extract_evenly_spaced_frames(video_path, output_dir, NUM_FRAMES)
        if ok:
            done += 1
            print(f"[OK] {clip_id} -> {output_dir}")
        else:
            failed.append((clip_id, msg))
            print(f"[FAILED] {clip_id} -> {msg}")

    print("\n===== SUMMARY =====")
    print(f"Processed successfully: {done}")
    print(f"Missing videos: {len(missing_videos)}")
    print(f"Failed: {len(failed)}")

    if missing_videos:
        print("\nFirst 10 missing videos:")
        for x in missing_videos[:10]:
            print(x)

    if failed:
        print("\nFirst 10 failed videos:")
        for x in failed[:10]:
            print(x)


if __name__ == "__main__":
    main()


#     reads the clip names from:
# train_multilabel.csv
# val_multilabel.csv
# test_multilabel.csv
# looks for each video file inside:
# data/raw/
# opens the video
# chooses 16 evenly spaced frame positions across the full clip
# saves those 16 frames as .jpg images inside:
# data/frames/<clip_name>/