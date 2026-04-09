import glob
import os

import pandas as pd

from src.config import PATHS


def _resolve_event_dirs():
    """Return existing event directories from config without duplicates."""
    candidates = [
        PATHS.get("events_dir"),
        PATHS.get("processed_events_dir"),
    ]

    dirs = []
    seen = set()
    for d in candidates:
        if not d:
            continue
        abs_dir = os.path.abspath(d)
        if abs_dir in seen:
            continue
        if os.path.isdir(abs_dir):
            dirs.append(abs_dir)
            seen.add(abs_dir)
    return dirs


def clean_annotations_remove_zero_duration():
    event_dirs = _resolve_event_dirs()
    if not event_dirs:
        print("No configured event directories exist. Nothing to clean.")
        return

    total_files = 0
    updated_files = 0
    skipped_non_zero = 0

    for event_dir in event_dirs:
        csv_files = sorted(glob.glob(os.path.join(event_dir, "*_events.csv")))
        print(f"Scanning {len(csv_files)} files in {event_dir}")

        for csv_file in csv_files:
            total_files += 1
            try:
                df = pd.read_csv(csv_file)
            except Exception as exc:
                print(f"Failed reading {os.path.basename(csv_file)}: {exc}")
                continue

            if "duration" not in df.columns:
                continue

            duration_numeric = pd.to_numeric(df["duration"], errors="coerce")
            non_na = duration_numeric.notna()

            # Drop only when all known duration values are zero; otherwise keep for safety.
            if non_na.any() and (duration_numeric[non_na] == 0).all():
                df = df.drop(columns=["duration"])
                df.to_csv(csv_file, index=False)
                updated_files += 1
            else:
                skipped_non_zero += 1

    print("\nAnnotation cleanup complete.")
    print(f"Total files scanned: {total_files}")
    print(f"Files updated (duration removed): {updated_files}")
    print(f"Files kept with non-zero/invalid duration: {skipped_non_zero}")


if __name__ == "__main__":
    clean_annotations_remove_zero_duration()
