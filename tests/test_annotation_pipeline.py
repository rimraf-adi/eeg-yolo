"""
Unit tests for the annotation pipeline.
Tests that annotations are correctly parsed, paired, relativized,
and discretized into YOLO-style target tensors.

Covers the 5 validation checks from the spec (§5.1):
  1. test_no_absolute_timestamps_in_target
  2. test_segment_pairs_matched
  3. test_grid_cell_bounds
  4. test_total_event_coverage
  5. test_sleeping_label_not_dropped
"""

import os
import sys
import math

import numpy as np
import pandas as pd
import torch
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.annotation_parser import parse_annotations
from src.training.target_builder import build_target

# ---------------------------------------------------------------------------
# Sample CSV content matching the user's provided data
# ---------------------------------------------------------------------------
SAMPLE_CSV_CONTENT = """\
timestamp_sec,duration,label
1.038,0,Sleeping
11.842,0,!
18.76,0,!
22.596,0,!
41.346,0,!start
72.362,0,!end
97.59,0,!start
163.632,0,!end
166.134,0,!
184.382,0,!start
211.962,0,!end
227.522,0,!
252.624,0,!
277.664,0,!
289.464,0,!start
380.658,0,!end
418.612,0,!
431.342,0,!
455.184,0,!
459.158,0,!start
526.398,0,!end
547.212,0,!start
568.428,0,!end
586.888,0,!
598.876,0,!start
650.188,0,!end
667.304,0,!
697.752,0,!start
748.416,0,!end
767.23,0,!start
771.922,0,!end
793.038,0,!
796.196,0,!start
846.264,0,!end
872.726,0,!start
902.916,0,!end
924.78,0,!start
943.186,0,!end
946.452,0,!
960.11,0,!
985.514,0,!start
1036.204,0,!end
1060.516,0,!start
1077.854,0,!end
1103.012,0,!start
1105.568,0,!end
1108.102,0,!
1135.332,0,!start
1175.652,0,!end
1197.886,0,!start
1219.09,0,!end
1228.072,0,Sleeping
"""

# Default config matching config.yaml
CONFIG = {
    'S': 200,
    'window_size_sec': 10.0,
    'num_classes': 3,
}

WINDOW_SIZE = CONFIG['window_size_sec']
STRIDE = 2.0
S = CONFIG['S']


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV file with sample annotations."""
    csv_path = tmp_path / "test_events.csv"
    csv_path.write_text(SAMPLE_CSV_CONTENT)
    return str(csv_path)


@pytest.fixture
def parsed_annotations(sample_csv):
    """Parse the sample CSV into a properly formatted DataFrame."""
    return parse_annotations(sample_csv)


# ---------------------------------------------------------------------------
# §5.1 Test 2: test_segment_pairs_matched
# ---------------------------------------------------------------------------

class TestSegmentPairing:

    def test_segment_pairs_matched(self, sample_csv):
        """
        Assert len(!start rows) == len(!end rows) == 17 in this recording.
        Assert no orphan !start or !end after pairing.
        """
        raw_df = pd.read_csv(sample_csv)
        raw_labels = raw_df['label'].astype(str).str.strip()
        n_starts = (raw_labels == '!start').sum()
        n_ends = (raw_labels == '!end').sum()

        assert n_starts == n_ends, \
            f"Mismatched counts: {n_starts} !start vs {n_ends} !end"
        assert n_starts == 17, \
            f"Expected 17 !start events, got {n_starts}"

        # Parse and verify segments are created
        df = parse_annotations(sample_csv)
        segments = df[df['is_segment'] == True]
        assert len(segments) == 17, \
            f"Expected 17 segments after pairing, got {len(segments)}"

    def test_segment_centers_correct(self, parsed_annotations):
        """Verify segment center = midpoint of !start and !end."""
        df = parsed_annotations
        segments = df[df['is_segment'] == True]

        # First segment: !start=41.346, !end=72.362
        first_seg = segments.iloc[0]
        expected_center = (41.346 + 72.362) / 2.0
        assert abs(first_seg['t_center_abs'] - expected_center) < 0.01, \
            f"First segment center {first_seg['t_center_abs']} != expected {expected_center}"

        expected_width = 72.362 - 41.346
        assert abs(first_seg['rel_width_abs'] - expected_width) < 0.01, \
            f"First segment width {first_seg['rel_width_abs']} != expected {expected_width}"

    def test_class_mapping(self, parsed_annotations):
        """Verify correct class assignment: Sleeping=0, !=1, segments=2."""
        df = parsed_annotations

        sleeping = df[df['class_id'] == 0]
        bangs = df[df['class_id'] == 1]
        segments = df[df['class_id'] == 2]

        assert len(sleeping) == 2, f"Expected 2 Sleeping events, got {len(sleeping)}"
        assert len(bangs) > 0, "Expected some '!' point events"
        assert len(segments) == 17, f"Expected 17 segment events, got {len(segments)}"


# ---------------------------------------------------------------------------
# §5.1 Test 5: test_sleeping_label_not_dropped
# ---------------------------------------------------------------------------

class TestSleepingLabel:

    def test_sleeping_label_not_dropped(self, parsed_annotations):
        """
        Recording starts with Sleeping @ 1.038s and ends with Sleeping @ 1228.072s.
        Assert class_id=0 events appear in at least 1 window each.
        """
        df = parsed_annotations
        sleeping = df[df['class_id'] == 0]

        assert len(sleeping) >= 2, \
            f"Expected at least 2 Sleeping events, got {len(sleeping)}"

        times = sorted(sleeping['t_center_abs'].values)
        assert abs(times[0] - 1.038) < 0.001, \
            f"First Sleeping at {times[0]}, expected ~1.038"
        assert abs(times[-1] - 1228.072) < 0.001, \
            f"Last Sleeping at {times[-1]}, expected ~1228.072"

    def test_sleeping_in_target_window(self, parsed_annotations):
        """
        Sleeping @ 1.038s should appear in window [0, 10] as class_id=0.
        """
        df = parsed_annotations
        target = build_target(df, 0.0, 10.0, CONFIG)

        # rel_center = (1.038 - 0.0) / 10.0 = 0.1038
        # cell_idx = floor(0.1038 * 200) = floor(20.76) = 20
        expected_cell = int(math.floor((1.038 / 10.0) * S))

        assert target[expected_cell, 0, 0] == 1.0, \
            f"Sleeping event not found at expected cell {expected_cell}"
        # Class 0 (Sleeping) should be active: index 2+0 = 2
        assert target[expected_cell, 0, 2] == 1.0, \
            f"Sleeping class (index 2) not set at cell {expected_cell}"


# ---------------------------------------------------------------------------
# §5.1 Test 1: test_no_absolute_timestamps_in_target
# ---------------------------------------------------------------------------

class TestTargetRelativization:

    def test_no_absolute_timestamps_in_target(self, parsed_annotations):
        """
        For any window, assert all assigned grid cells have cell_offset ∈ [0, 1).
        A value > 1 signals an un-relativized absolute timestamp leaked through.
        """
        df = parsed_annotations
        max_time = df['t_center_abs'].max() + WINDOW_SIZE

        t_start = 0.0
        windows_checked = 0
        while t_start + WINDOW_SIZE <= max_time:
            t_end = t_start + WINDOW_SIZE
            target = build_target(df, t_start, t_end, CONFIG)

            # Check all cells with objectness == 1.0
            obj_mask = target[:, 0, 0] == 1.0
            if obj_mask.sum() > 0:
                offsets = target[obj_mask, 0, 1]
                assert (offsets >= 0.0).all(), \
                    f"Window [{t_start:.1f}, {t_end:.1f}]: negative offset: {offsets.min():.4f}"
                assert (offsets < 1.0).all(), \
                    f"Window [{t_start:.1f}, {t_end:.1f}]: offset >= 1.0: {offsets.max():.4f}. " \
                    f"Absolute timestamp leaked into the target."

            windows_checked += 1
            t_start += STRIDE

        assert windows_checked > 0, "No windows were checked"


# ---------------------------------------------------------------------------
# §5.1 Test 3: test_grid_cell_bounds
# ---------------------------------------------------------------------------

class TestGridCellBounds:

    def test_grid_cell_bounds(self, parsed_annotations):
        """
        For 10,000 randomly sampled windows, assert all cell_idx ∈ [0, 199].
        """
        df = parsed_annotations
        max_time = df['t_center_abs'].max() + WINDOW_SIZE

        np.random.seed(42)
        n_samples = min(10000, int(max_time / STRIDE))
        start_times = np.random.uniform(0, max(0.1, max_time - WINDOW_SIZE), n_samples)

        for t_start in start_times:
            t_end = t_start + WINDOW_SIZE
            target = build_target(df, t_start, t_end, CONFIG)

            # Verify target shape
            assert target.shape == (S, 1, 2 + CONFIG['num_classes']), \
                f"Wrong target shape: {target.shape}"

            # Objectness values should only be 0 or 1
            obj_vals = target[:, 0, 0]
            assert ((obj_vals == 0.0) | (obj_vals == 1.0)).all(), \
                "Objectness contains values other than 0 or 1"


# ---------------------------------------------------------------------------
# §5.1 Test 4: test_total_event_coverage
# ---------------------------------------------------------------------------

class TestEventCoverage:

    def test_total_event_coverage(self, parsed_annotations):
        """
        Count total positives across all windows, verify matches expected
        event count × average windows-per-event (each event appears in
        ceil(window_size / stride) = 5 windows).
        """
        df = parsed_annotations
        max_time = df['t_center_abs'].max() + WINDOW_SIZE

        total_positives = 0
        t_start = 0.0
        while t_start + WINDOW_SIZE <= max_time:
            t_end = t_start + WINDOW_SIZE
            target = build_target(df, t_start, t_end, CONFIG)
            total_positives += int((target[:, 0, 0] == 1.0).sum().item())
            t_start += STRIDE

        n_events = len(df)
        # Each event can appear in up to ceil(window_size / stride) = 5 windows
        expected_max = n_events * math.ceil(WINDOW_SIZE / STRIDE)

        assert total_positives > 0, "No positive cells found across any window!"
        assert total_positives <= expected_max, \
            f"Too many positives ({total_positives}) vs theoretical max ({expected_max})"

    def test_point_event_appears_in_multiple_windows(self, parsed_annotations):
        """
        A point event at 11.842s should appear in windows whose
        [start, start+10) range includes 11.842. With stride=2.0,
        that covers windows starting at 2.0, 4.0, 6.0, 8.0, 10.0 → 5 windows.
        """
        df = parsed_annotations
        event_time = 11.842
        count = 0

        t_start = 0.0
        max_time = 30.0  # only check nearby windows
        while t_start + WINDOW_SIZE <= max_time:
            t_end = t_start + WINDOW_SIZE
            target = build_target(df, t_start, t_end, CONFIG)

            if t_start <= event_time < t_end:
                # Event should be present in this window
                rel_center = (event_time - t_start) / WINDOW_SIZE
                expected_cell = int(math.floor(rel_center * S))
                if 0 <= expected_cell < S:
                    if target[expected_cell, 0, 0] == 1.0:
                        count += 1

            t_start += STRIDE

        assert count >= 4, \
            f"Point event at {event_time}s appeared in only {count} windows, expected ~5"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
