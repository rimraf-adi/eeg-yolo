"""Unit tests for the point-event annotation pipeline."""

import os
import sys
import math

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.annotation_parser import parse_annotations
from src.training.target_builder import build_target, build_target_soft


SAMPLE_CSV_CONTENT = """\
timestamp_sec,duration,label
1.038,0,Sleeping
11.842,0,!
18.760,0,!
22.596,0,!
41.346,0,!start
72.362,0,!end
97.590,0,!start
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
767.230,0,!start
771.922,0,!end
793.038,0,!
796.196,0,!start
846.264,0,!end
872.726,0,!start
902.916,0,!end
924.780,0,!start
943.186,0,!end
946.452,0,!
960.110,0,!
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
1219.090,0,!end
1228.072,0,Waking
"""

CONFIG = {
    'S': 200,
    'window_size_sec': 10.0,
    'num_classes': 3,
    'gaussian_sigma_cells': 1.0,
    'gaussian_radius_cells': 3.0,
}

WINDOW_SIZE = CONFIG['window_size_sec']
STRIDE = 2.0
S = CONFIG['S']


@pytest.fixture
def sample_csv(tmp_path):
    csv_path = tmp_path / "test_events.csv"
    csv_path.write_text(SAMPLE_CSV_CONTENT)
    return str(csv_path)


@pytest.fixture
def parsed_annotations(sample_csv):
    return parse_annotations(sample_csv)


def test_parser_keeps_only_three_active_labels(parsed_annotations):
    labels = set(parsed_annotations['label'].astype(str).str.strip().unique())
    assert labels == {'!', '!start', '!end'}
    assert set(parsed_annotations['class_id'].unique()) == {0, 1, 2}


def test_sleeping_and_waking_are_removed(parsed_annotations):
    labels = parsed_annotations['label'].astype(str).str.strip().str.lower()
    assert 'sleeping' not in set(labels)
    assert 'waking' not in set(labels)


def test_parser_works_without_duration_column(tmp_path):
    csv_path = tmp_path / "events_no_duration.csv"
    csv_path.write_text(
        "timestamp_sec,label\n"
        "11.842,!\n"
        "41.346,!start\n"
        "72.362,!end\n"
        "100.000,Sleeping\n"
    )

    df = parse_annotations(str(csv_path))
    assert len(df) == 3
    assert set(df['class_id'].unique()) == {0, 1, 2}


def test_target_is_relativized_and_bounded(parsed_annotations):
    df = parsed_annotations
    max_time = df['t_center_abs'].max() + WINDOW_SIZE

    t_start = 0.0
    while t_start + WINDOW_SIZE <= max_time:
        t_end = t_start + WINDOW_SIZE
        target = build_target(df, t_start, t_end, CONFIG)

        assert target.shape == (S, 2 + CONFIG['num_classes'])

        obj_mask = target[:, 0] == 1.0
        if obj_mask.sum() > 0:
            offsets = target[obj_mask, 1]
            assert (offsets >= 0.0).all()
            assert (offsets < 1.0).all()

        t_start += STRIDE


def test_point_event_maps_to_expected_cell(parsed_annotations):
    df = parsed_annotations
    target = build_target(df, 10.0, 20.0, CONFIG)

    event_time = 11.842
    rel_center = (event_time - 10.0) / WINDOW_SIZE
    expected_cell = int(math.floor(rel_center * S))

    assert target[expected_cell, 0] == 1.0
    assert target[expected_cell, 2] == 1.0


def test_soft_target_spreads_around_event(parsed_annotations):
    df = parsed_annotations
    target = build_target_soft(df, 10.0, 20.0, CONFIG)

    center_time = 11.842
    rel_center = (center_time - 10.0) / WINDOW_SIZE
    expected_cell = int(math.floor(rel_center * S))

    assert target.shape == (S, 2 + CONFIG['num_classes'])
    assert float(target[expected_cell, 0]) > 0.0
    left = max(0, expected_cell - 2)
    right = min(S, expected_cell + 3)
    neighborhood = target[left:right, 0]
    assert float(neighborhood.max()) <= 1.0
    assert float(neighborhood.sum()) >= float(target[expected_cell, 0])


def test_soft_target_keeps_values_bounded(parsed_annotations):
    df = parsed_annotations
    target = build_target_soft(df, 10.0, 20.0, CONFIG)
    assert target.shape == (S, 2 + CONFIG['num_classes'])
    assert float(target[:, 0].min()) >= 0.0
    assert float(target[:, 0].max()) <= 1.0
    assert float(target[:, 1].min()) >= 0.0
    assert float(target[:, 1].max()) < 1.0


def test_event_reappears_in_overlapping_windows(parsed_annotations):
    df = parsed_annotations
    event_time = 11.842
    count = 0

    t_start = 0.0
    while t_start + WINDOW_SIZE <= 30.0:
        t_end = t_start + WINDOW_SIZE
        target = build_target(df, t_start, t_end, CONFIG)

        if t_start <= event_time < t_end:
            rel_center = (event_time - t_start) / WINDOW_SIZE
            expected_cell = int(math.floor(rel_center * S))
            if target[expected_cell, 0] == 1.0:
                count += 1

        t_start += STRIDE

    assert count >= 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
