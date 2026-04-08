import pandas as pd
import numpy as np

def parse_annotations(csv_path):
    """
    Parses a raw CSV file and processes annotations to correctly pair "!start" and "!end" segments
    while passing individual "Sleeping" and "!" point events gracefully.

    Returns:
        pd.DataFrame with columns: ['t_center_abs', 'rel_width_abs', 'class_id', 'is_segment']
    """
    df = pd.read_csv(csv_path)
    
    # 1. Strip whitespace
    df['label'] = df['label'].astype(str).str.strip()
    
    out_rows = []
    
    # Track segments
    start_events = []
    end_events = []
    
    for idx, row in df.iterrows():
        label = row['label']
        ts = float(row['timestamp_sec'])
        
        if label == "Sleeping":
            out_rows.append({
                't_center_abs': ts,
                'rel_width_abs': 0.0,
                'class_id': 0,
                'is_segment': False
            })
        elif label == "!":
            out_rows.append({
                't_center_abs': ts,
                'rel_width_abs': 0.0,
                'class_id': 1,
                'is_segment': False
            })
        elif label == "!start":
            start_events.append(ts)
        elif label == "!end":
            end_events.append(ts)
    
    # Rule 4: Match !start to nearest subsequent !end
    start_events.sort()
    end_events.sort()
    
    matched_starts = set()
    matched_ends = set()
    
    for t_start in start_events:
        # Find the earliest !end that occurs after this !start
        for t_end in end_events:
            if t_end > t_start and t_end not in matched_ends:
                t_center_abs = (t_start + t_end) / 2.0
                rel_width_abs = t_end - t_start
                out_rows.append({
                    't_center_abs': t_center_abs,
                    'rel_width_abs': rel_width_abs,
                    'class_id': 2,
                    'is_segment': True
                })
                matched_starts.add(t_start)
                matched_ends.add(t_end)
                break
                
    # Rule 5: Assert no unmatched !start or !end
    unmatched_starts = len(start_events) - len(matched_starts)
    unmatched_ends = len(end_events) - len(matched_ends)
    assert unmatched_starts == 0, f"Found {unmatched_starts} unmatched '!start' events in {csv_path}"
    assert unmatched_ends == 0, f"Found {unmatched_ends} unmatched '!end' events in {csv_path}"
    
    # Ensure consistent output if file is empty of valid labels
    if len(out_rows) == 0:
        return pd.DataFrame(columns=['t_center_abs', 'rel_width_abs', 'class_id', 'is_segment'])
        
    out_df = pd.DataFrame(out_rows)
    # Sort temporally to maintain chronological parsing easily
    out_df = out_df.sort_values('t_center_abs').reset_index(drop=True)
    
    return out_df
