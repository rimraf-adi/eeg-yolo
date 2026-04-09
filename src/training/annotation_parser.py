import pandas as pd


CLASS_MAP = {
    '!': 0,
    '!start': 1,
    '!end': 2,
}

def parse_annotations(csv_path):
    """
    Parses a raw CSV file and keeps only the three active point classes:
    '!', '!start', and '!end'. Sleeping/Waking rows are removed entirely.

    Returns:
        pd.DataFrame with columns: ['t_center_abs', 'rel_width_abs', 'class_id', 'is_segment']
    """
    df = pd.read_csv(csv_path)
    
    # 1. Strip whitespace
    df['label'] = df['label'].astype(str).str.strip()

    # 2. Keep only the active point labels.
    df = df[df['label'].isin(CLASS_MAP)].copy()

    if len(df) == 0:
        return pd.DataFrame(columns=['t_center_abs', 'rel_width_abs', 'class_id', 'is_segment', 'label'])

    df['timestamp_sec'] = pd.to_numeric(df['timestamp_sec'], errors='coerce')
    df = df.dropna(subset=['timestamp_sec']).copy()

    if len(df) == 0:
        return pd.DataFrame(columns=['t_center_abs', 'rel_width_abs', 'class_id', 'is_segment', 'label'])

    df['t_center_abs'] = df['timestamp_sec'].astype(float)
    df['rel_width_abs'] = 0.0
    df['class_id'] = df['label'].map(CLASS_MAP).astype(int)
    df['is_segment'] = False

    out_df = df[['t_center_abs', 'rel_width_abs', 'class_id', 'is_segment', 'label']].copy()
    out_df = out_df.sort_values('t_center_abs').reset_index(drop=True)
    return out_df
