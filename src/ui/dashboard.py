import streamlit as st
import os
import glob
import scipy.io as sio
import pandas as pd
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.config import PATHS

# ---- Configuration ----
MAT_DIR = PATHS["mat_dir"]
EVENTS_DIR = PATHS["processed_events_dir"]
SAMPLE_RATE = 500  # Native sampling frequency for this dataset

st.set_page_config(page_title="EEG MAT Explorer", layout="wide")

st.title("🧠 EEG Interactive Dataset Dashboard")
st.markdown("Explore identical `.mat` structures, visualize 29-channel contiguous EEG segments instantly, and cross-reference annotated clinical events (`'`, `!start`, `Waking`).")

st.sidebar.header("Navigation")

# Fetch all patients
mat_files = sorted([f for f in glob.glob(os.path.join(MAT_DIR, "*.mat")) if not os.path.basename(f).startswith("._")])
pids = [os.path.basename(f).replace(".mat", "") for f in mat_files]

if not pids:
    st.error(f"No MAT files found down inside {MAT_DIR}. Run extraction jobs first.")
    st.stop()

selected_pid = st.sidebar.selectbox("Select Patient to View", pids)

# Known Paths
mat_path = os.path.join(MAT_DIR, f"{selected_pid}.mat")
events_csv = os.path.join(EVENTS_DIR, f"{selected_pid}_events.csv")

# Caching helper functions so jumping around the slider is blazingly fast
@st.cache_data
def get_mat_structure(filepath):
    try:
        # Whosmat grabs keys/shapes natively without bloating RAM!
        return sio.whosmat(filepath)
    except Exception as e:
        return str(e)

@st.cache_data
def load_events(filepath):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['timestamp_sec'] = pd.to_numeric(df['timestamp_sec'], errors='coerce')
        return df
    return pd.DataFrame()

@st.cache_data
def get_global_events_distribution(events_dir):
    all_files = glob.glob(os.path.join(events_dir, "*_events.csv"))
    data = []
    for f in all_files:
        pid = os.path.basename(f).replace("_events.csv", "")
        df = pd.read_csv(f)
        if not df.empty and 'label' in df.columns:
            counts = df['label'].value_counts().to_dict()
            counts['Patient'] = pid
            data.append(counts)
    if data:
         df_global = pd.DataFrame(data).fillna(0)
         return df_global
    return pd.DataFrame()

@st.cache_data
def load_eeg_chunk(filepath, start_sec, end_sec, fs=SAMPLE_RATE):
    # Determine bounds
    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)
    
    mat = sio.loadmat(filepath, variable_names=['eeg_data'])
    if 'eeg_data' in mat:
        data = mat['eeg_data'] # Shape: (Channels, Time)
        
        # Ensure we don't index past dataset
        max_idx = data.shape[1]
        
        if start_idx >= max_idx:
            return None
        
        end_idx = min(end_idx, max_idx)
        return data[:, start_idx:end_idx].astype(np.float32) # Downcast for render speed
    return None

# Base calculations from metadata
structure = get_mat_structure(mat_path)
max_duration = 1000.0 # fallback

if isinstance(structure, list):
    for var in structure:
        if var[0] == 'eeg_data':
            points = var[1][1]
            max_duration = float(points) / SAMPLE_RATE
            break

# UI Dashboard
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("📁 Interior Scope (.mat Structure)")
    if isinstance(structure, list):
        struct_df = pd.DataFrame(structure, columns=['Variable Label', 'Array Shape', 'Data Type'])
        st.dataframe(struct_df, use_container_width=True, hide_index=True)
    else:
        st.error(f"Read format failure: {structure}")
        
    st.subheader("⏱️ Event Timecodes")
    events_df = load_events(events_csv)
    if not events_df.empty:
        st.dataframe(events_df, use_container_width=True, height=250, hide_index=True)
        
        st.subheader("📊 Local Event Distribution")
        label_counts = events_df['label'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']
        fig_bar = go.Figure(go.Bar(
            x=label_counts['Label'].astype(str),
            y=label_counts['Count'],
            marker_color='#1E88E5'
        ))
        fig_bar.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No recorded events exist for this specific patient.")

    st.subheader("🌍 Global Event Distribution")
    with st.expander("View distribution across all patients"):
        global_df = get_global_events_distribution(EVENTS_DIR)
        if not global_df.empty:
            global_df = global_df.set_index('Patient')
            st.dataframe(global_df, use_container_width=True)
            
            fig_glob = go.Figure()
            for col in global_df.columns:
                fig_glob.add_trace(go.Bar(name=col, x=global_df.index, y=global_df[col]))
            fig_glob.update_layout(barmode='stack', height=350, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig_glob, use_container_width=True)
        else:
            st.info("No global records available yet.")

with col2:
    st.subheader("📈 Interactive Time-Series Modality")
    st.info("💡 Selecting the entirety of an 8GB signal causes browser buffering exhaustion. Use the exact time-slider beneath to focus specifically onto local segments.")
    
    # Render time sliders inside the column
    t1, t2 = st.columns(2)
    start_time = t1.number_input("Window Setup Offset (seconds)", min_value=0.0, max_value=max_duration - 1, value=0.0, step=10.0)
    window_length = t2.slider("Window Gap Span (seconds)", min_value=1, max_value=60, value=15, step=1)
    
    end_time = min(start_time + window_length, max_duration)
    
    st.markdown(f"***Visible Horizon:*** `{start_time:.1f}s` ➔ `{end_time:.1f}s` | *(Maximum track duration: {max_duration:.1f}s)*")
    
    with st.spinner("Decoding sub-matrix matrix slices from MATLAB Engine..."):
        chunk = load_eeg_chunk(mat_path, start_time, end_time)
        
    if chunk is not None:
        channels, time_steps = chunk.shape
        t_axis = np.linspace(start_time, end_time, time_steps)
        
        # Calculate dynamic channel offsets visually
        std_val = np.std(chunk)
        offset = std_val * 4 if std_val > 0 else 5.0
        
        fig = go.Figure()
        
        # We loop backward so Ch_01 is visually at the top rather than bottom. Typical EEG rendering custom.
        for i in range(channels-1, -1, -1):
            fig.add_trace(go.Scatter(
                x=t_axis, 
                y=chunk[i, :] + ((channels - 1 - i) * offset), 
                mode='lines',
                line=dict(color='black', width=1.2),
                name=f"Ch_{i+1}",
                hoverinfo='none' # Massively speeds up DOM browser graph repainting
            ))
            
        # Draw Overlaid Annotation Lines precisely above the actual signals
        if not events_df.empty:
            window_events = events_df[(events_df['timestamp_sec'] >= start_time) & (events_df['timestamp_sec'] <= end_time)]
            
            for _, row in window_events.iterrows():
                ev_time = row['timestamp_sec']
                ev_label = str(row['label'])
                
                # Dynamic conditional coloring (e.g. spikes/discharges = red, phases = blue/purple)
                color = "#E53935" if "!" in ev_label else "#1E88E5"
                line_type = "dot" if "!" not in ev_label else "dash"
                
                fig.add_vline(x=ev_time, line_width=2.5, line_dash=line_type, line_color=color)
                fig.add_annotation(
                    x=ev_time, y=(channels * offset),
                    text=f"<b>{ev_label}</b>", showarrow=False,
                    font=dict(color=color, size=15),
                    yanchor="bottom"
                )
                
        yaxis_vals = [(channels - 1 - i) * offset for i in range(channels)]
        yaxis_text = [f"Ch_{i+1}" for i in range(channels)]

        fig.update_layout(
            height=max(600, channels * 30), # Grows dynamically if there's endless channels
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis=dict(
                tickmode='array',
                tickvals=yaxis_vals,
                ticktext=yaxis_text,
                showgrid=True,
                gridcolor='#f0f0f0',
                zeroline=False
            ),
            xaxis=dict(
                title="Exact Native Time (Seconds)",
                showgrid=True,
                gridcolor='#f0f0f0',
                zeroline=False
            ),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
        
    else:
        st.warning("Buffer constraint out of spatial bounds. Pull `Window Setup Offset` slider closer to zero index.")
