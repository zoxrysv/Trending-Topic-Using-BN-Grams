import streamlit as st
import pandas as pd
import os

# 1. Konfigurasi
SAVE_FOLDER = "data"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# List index untuk 4 dataset
INDEX_LIST = [1, 2, 3, 4]

# 2. Inisialisasi Session State menggunakan Dictionary
if 'datasets' not in st.session_state:
    st.session_state['datasets'] = {i: None for i in INDEX_LIST}
if 'ground_truths' not in st.session_state:
    st.session_state['ground_truths'] = {i: None for i in INDEX_LIST}

# Fungsi simpan file
def save_uploaded_file(uploaded_file, custom_name):
    path = os.path.join(SAVE_FOLDER, f"{custom_name}.csv")
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

# Auto-load dari folder 'data' jika file sudah ada
for i in INDEX_LIST:
    ds_path = f"data/dataset_{i}.csv"
    gt_path = f"data/ground_truth_{i}.csv"
    
    if st.session_state['datasets'][i] is None and os.path.exists(ds_path):
        st.session_state['datasets'][i] = pd.read_csv(ds_path, sep=";")
    
    if st.session_state['ground_truths'][i] is None and os.path.exists(gt_path):
        st.session_state['ground_truths'][i] = pd.read_csv(gt_path)

# --- UI STREAMLIT ---
st.title("🚀 Multi-Dataset Management")
st.info("Upload hingga 4 dataset dan ground truth untuk diproses.")

# Gunakan Tabs agar tampilan rapi tidak memanjang ke bawah
tabs = st.tabs([f"Dataset {i}" for i in INDEX_LIST])

for i, tab in enumerate(tabs, 1):
    with tab:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Data Source {i}")
            up_ds = st.file_uploader(f"Upload Dataset {i}", type=["csv"], key=f"up_ds_{i}")
            if up_ds:
                save_uploaded_file(up_ds, f"dataset_{i}")
                st.session_state['datasets'][i] = pd.read_csv(up_ds, sep=";")

                st.success(f"Dataset {i} tersimpan!")
            
            if st.session_state['datasets'][i] is not None:
                st.dataframe(st.session_state['datasets'][i].head(5), use_container_width=True)

        with col2:
            st.subheader(f"Ground Truth {i}")
            up_gt = st.file_uploader(f"Upload Ground Truth {i}", type=["csv"], key=f"up_gt_{i}")
            if up_gt:
                save_uploaded_file(up_gt, f"ground_truth_{i}")
                st.session_state['ground_truths'][i] = pd.read_csv(up_gt)
                st.success(f"Ground Truth {i} tersimpan!")
                
            if st.session_state['ground_truths'][i] is not None:
                st.dataframe(st.session_state['ground_truths'][i].head(5), use_container_width=True)

st.divider()
st.caption("Semua data disimpan secara lokal di folder `/data` dan dipertahankan dalam session.")
