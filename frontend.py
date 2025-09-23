import streamlit as st
import tempfile
import os
import pandas as pd
from pathlib import Path
from bert_final import run_pipeline

st.set_page_config(page_title="SEC Filings NLP Pipeline - peak-tech", layout="wide")
st.title("SEC Filings NLP Pipeline - peak-tech")

# --- File uploader allowing any type ---
uploaded_files = st.file_uploader(
    "Upload your HTML filings (any file type allowed)",
    type=None,  # allow any type
    accept_multiple_files=True
)

# --- Output path ---
out_xlsx = os.path.join(tempfile.gettempdir(), "pipeline_output.xlsx")

# --- Options ---
limit = st.number_input("Max number of files", min_value=1, max_value=50, value=5, step=1)
use_bertopic = st.checkbox("Use BERTopic (fallback to LDA if not available)", value=False)

if st.button("Run Pipeline"):
    if not uploaded_files:
        st.error("Please upload at least one file before running the pipeline.")
    else:
        try:
            with st.spinner("Starting pipeline..."):
                progress_text = st.empty()

                # Create temporary directory
                tmp_dir = tempfile.mkdtemp()
                saved_files = 0

                for uploaded in uploaded_files:
                    # Only save .html files, skip others
                    if uploaded.name.lower().endswith(".html"):
                        file_path = Path(tmp_dir) / uploaded.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded.getbuffer())
                        saved_files += 1
                        if saved_files >= limit:
                            break

                if saved_files == 0:
                    st.error("No valid HTML files found. Please upload files ending with .html.")
                else:
                    # Progress stages
                    stages = [
                        "Preprocessing files...",
                        "Filtering data...",
                        "Extracting sections...",
                        "Parsing filings...",
                        "Post-processing and saving output..."
                    ]

                    for i, stage in enumerate(stages, 1):
                        progress_text.text(f"Step {i}/{len(stages)}: {stage}")
                        run_pipeline(tmp_dir, out_xlsx, limit=limit, use_bertopic=use_bertopic)

                    st.success("Pipeline completed!")

                    # --- Preview results ---
                    if os.path.exists(out_xlsx):
                        try:
                            df = pd.read_excel(out_xlsx, sheet_name="summary")
                            st.subheader("Preview of Results (Summary Sheet)")
                            st.dataframe(df.head(50))
                        except Exception as e:
                            st.warning(f"Could not preview Excel file: {e}")

                    # --- Download button ---
                    with open(out_xlsx, "rb") as f:
                        st.download_button(
                            "Download Excel Output",
                            f,
                            file_name="pipeline_output.xlsx"
                        )
        except Exception as e:
            st.error(f"Error: {e}")
