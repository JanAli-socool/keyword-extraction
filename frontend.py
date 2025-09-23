import streamlit as st
import tempfile
import os
import pandas as pd
from pathlib import Path
from bert_final import run_pipeline

st.set_page_config(page_title="SEC Filings NLP Pipeline - peak-tech", layout="wide")
st.title("SEC Filings NLP Pipeline - peak-tech")

# File uploader (multiple HTML files)
uploaded_files = st.file_uploader(
    "Upload your HTML filings", type=["html"], accept_multiple_files=True
)

# Set output file path (temporary)
out_xlsx = os.path.join(tempfile.gettempdir(), "pipeline_output.xlsx")

# Options
limit = st.number_input("Max number of files", min_value=1, max_value=50, value=5, step=1)
use_bertopic = st.checkbox("Use BERTopic (fallback to LDA if not available)", value=False)

if st.button("Run Pipeline"):
    if not uploaded_files:
        st.error("Please upload at least one HTML file before running the pipeline.")
    else:
        try:
            with st.spinner("Starting pipeline..."):
                progress_text = st.empty()

                # Create temporary input directory
                tmp_dir = tempfile.mkdtemp()
                for uploaded in uploaded_files[:limit]:
                    file_path = Path(tmp_dir) / uploaded.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded.getbuffer())

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
                    run_pipeline(tmp_dir, out_xlsx, limit=limit, use_bertopic=use_bertopic, stage=i)

            st.success("Pipeline completed!")

            # Preview results inside app
            if os.path.exists(out_xlsx):
                try:
                    df = pd.read_excel(out_xlsx, sheet_name="summary")
                    st.subheader("Preview of Results (Summary Sheet)")
                    st.dataframe(df.head(50))
                except Exception as e:
                    st.warning(f"Could not preview Excel file: {e}")

            # Download button
            with open(out_xlsx, "rb") as f:
                st.download_button(
                    "Download Excel Output",
                    f,
                    file_name="pipeline_output.xlsx"
                )
        except Exception as e:
            st.error(f"Error: {e}")
