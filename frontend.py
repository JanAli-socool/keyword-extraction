import streamlit as st
import tempfile
import os
import pandas as pd
from bert_final import run_pipeline

st.title("SEC Filings NLP Pipeline")

# Select source directory
src_dir = st.text_input("Enter source directory (with .txt filings):", value="data")

# Set output file path (in temp dir for safety)
out_xlsx = os.path.join(tempfile.gettempdir(), "pipeline_output.xlsx")

# Options
limit = st.number_input("Max number of files", min_value=1, max_value=50, value=5, step=1)
use_bertopic = st.checkbox("Use BERTopic (fallback to LDA if not available)", value=False)

if st.button("Run Pipeline"):
    try:
        with st.spinner("Processing files..."):
            run_pipeline(src_dir, out_xlsx, limit=limit, use_bertopic=use_bertopic)
        st.success("Pipeline completed!")

        # Preview results inside app
        if os.path.exists(out_xlsx):
            try:
                df = pd.read_excel(out_xlsx)
                st.subheader("Preview of Results")
                st.dataframe(df.head(50))  # show first 50 rows
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
