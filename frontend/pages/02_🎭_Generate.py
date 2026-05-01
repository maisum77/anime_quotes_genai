"""
Generate page — Create anime quotes with the fine-tuned GPT-2 model.

Simple, focused interface to demonstrate ML model generation
and AWS Lambda pipeline integration.
"""

import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.aws_client import submit_generation, submit_batch_generation, get_job_status
from utils.visualization import render_quote_card, status_badge

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Generate", page_icon="🎭", layout="wide")

st.title("🎭 Generate Anime Quotes")
st.markdown("Generate anime-style speech using the fine-tuned GPT-2 model deployed on AWS Lambda.")
st.divider()

# ---------------------------------------------------------------------------
# Single Generation
# ---------------------------------------------------------------------------

st.markdown("### ✨ Single Generation")

with st.form("generation_form"):
    col1, col2 = st.columns(2)

    with col1:
        speech_type = st.selectbox(
            "Speech Type",
            ["motivational", "villain", "philosophical", "heroic", "emotional", "comedic"],
            help="Type of anime speech to generate",
        )
        generation_type = st.selectbox(
            "Generation Type",
            ["quote", "dialogue", "monologue"],
            help="Format of the generated content",
        )

    with col2:
        temperature = st.slider(
            "Temperature",
            min_value=0.3,
            max_value=1.5,
            value=0.8,
            step=0.1,
            help="Higher = more creative, Lower = more focused",
        )
        max_length = st.slider(
            "Max Length",
            min_value=50,
            max_value=500,
            value=200,
            step=25,
            help="Maximum token length of generated text",
        )

    custom_prompt = st.text_area(
        "Custom Prompt (optional)",
        placeholder="E.g., A hero facing their final battle...",
        help="Add context to guide the generation",
    )

    characters = st.text_input(
        "Characters (for dialogue, comma-separated)",
        placeholder="E.g., Hero, Rival",
    )

    submitted = st.form_submit_button("🎭 Generate Quote", type="primary", use_container_width=True)

if submitted:
    with st.spinner("🚀 Submitting to AWS Lambda pipeline..."):
        result = submit_generation(
            speech_type=speech_type,
            generation_type=generation_type,
            custom_prompt=custom_prompt,
            characters=[c.strip() for c in characters.split(",")] if characters else [],
            temperature=temperature,
            max_length=max_length,
        )

    if "error" in result:
        st.error(f"❌ Generation failed: {result.get('detail', result.get('error'))}")
    else:
        job_id = result.get("job_id", "")
        st.success(f"✅ Job submitted! Job ID: `{job_id}`")

        # Track in session
        if "recent_jobs" not in st.session_state:
            st.session_state["recent_jobs"] = []
        st.session_state["recent_jobs"].append({
            "job_id": job_id,
            "status": "pending",
            "speech_type": speech_type,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Poll for result
        status_placeholder = st.empty()
        progress = st.progress(0)

        for i in range(20):
            time.sleep(1)
            progress.progress((i + 1) / 20)

            job = get_job_status(job_id)
            current_status = job.get("status", "pending")

            with status_placeholder.container():
                st.markdown(status_badge(current_status), unsafe_allow_html=True)

            if current_status == "completed":
                result_data = job.get("result", {})
                st.balloons()
                st.markdown("#### 🎉 Generated Quote")
                render_quote_card(
                    quote_text=result_data.get("text", result_data.get("quote", "Generation complete!")),
                    speech_type=result_data.get("speech_type", speech_type),
                    character=result_data.get("character", ""),
                    source=result_data.get("source", "gpt2"),
                    created_at=result_data.get("created_at", ""),
                )
                break
            elif current_status == "failed":
                st.error(f"❌ Generation failed: {job.get('error', 'Unknown error')}")
                break

st.divider()

# ---------------------------------------------------------------------------
# Batch Generation
# ---------------------------------------------------------------------------

st.markdown("### 📦 Batch Generation")
st.caption("Generate multiple quotes at once across different speech types.")

with st.expander("Batch Generation Settings", expanded=False):
    with st.form("batch_form"):
        batch_speech_types = st.multiselect(
            "Speech Types",
            ["motivational", "villain", "philosophical", "heroic", "emotional", "comedic"],
            default=["motivational", "villain"],
        )
        batch_count = st.number_input(
            "Quotes per type",
            min_value=1,
            max_value=20,
            value=3,
        )
        batch_temp = st.slider(
            "Temperature",
            min_value=0.3,
            max_value=1.5,
            value=0.8,
            step=0.1,
            key="batch_temp",
        )

        batch_submitted = st.form_submit_button("📦 Start Batch Generation", use_container_width=True)

    if batch_submitted and batch_speech_types:
        with st.spinner(f"Submitting batch: {len(batch_speech_types)} types × {batch_count} quotes..."):
            batch_result = submit_batch_generation(
                speech_types=batch_speech_types,
                count=batch_count,
                temperature=batch_temp,
            )
        if "error" in batch_result:
            st.error(f"Batch failed: {batch_result.get('detail')}")
        else:
            st.success(
                f"✅ Batch submitted! {batch_result.get('total_jobs', 0)} jobs queued. "
                f"Batch ID: `{batch_result.get('batch_id', '')}`"
            )

# ---------------------------------------------------------------------------
# Model Info
# ---------------------------------------------------------------------------

st.divider()
st.markdown("### 🧠 Model Information")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown(
        """
        **Fine-tuned GPT-2 Model**
        - **Base Model**: GPT-2 (124M parameters)
        - **Training Data**: Anime quotes dataset
        - **Fine-tuning**: Custom dataset with anime speech patterns
        - **Deployment**: AWS Lambda with PyTorch runtime
        """
    )

with info_col2:
    st.markdown(
        """
        **Generation Pipeline**
        1. 📥 **Preprocessing** — Input validation & sanitization
        2. 🧠 **Generation** — Gemini → GPT-2 → Fallback
        3. 📤 **Postprocessing** — Formatting & S3 storage
        4. 📊 **Tracking** — DynamoDB job metadata
        """
    )
