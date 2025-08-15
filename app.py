#building AI based Text Summarizers 

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

st.title("AI Based Text Summarizer")

@st.cache_resource
def load_model():
    try:
        model_name = "google/pegasus-xsum"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Move model to appropriate device
        model = model.to(device)
        
        return pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
            framework="pt"
        )
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

summarizer = load_model()

text = st.text_area("Enter your article:", height=300, max_chars=10240)

if st.button("Summarize"):
    if not text.strip():
        st.warning("Please enter text to summarize")
    elif summarizer is None:
        st.error("Model failed to load. Check installation!")
    else:
        with st.spinner("Generating summary..."):
            try:
                summary = summarizer(
                    text,
                    max_length=250,
                    min_length=50,
                    do_sample=False,
                    truncation=True,
                    no_repeat_ngram_size=3
                )
                st.subheader("Summary")
                st.success(summary[0]['summary_text'])
            except Exception as e:
                st.error(f"Summarization failed: {str(e)}")