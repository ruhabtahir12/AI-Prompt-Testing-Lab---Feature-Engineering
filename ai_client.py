
import streamlit as st
import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
client = Groq(api_key=api_key)

AVAILABLE_MODELS = {
    "Llama 3.3 70B (Balanced)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Fast)": "llama-3.1-8b-instant",
    "GPT-OSS 20B (Efficient)": "openai/gpt-oss-20b",
}

def get_ai_response(prompt_text: str, model: str = "llama-3.3-70b-versatile") -> dict:
    """Sends a prompt to Groq and returns response text + performance metadata."""
    if not prompt_text or not prompt_text.strip():
        raise ValueError("Prompt cannot be empty.")

    try:
        start_time = time.time()
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_text}],
            model=model,
            temperature=0.7,
            timeout=30,
        )
        elapsed = round(time.time() - start_time, 2)
        usage = chat_completion.usage

        return {
            "text": chat_completion.choices[0].message.content,
            "latency_seconds": elapsed,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    except Exception as e:
        raise RuntimeError(f"AI request failed: {str(e)}")