




import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

MODELS = {
    "Llama 3.3 70B (Balanced)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Fast)": "llama-3.1-8b-instant",
    "GPT-OSS 120B (Strong reasoning)": "openai/gpt-oss-120b",
    "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.11, "output": 0.34},
}

st.set_page_config(page_title="AI Prompt Testing Lab", layout="wide", page_icon="🧪")

st.markdown("""
<style>
.stApp { background-color: #000000; }
[data-testid="stAppViewContainer"] { background-color: #000000; }
[data-testid="stHeader"] { background-color: #000000; }
.card {
    border-radius: 14px;
    padding: 18px;
    background: #111111;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    margin-bottom: 10px;
    color: #FAFAFA;
}
.winner-card { border: 2px solid #FF4B4B; }
.prompt-text { font-style: italic; color: #AAAAAA; font-size: 13px; }
.response-text { font-size: 14px; margin-top: 8px; color: #FAFAFA; }
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
page = st.sidebar.radio("Navigate", ["New Test", "History"])
st.sidebar.divider()
selected_model_label = st.sidebar.selectbox("AI Model", list(MODELS.keys()))
selected_model = MODELS[selected_model_label]
num_runs = st.sidebar.slider(
    "Runs per prompt",
    min_value=1, max_value=5, value=1,
    help="Run each prompt multiple times and average the scores for more reliable results.",
)

# =========================================================
# PAGE 1: NEW TEST
# =========================================================
if page == "New Test":
    st.title("🧪 AI Prompt Testing Lab")
    st.caption("Compare multiple prompts and find the best one.")

    task = st.text_area(
        "Task Description",
        placeholder="e.g. Write a one-sentence product description for a reusable water bottle.",
    )

    st.subheader("Prompts to Compare")

    if "num_prompts" not in st.session_state:
        st.session_state.num_prompts = 2

    prompts = []
    for i in range(st.session_state.num_prompts):
        p = st.text_area(f"Prompt {i + 1}", key=f"prompt_{i}")
        prompts.append(p)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ Add Another Prompt") and st.session_state.num_prompts < 5:
            st.session_state.num_prompts += 1
            st.rerun()
    with col2:
        if st.button("➖ Remove Last Prompt") and st.session_state.num_prompts > 2:
            st.session_state.num_prompts -= 1
            st.rerun()

    st.divider()

    if st.button("🚀 Run Test", type="primary"):
        filled_prompts = [p for p in prompts if p.strip() != ""]
        unique_prompts = list(dict.fromkeys(filled_prompts))  # remove exact duplicates

        if not task.strip():
            st.error("Please enter a task description.")
        elif len(unique_prompts) < 2:
            st.error("Please enter at least 2 different, non-empty prompts.")
        else:
            if len(unique_prompts) < len(filled_prompts):
                st.warning("Duplicate prompts were removed before running the test.")

            with st.spinner(f"Running {len(unique_prompts)} prompt(s) x {num_runs} run(s) through {selected_model_label}..."):
                try:
                    response = requests.post(
                        f"{API_URL}/run-test",
                        json={
                            "task_description": task,
                            "prompts": unique_prompts,
                            "model": selected_model,
                            "num_runs": num_runs,
                        },
                        timeout=90 * num_runs,
                    )
                    if response.status_code >= 400:
                        st.error(f"Server error: {response.json().get('detail', 'Unknown error')}")
                    else:
                        st.session_state.results = response.json()
                        st.session_state.last_task = task
                except requests.exceptions.ConnectionError:
                    st.error("Could not reach the backend. Make sure `uvicorn main:app --reload` is running.")
                except requests.exceptions.Timeout:
                    st.error("The request timed out. The AI model may be slow — try again.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    if "results" in st.session_state:
        results = st.session_state.results

        if results.get("errors"):
            for err in results["errors"]:
                st.warning(f"⚠️ Prompt failed: \"{err['prompt'][:50]}...\" — {err['error']}")

        st.success(f"🏆 Best Prompt: **{results['best_prompt']}**")

        cols = st.columns(len(results["results"]))
        for i, r in enumerate(results["results"]):
            with cols[i]:
                is_winner = r["prompt"] == results["best_prompt"]
                card_class = "card winner-card" if is_winner else "card"
                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <h4>Prompt {i + 1} {"🏆" if is_winner else ""}</h4>
                        <p class="prompt-text">{r['prompt']}</p>
                        <p class="response-text">{r['response']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                for key, value in r["scores"].items():
                    variance = r.get("score_variance", {}).get(key, 0)
                    variance_note = f" (±{variance})" if variance > 0 else ""
                    st.write(f"- {key.replace('_', ' ').title()}: **{value}**{variance_note}")

                if r.get("num_runs", 1) > 1:
                    st.caption(f"Averaged across {r['num_runs']} runs")

                # ---- Cost & latency metrics ----
                if "avg_latency_seconds" in r:
                    st.divider()
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    perf_col1.metric("⏱️ Latency", f"{r.get('avg_latency_seconds', 0)}s")
                    perf_col2.metric("🔢 Tokens", f"{int(r.get('avg_tokens', 0))}")
                    perf_col3.metric("💰 Cost", f"${r.get('avg_cost_usd', 0):.6f}")

                # ---- AI-assisted prompt improvement ----
                if st.button("✨ Suggest Improvement", key=f"improve_{i}"):
                    with st.spinner("Analyzing weak points and generating a better prompt..."):
                        try:
                            improve_response = requests.post(
                                f"{API_URL}/improve-prompt",
                                json={
                                    "task_description": task if task.strip() else st.session_state.get("last_task", ""),
                                    "prompt": r["prompt"],
                                    "response": r["response"],
                                    "scores": r["scores"],
                                    "model": selected_model,
                                },
                                timeout=30,
                            )
                            if improve_response.status_code >= 400:
                                st.error(f"Server error: {improve_response.json().get('detail', 'Unknown error')}")
                            else:
                                improved_text = improve_response.json()["improved_prompt"]
                                st.info(f"**Suggested improved prompt:**\n\n{improved_text}")
                        except Exception as e:
                            st.error(f"Could not generate suggestion: {e}")

        st.divider()
        st.subheader("📊 Score Comparison")

        chart_data = pd.DataFrame({
            f"Prompt {i+1}": r["scores"]
            for i, r in enumerate(results["results"])
        }).drop("overall_score", errors="ignore")
        st.bar_chart(chart_data)

        overall_data = pd.DataFrame({
            "Prompt": [f"Prompt {i+1}" for i in range(len(results["results"]))],
            "Overall Score": [r["scores"]["overall_score"] for r in results["results"]],
        }).set_index("Prompt")
        st.bar_chart(overall_data)

# =========================================================
# PAGE 2: HISTORY
# =========================================================
elif page == "History":
    st.title("📜 Test History")
    st.caption("Browse and reopen previous prompt tests.")

    try:
        response = requests.get(f"{API_URL}/tests", timeout=10)
        response.raise_for_status()
        tests = response.json()
    except requests.exceptions.ConnectionError:
        st.error("Could not reach the backend. Make sure `uvicorn main:app --reload` is running.")
        tests = []
    except Exception as e:
        st.error(f"Could not load history: {e}")
        tests = []

    if not tests:
        st.info("No tests yet. Run your first test from the 'New Test' page!")
    else:
        for t in tests:
            with st.expander(f"🧪 {t['task_description'][:70]}  —  {t['created_at'][:10]}"):
                if st.button("View Full Results", key=f"view_{t['id']}"):
                    try:
                        detail_response = requests.get(f"{API_URL}/tests/{t['id']}", timeout=10)
                        detail = detail_response.json()
                    except Exception as e:
                        st.error(f"Could not load test details: {e}")
                        continue

                    st.write(f"**Task:** {detail['task']}")
                    valid_results = [r for r in detail["results"] if r["scores"]]
                    if not valid_results:
                        st.warning("No scored results for this test.")
                        continue

                    cols = st.columns(len(detail["results"]))
                    best_score = max(r["scores"]["overall_score"] for r in valid_results)

                    for i, r in enumerate(detail["results"]):
                        with cols[i]:
                            is_winner = r["scores"] and r["scores"]["overall_score"] == best_score
                            card_class = "card winner-card" if is_winner else "card"
                            st.markdown(
                                f"""
                                <div class="{card_class}">
                                    <h5>Prompt {i + 1} {"🏆" if is_winner else ""}</h5>
                                    <p class="prompt-text">{r['prompt']}</p>
                                    <p class="response-text">{r['response']}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            if r["scores"]:
                                for key, value in r["scores"].items():
                                    st.write(f"- {key.replace('_', ' ').title()}: **{value}**")