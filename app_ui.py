

import streamlit as st
import pandas as pd
from statistics import mean, stdev

from ai_client import get_ai_response
from scoring import score_response, suggest_improved_prompt
from database import init_db, get_db, Test, Prompt, Response

MODELS = {
    "Llama 3.3 70B (Balanced)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Fast)": "llama-3.1-8b-instant",
    "GPT-OSS 20B (Efficient)": "openai/gpt-oss-20b",
}

MODEL_PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
    cost = (input_tokens / 1_000_000 * pricing["input"]) + (output_tokens / 1_000_000 * pricing["output"])
    return round(cost, 6)


init_db()

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


def run_all_prompts(task_description: str, clean_prompts: list, model: str, runs_per_prompt: int, db):
    """Runs each prompt `runs_per_prompt` times, averages scores/cost/latency, saves to DB."""
    test = Test(task_description=task_description)
    db.add(test)
    db.commit()
    db.refresh(test)

    results = []
    errors = []

    for prompt_text in clean_prompts:
        run_results = []
        run_errors = []

        for _ in range(runs_per_prompt):
            try:
                ai_result = get_ai_response(prompt_text, model=model)
                ai_response_text = ai_result["text"]
                scores = score_response(task_description, prompt_text, ai_response_text, model=model)
                cost = calculate_cost(model, ai_result["input_tokens"], ai_result["output_tokens"])

                run_results.append({
                    "response": ai_response_text,
                    "scores": scores,
                    "latency_seconds": ai_result["latency_seconds"],
                    "total_tokens": ai_result["total_tokens"],
                    "cost_usd": cost,
                })
            except (RuntimeError, ValueError) as e:
                run_errors.append(str(e))

        if not run_results:
            errors.append({"prompt": prompt_text, "error": "; ".join(run_errors)})
            continue

        score_keys = run_results[0]["scores"].keys()
        avg_scores = {k: round(mean([r["scores"][k] for r in run_results]), 2) for k in score_keys}
        std_scores = {
            k: round(stdev([r["scores"][k] for r in run_results]), 2) if len(run_results) > 1 else 0
            for k in score_keys
        }
        representative_response = run_results[0]["response"]

        avg_latency = round(mean([r["latency_seconds"] for r in run_results]), 2)
        avg_tokens = round(mean([r["total_tokens"] for r in run_results]), 0)
        avg_cost = round(mean([r["cost_usd"] for r in run_results]), 6)

        prompt_row = Prompt(test_id=test.id, prompt_text=prompt_text)
        db.add(prompt_row)
        db.commit()
        db.refresh(prompt_row)

        response_row = Response(
            prompt_id=prompt_row.id,
            response_text=representative_response,
            accuracy_score=avg_scores.get("accuracy"),
            relevance_score=avg_scores.get("relevance"),
            completeness_score=avg_scores.get("completeness"),
            clarity_score=avg_scores.get("clarity"),
            creativity_score=avg_scores.get("creativity"),
            conciseness_score=avg_scores.get("conciseness"),
            instruction_following_score=avg_scores.get("instruction_following"),
            overall_score=avg_scores.get("overall_score"),
        )
        db.add(response_row)
        db.commit()

        results.append({
            "prompt": prompt_text,
            "response": representative_response,
            "scores": avg_scores,
            "score_variance": std_scores,
            "num_runs": len(run_results),
            "avg_latency_seconds": avg_latency,
            "avg_tokens": avg_tokens,
            "avg_cost_usd": avg_cost,
        })

    if not results:
        return None, errors

    best = max(results, key=lambda r: r["scores"]["overall_score"])
    return {
        "test_id": test.id,
        "task": task_description,
        "results": results,
        "errors": errors,
        "best_prompt": best["prompt"],
    }, errors


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
        unique_prompts = list(dict.fromkeys(filled_prompts))

        if not task.strip():
            st.error("Please enter a task description.")
        elif len(unique_prompts) < 2:
            st.error("Please enter at least 2 different, non-empty prompts.")
        else:
            if len(unique_prompts) < len(filled_prompts):
                st.warning("Duplicate prompts were removed before running the test.")

            with st.spinner(f"Running {len(unique_prompts)} prompt(s) x {num_runs} run(s) through {selected_model_label}..."):
                db = next(get_db())
                try:
                    result, run_errors = run_all_prompts(task, unique_prompts, selected_model, num_runs, db)
                    if result is None:
                        st.error("All prompts failed. Check your AI provider connection.")
                    else:
                        st.session_state.results = result
                        st.session_state.last_task = task
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                finally:
                    db.close()

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

                if "avg_latency_seconds" in r:
                    st.divider()
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    perf_col1.metric("⏱️ Latency", f"{r.get('avg_latency_seconds', 0)}s")
                    perf_col2.metric("🔢 Tokens", f"{int(r.get('avg_tokens', 0))}")
                    perf_col3.metric("💰 Cost", f"${r.get('avg_cost_usd', 0):.6f}")

                if st.button("✨ Suggest Improvement", key=f"improve_{i}"):
                    with st.spinner("Analyzing weak points and generating a better prompt..."):
                        try:
                            improved_text = suggest_improved_prompt(
                                task if task.strip() else st.session_state.get("last_task", ""),
                                r["prompt"],
                                r["response"],
                                r["scores"],
                                model=selected_model,
                            )
                            st.info(f"**Suggested improved prompt:**\n\n{improved_text}")
                        except RuntimeError as e:
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

    db = next(get_db())
    try:
        tests_db = db.query(Test).order_by(Test.created_at.desc()).all()
        tests = [{"id": t.id, "task_description": t.task_description, "created_at": str(t.created_at)} for t in tests_db]
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
                        test_obj = db.query(Test).filter(Test.id == t["id"]).first()
                        detail_results = []
                        for prompt in test_obj.prompts:
                            r = prompt.response
                            detail_results.append({
                                "prompt": prompt.prompt_text,
                                "response": r.response_text if r else None,
                                "scores": {
                                    "accuracy": r.accuracy_score,
                                    "relevance": r.relevance_score,
                                    "completeness": r.completeness_score,
                                    "clarity": r.clarity_score,
                                    "creativity": r.creativity_score,
                                    "conciseness": r.conciseness_score,
                                    "instruction_following": r.instruction_following_score,
                                    "overall_score": r.overall_score,
                                } if r else None,
                            })
                        detail = {"task": test_obj.task_description, "results": detail_results}
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
    db.close()