from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from statistics import mean, stdev

from database import init_db, get_db, Test, Prompt, Response
from ai_client import get_ai_response
from scoring import score_response, suggest_improved_prompt

app = FastAPI(title="AI Prompt Testing Lab")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

MODEL_PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
    cost = (input_tokens / 1_000_000 * pricing["input"]) + (output_tokens / 1_000_000 * pricing["output"])
    return round(cost, 6)


class RunTestRequest(BaseModel):
    task_description: str
    prompts: List[str]
    model: Optional[str] = "llama-3.3-70b-versatile"
    num_runs: Optional[int] = 1


class ImprovePromptRequest(BaseModel):
    task_description: str
    prompt: str
    response: str
    scores: dict
    model: Optional[str] = "llama-3.3-70b-versatile"


@app.post("/run-test")
def run_test(payload: RunTestRequest, db: Session = Depends(get_db)):
    if not payload.task_description.strip():
        raise HTTPException(status_code=400, detail="Task description cannot be empty.")

    clean_prompts = [p.strip() for p in payload.prompts if p.strip()]
    if len(clean_prompts) < 2:
        raise HTTPException(status_code=400, detail="At least 2 non-empty prompts are required.")

    num_runs = max(1, min(payload.num_runs, 5))

    test = Test(task_description=payload.task_description)
    db.add(test)
    db.commit()
    db.refresh(test)

    results = []
    errors = []

    score_keys = ["accuracy", "relevance", "completeness", "clarity",
                  "creativity", "conciseness", "instruction_following", "overall_score"]

    for prompt_text in clean_prompts:
        run_results = []
        run_errors = []

        for run_num in range(num_runs):
            try:
                ai_result = get_ai_response(prompt_text, model=payload.model)
                ai_response_text = ai_result["text"]

                scores = score_response(payload.task_description, prompt_text, ai_response_text, model=payload.model)

                cost = calculate_cost(payload.model, ai_result["input_tokens"], ai_result["output_tokens"])

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
            errors.append({"prompt": prompt_text, "error": "; ".join(run_errors) or "All runs failed."})
            continue

        avg_scores = {}
        std_scores = {}
        for key in score_keys:
            values = [r["scores"][key] for r in run_results]
            avg_scores[key] = round(mean(values), 2)
            std_scores[key] = round(stdev(values), 2) if len(values) > 1 else 0.0

        avg_latency = round(mean([r["latency_seconds"] for r in run_results]), 2)
        avg_tokens = round(mean([r["total_tokens"] for r in run_results]), 0)
        avg_cost = round(mean([r["cost_usd"] for r in run_results]), 6)

        representative_response = run_results[0]["response"]

        prompt_row = Prompt(test_id=test.id, prompt_text=prompt_text)
        db.add(prompt_row)
        db.commit()
        db.refresh(prompt_row)

        response_row = Response(
            prompt_id=prompt_row.id,
            response_text=representative_response,
            accuracy_score=avg_scores["accuracy"],
            relevance_score=avg_scores["relevance"],
            completeness_score=avg_scores["completeness"],
            clarity_score=avg_scores["clarity"],
            creativity_score=avg_scores["creativity"],
            conciseness_score=avg_scores["conciseness"],
            instruction_following_score=avg_scores["instruction_following"],
            overall_score=avg_scores["overall_score"],
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
        raise HTTPException(status_code=502, detail="All prompts failed. Check your AI provider connection.")

    best = max(results, key=lambda r: r["scores"]["overall_score"])

    return {
        "test_id": test.id,
        "task": payload.task_description,
        "results": results,
        "errors": errors,
        "best_prompt": best["prompt"],
        "num_runs": num_runs,
    }


@app.post("/improve-prompt")
def improve_prompt(payload: ImprovePromptRequest):
    try:
        improved = suggest_improved_prompt(
            payload.task_description,
            payload.prompt,
            payload.response,
            payload.scores,
            model=payload.model,
        )
        return {"improved_prompt": improved}
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/tests")
def get_all_tests(db: Session = Depends(get_db)):
    tests = db.query(Test).order_by(Test.created_at.desc()).all()
    return [{"id": t.id, "task_description": t.task_description, "created_at": t.created_at} for t in tests]


@app.get("/tests/{test_id}")
def get_test(test_id: int, db: Session = Depends(get_db)):
    test = db.query(Test).filter(Test.id == test_id).first()
    if not test:
        raise HTTPException(status_code=404, detail="Test not found.")

    results = []
    for prompt in test.prompts:
        r = prompt.response
        results.append({
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

    return {"test_id": test.id, "task": test.task_description, "results": results}