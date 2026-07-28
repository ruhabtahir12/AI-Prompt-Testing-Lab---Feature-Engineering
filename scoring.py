


import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def score_response(task: str, prompt_text: str, response_text: str, model: str = "llama-3.3-70b-versatile") -> dict:
    scoring_prompt = f"""You are an expert evaluator judging how well an AI response fulfills a task.

Task: {task}
Prompt used: {prompt_text}
AI Response: {response_text}

Score the response from 1 to 10 on each of these criteria:
- accuracy
- relevance
- completeness
- clarity
- creativity
- conciseness
- instruction_following

Return ONLY valid JSON in this exact format, nothing else, no explanation:
{{
  "accuracy": <number>,
  "relevance": <number>,
  "completeness": <number>,
  "clarity": <number>,
  "creativity": <number>,
  "conciseness": <number>,
  "instruction_following": <number>
}}"""

    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": scoring_prompt}],
            model=model,
            temperature=0.2,
            timeout=30,
        )
        raw_output = completion.choices[0].message.content.strip()

        if raw_output.startswith("```"):
            raw_output = raw_output.strip("`")
            raw_output = raw_output.replace("json", "", 1).strip()

        scores = json.loads(raw_output)

        required_keys = ["accuracy", "relevance", "completeness", "clarity",
                          "creativity", "conciseness", "instruction_following"]
        for key in required_keys:
            if key not in scores:
                scores[key] = 5  # fallback default if AI missed a field

        overall = sum(scores[k] for k in required_keys) / len(required_keys)
        scores["overall_score"] = round(overall, 2)
        return scores

    except json.JSONDecodeError:
        raise RuntimeError("Scoring failed: AI did not return valid JSON.")
    except Exception as e:
        raise RuntimeError(f"Scoring failed: {str(e)}")


def suggest_improved_prompt(task: str, weak_prompt: str, weak_response: str, scores: dict, model: str = "llama-3.3-70b-versatile") -> str:
    """Asks the AI to analyze a low-scoring prompt and suggest a better version."""

    weak_areas = sorted(
        [(k, v) for k, v in scores.items() if k != "overall_score"],
        key=lambda x: x[1]
    )[:3]  # find the 3 weakest scoring criteria
    weak_areas_text = ", ".join([f"{k} ({v}/10)" for k, v in weak_areas])

    improvement_prompt = f"""You are an expert prompt engineer. Analyze this prompt and suggest a better version.

Task: {task}
Original Prompt: {weak_prompt}
Response it produced: {weak_response}
Weakest scoring areas: {weak_areas_text}

Write ONE improved version of the prompt that would likely produce a better response,
specifically targeting the weak areas above. Return ONLY the improved prompt text,
nothing else — no explanation, no quotes, no preamble."""

    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": improvement_prompt}],
            model=model,
            temperature=0.5,
            timeout=30,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Prompt improvement failed: {str(e)}")