from __future__ import annotations

FOOD_EVAL_PROMPT = """
You are evaluating food image recognition for a Korean healthcare app.
Return JSON only. Do not use Markdown.

Schema:
{
  "foods": [
    {
      "name": "specific food name",
      "confidence": 0.0
    }
  ],
  "analysis_status": "success|low_confidence|failed",
  "fail_reason": null
}

Rules:
- Prefer specific Korean food names when possible.
- Include cooking method or dish type when visually clear.
- If the image contains no food, return foods=[] and analysis_status="failed".
- Do not provide medical advice.
"""


def build_food_eval_prompt(allowed_foods: list[str] | None = None) -> str:
    if not allowed_foods:
        return FOOD_EVAL_PROMPT
    allowed_food_lines = "\n".join(f"- {food_name}" for food_name in allowed_foods)
    return f"""{FOOD_EVAL_PROMPT}

Constrained label mode:
- You must choose food names only from the allowed_foods list below.
- Return each food name exactly as it appears in the list.
- If no allowed label matches the image, return foods=[] and analysis_status="low_confidence".

allowed_foods:
{allowed_food_lines}
"""
