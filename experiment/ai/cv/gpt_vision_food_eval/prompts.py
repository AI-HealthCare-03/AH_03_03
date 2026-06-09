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
- Return visible fruit, beverages, ingredients, single raw ingredients, snacks, and packaged food as food candidates.
- If the exact food is uncertain, return the most plausible visible food name with lower confidence instead of returning an empty list.
- Return foods=[] only when no food, drink, or edible ingredient is visible.
- Do not provide medical advice.
"""


def build_food_eval_prompt(allowed_foods: list[str] | None = None) -> str:
    if not allowed_foods:
        return FOOD_EVAL_PROMPT
    allowed_food_lines = "\n".join(f"- {food_name}" for food_name in allowed_foods)
    return f"""{FOOD_EVAL_PROMPT}

Allowed-food candidate mode:
- The allowed_foods list is a candidate label list, not a hard restriction.
- First identify the visible food, drink, fruit, ingredient, or dish as directly as possible.
- If an allowed_foods label clearly matches the image, return that label exactly.
- If no allowed label clearly matches, still return the best raw food name you can infer.
- Use lower confidence for uncertain or generic names, but do not return foods=[] just because the label is absent from allowed_foods.
- Return foods=[] only when no food, drink, or edible ingredient is visible at all.

allowed_foods:
{allowed_food_lines}
"""
