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
- Treat visible foods, drinks, fruit, vegetables, raw ingredients, single ingredients, snacks, sauces,
  pickles, salads, and packaged edible items as valid food candidates.
- Always populate foods with the best visible raw food-name candidates when any edible item is visible.
- If the exact dish name is uncertain, return a broader visible candidate with lower confidence instead of
  returning an empty list. Examples: "파프리카", "자몽", "석류", "피칸", "녹색피망", "블루베리",
  "핫초코", "홍차", "카페라떼", "샐러드", "피클", "소스".
- Low confidence should be represented by a lower confidence value, not by foods=[].
- Return foods=[] only when no food, drink, edible ingredient, condiment, or packaged edible item is visible.
- Do not invent food that is not visible.
- Do not provide medical advice.
"""


def build_food_eval_prompt(allowed_foods: list[str] | None = None) -> str:
    if not allowed_foods:
        return FOOD_EVAL_PROMPT
    allowed_food_lines = "\n".join(f"- {food_name}" for food_name in allowed_foods)
    return f"""{FOOD_EVAL_PROMPT}

Allowed-food candidate mode:
- allowed_foods is a helpful candidate list, not a hard answer constraint.
- First identify visible raw food names from the image. Do not start by rejecting the image because an exact
  allowed label is hard to choose.
- If an allowed_foods label clearly matches the visible item, you may return that label exactly.
- If no allowed label clearly matches, still return the best visible raw food name such as a fruit, vegetable,
  ingredient, drink, sauce, pickle, salad, or broad dish type.
- Never return foods=[] solely because the visible item is absent from allowed_foods.
- Use lower confidence for uncertain or generic names, but keep the candidate so downstream nutrition lookup
  has a search query.
- Return foods=[] only when no food, drink, edible ingredient, condiment, or packaged edible item is visible at all.

allowed_foods:
{allowed_food_lines}
"""
