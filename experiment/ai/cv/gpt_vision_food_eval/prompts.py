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
