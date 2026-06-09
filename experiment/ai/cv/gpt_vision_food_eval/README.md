# GPT Vision Food Evaluation

This experiment evaluates GPT Vision food-name extraction without changing production runtime code.

## Input

`sample_labels.csv` format:

```csv
image_path,expected_foods
samples/rice.jpg,쌀밥
samples/rice_cake.jpg,가래떡
```

`expected_foods` supports comma-separated or pipe-separated food names.

## Run

```bash
uv run python experiment/ai/cv/gpt_vision_food_eval/run_gpt_vision_eval.py \
  --labels experiment/ai/cv/gpt_vision_food_eval/sample_labels.csv \
  --limit 5
```

The script reads `OPENAI_API_KEY` from the environment. Missing images, missing API keys, API failures, JSON parse failures, and empty food results are recorded as failure rows instead of stopping the full evaluation.

## Outputs

Outputs are written to `experiment/ai/cv/gpt_vision_food_eval/outputs/` by default:

- `predictions.csv`
- `metrics.json`
- `report.md`

## Metrics

- `image_count`
- `api_success_rate`
- `json_parse_success_rate`
- `exact_match_rate_raw`
- `exact_match_rate_canonical`
- `any_food_hit_rate`
- `unmatched_food_rate`
- `empty_result_rate`
- `avg_latency_seconds`
- `p95_latency_seconds`

## Matching

The evaluation records both raw food names and canonical names. Canonical matching reuses:

- `ai_runtime/cv/food/normalization.py`
- `ai_runtime/cv/food/matcher.py`

This keeps experiment scoring aligned with the runtime food matching layer while keeping this script independent from the production API contract.
