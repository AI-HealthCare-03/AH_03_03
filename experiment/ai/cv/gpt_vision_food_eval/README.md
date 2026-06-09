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

For AI-Hub data, keep local files under:

```text
experiment/ai/cv/gpt_vision_food_eval/
├── data/
│   ├── json_zips/  # AI-Hub label zip files
│   ├── images/     # food image files
│   └── labels/     # generated labels CSV files
└── outputs/        # generated evaluation reports
```

The data and output directories are local-only. Keep the `.gitkeep` files, but do not commit AI-Hub zip files, images, generated labels, or generated reports.

## Run

```bash
uv run python experiment/ai/cv/gpt_vision_food_eval/run_gpt_vision_eval.py \
  --labels experiment/ai/cv/gpt_vision_food_eval/sample_labels.csv \
  --limit 5
```

The script reads `OPENAI_API_KEY` from the environment. Missing images, missing API keys, API failures, JSON parse failures, and empty food results are recorded as failure rows instead of stopping the full evaluation.

## Build AI-Hub Labels

AI-Hub food archives can contain nested `*_Val_json.zip` files. Convert those JSON labels to an evaluation CSV with:

```bash
uv run python experiment/ai/cv/gpt_vision_food_eval/build_aihub_labels.py \
  --zip-path "experiment/ai/cv/gpt_vision_food_eval/data/json_zips/아카이브.zip" \
  --output experiment/ai/cv/gpt_vision_food_eval/data/labels/aihub_labels_sample.csv \
  --limit 100
```

The output CSV columns are:

- `image_filename`
- `expected_foods`
- `label_source`
- `annotation_count`
- `cat_1`
- `cat_2`
- `cat_3`

The builder uses JSON `"Code Name"` as `image_filename`, JSON `"Name"` as the first food label candidate, and path categories as fallback. Broken JSON files are skipped and summarized in `outputs/aihub_label_summary.json`.

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
