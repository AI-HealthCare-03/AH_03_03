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

By default, evaluation outputs are written to:

```text
experiment/ai/cv/gpt_vision_food_eval/outputs/
```

Use `--output-dir <path>` only when you want a custom output location.

## Optional Langfuse Tracing

The evaluation runner can record one Langfuse generation per image when Langfuse is configured. Tracing is optional and best-effort: missing Langfuse settings or Langfuse SDK errors do not fail the evaluation.

Local terminal example. Use `localhost:3000` when running the script directly with local `uv run`:

```bash
export LANGFUSE_ENABLED=true
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_BASE_URL=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=<LANGFUSE_PUBLIC_KEY>
export LANGFUSE_SECRET_KEY=<LANGFUSE_SECRET_KEY>
```

Docker container example. Use `host.docker.internal:3000` when running from inside a Docker container:

```bash
export LANGFUSE_ENABLED=true
export LANGFUSE_HOST=http://host.docker.internal:3000
export LANGFUSE_BASE_URL=http://host.docker.internal:3000
export LANGFUSE_PUBLIC_KEY=<LANGFUSE_PUBLIC_KEY>
export LANGFUSE_SECRET_KEY=<LANGFUSE_SECRET_KEY>
```

Run with a stable eval run ID:

```bash
uv run python experiment/ai/cv/gpt_vision_food_eval/run_gpt_vision_eval.py \
  --labels experiment/ai/cv/gpt_vision_food_eval/data/labels/aihub_labels_sample.csv \
  --allowed-foods experiment/ai/cv/gpt_vision_food_eval/outputs/allowed_foods.json \
  --limit 5 \
  --eval-run-id smoke-langfuse
```

If `--eval-run-id` is omitted, the runner creates a timestamp-based ID. Langfuse metadata records image path, image filename, expected foods, predicted food names, canonical matches, row metrics, status, failure reason, confidence, and latency. Image bytes are never uploaded to Langfuse.

## Build AI-Hub Labels

AI-Hub food labels can be used either as extracted JSON directories or as archive zips with nested `*_Val_json.zip` files.

For the current local extracted dataset, run:

```bash
uv run python experiment/ai/cv/gpt_vision_food_eval/build_aihub_labels.py \
  --json-root experiment/ai/cv/gpt_vision_food_eval/data/json_zips \
  --image-root experiment/ai/cv/gpt_vision_food_eval/data/images \
  --output experiment/ai/cv/gpt_vision_food_eval/data/labels/aihub_labels_sample.csv \
  --limit 1000
```

Zip input is still supported:

```bash
uv run python experiment/ai/cv/gpt_vision_food_eval/build_aihub_labels.py \
  --zip-path "experiment/ai/cv/gpt_vision_food_eval/data/json_zips/아카이브.zip" \
  --image-root experiment/ai/cv/gpt_vision_food_eval/data/images \
  --output experiment/ai/cv/gpt_vision_food_eval/data/labels/aihub_labels_sample.csv \
  --limit 100
```

The output CSV columns are:

- `image_path`
- `image_filename`
- `expected_foods`
- `image_exists`
- `label_source`
- `annotation_count`
- `cat_1`
- `cat_2`
- `cat_3`

The builder uses JSON `"Code Name"` as `image_filename`. For `expected_foods`, it prefers the nearest Korean parent folder name such as `군고구마 json` -> `군고구마`; if no Korean folder label is found, it falls back to JSON `"Name"`. If `--image-root` is provided, the builder matches `image_filename` to local jpg/png files and writes an `image_path` relative to the generated CSV so `run_gpt_vision_eval.py` can read it directly. Broken JSON files and missing images are summarized in `outputs/aihub_label_summary.json`.

The builder also writes `outputs/allowed_foods.json`, a unique list of generated `expected_foods` labels.

Balanced per-class sampling is available for small evaluation batches:

```bash
uv run python experiment/ai/cv/gpt_vision_food_eval/build_aihub_labels.py \
  --json-root experiment/ai/cv/gpt_vision_food_eval/data/json_zips \
  --image-root experiment/ai/cv/gpt_vision_food_eval/data/images \
  --output experiment/ai/cv/gpt_vision_food_eval/data/labels/aihub_balanced_10_per_class.csv \
  --per-class-limit 10 \
  --seed 42
```

By default, balanced sampling excludes `image_exists=false` rows from the output CSV while still recording them in `outputs/aihub_label_summary.json` and `outputs/aihub_missing_images.csv`. Add `--include-missing` only when you intentionally want missing-image rows in the sampled CSV.

## Constrained Label Eval

Free-form GPT Vision evaluation lets the model produce arbitrary food names. Constrained evaluation passes `allowed_foods.json` to the prompt and asks the model to return only labels from that list:

```bash
uv run python experiment/ai/cv/gpt_vision_food_eval/run_gpt_vision_eval.py \
  --labels experiment/ai/cv/gpt_vision_food_eval/data/labels/aihub_labels_sample.csv \
  --allowed-foods experiment/ai/cv/gpt_vision_food_eval/outputs/allowed_foods.json \
  --limit 5
```

If GPT returns a food name outside the allowed list, the runner records it as an invalid label, then tries to correct it through the runtime normalization/matcher layer and a conservative string match. Uncorrectable labels are written as `unknown`.

This mode can increase prompt size and API cost when `allowed_foods.json` is large. For broad full-dataset evaluations, consider limiting the allowed list by class group or running smaller batches.

## Outputs

Evaluation outputs are written to `experiment/ai/cv/gpt_vision_food_eval/outputs/` by default:

- `predictions.csv`
- `metrics.json`
- `report.md`

Builder outputs are written to `experiment/ai/cv/gpt_vision_food_eval/outputs/`:

- `allowed_foods.json`
- `aihub_label_summary.json`
- `aihub_missing_images.csv`

## Metrics

- `total_rows`
- `evaluable_image_count`
- `data_missing_count`
- `api_failed_count`
- `json_parse_failed_count`
- `empty_result_count`
- `api_success_rate`
- `json_parse_success_rate`
- `raw_exact_match_rate`
- `canonical_exact_match_rate`
- `constrained_exact_match_rate`
- `any_hit_rate`
- `canonical_any_hit_rate`
- `constrained_any_hit_rate`
- `precision`
- `recall`
- `f1_score`
- `macro_precision`
- `macro_recall`
- `macro_f1_score`
- `invalid_label_count`
- `invalid_label_rate`
- `unknown_count`
- `unknown_rate`
- `unmatched_food_rate`
- `empty_result_rate`
- `avg_confidence`
- `confidence_correct_avg`
- `confidence_wrong_avg`
- `confidence_bins`
- `avg_latency_seconds`
- `p50_latency_seconds`
- `p95_latency_seconds`
- `max_latency_seconds`
- `class_distribution`
- `class_level_metrics`

## Matching

The evaluation records both raw food names and canonical names. Canonical matching reuses:

- `ai_runtime/cv/food/normalization.py`
- `ai_runtime/cv/food/matcher.py`

This keeps experiment scoring aligned with the runtime food matching layer while keeping this script independent from the production API contract.
