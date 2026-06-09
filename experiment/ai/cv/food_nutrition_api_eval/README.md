# Food Nutrition API Lookup Evaluation

This experiment prepares the food-name to nutrition-API lookup flow without changing production runtime code.

## Purpose

The goal is to evaluate whether food names produced by GPT Vision can be connected to public nutrition databases before replacing the current Excel/CSV-backed nutrition reference.

Target future providers:

- MFDS food nutrition database API
- RDA / 농식품올바로 food nutrition APIs

Current implementation is experiment-only and uses a stub provider. It is not connected to `app/services/diets.py` or production scoring.

## Input

Primary input:

```text
experiment/ai/cv/gpt_vision_food_eval/outputs/predictions.csv
```

The runner builds lookup queries from:

- `raw_food_names`
- `canonical_food_names`
- `allowed_food_names`

`expected_foods` is used only as the evaluation reference.

## Outputs

Outputs are written to `experiment/ai/cv/food_nutrition_api_eval/outputs/` by default:

- `nutrition_predictions.csv`
- `nutrition_metrics.json`
- `nutrition_report.md`
- `nutrition_lookup_cache.json`

Generated outputs are local-only and should not be committed.

## Run

Stub provider smoke run:

```bash
uv run python experiment/ai/cv/food_nutrition_api_eval/run_nutrition_lookup_eval.py \
  --predictions experiment/ai/cv/gpt_vision_food_eval/outputs/predictions.csv \
  --output-dir experiment/ai/cv/food_nutrition_api_eval/outputs \
  --provider stub \
  --limit 30
```

MFDS public API probe:

```bash
uv run python experiment/ai/cv/food_nutrition_api_eval/probe_public_api.py \
  --provider mfds \
  --limit 20 \
  --enable-fallback
```

The MFDS probe uses `FOOD_NM_KR` search against:

```text
https://apis.data.go.kr/1471000/FoodNtrCpntDbInfo02/getFoodNtrCpntDbInq02
```

It records original top1 and reranked top1 separately. The reranker is intentionally conservative:

- `matched` can be considered an experiment-level auto-accept candidate.
- `likely_match`, `multiple_candidates`, and `weak_match` should remain user-confirmation candidates.
- `no_candidates`, `api_unavailable`, and `parse_failed` should not block the whole batch.
- `fallback_used` means the original query was not enough and a fallback query provided the selected candidate.

Current penalty keywords are heuristic and experiment-only:

```text
과자, 크래커, 케이크, 라떼, 새우칩, 스낵, 음료, 아이스크림, 초콜릿, 사탕, 쿠키, 와플, 시리얼, 소스
```

These can incorrectly penalize legitimate foods, so they must be validated before any production use.

## Metrics

- `nutrition_lookup_success_rate`
- `top1_food_match_rate`
- `top3_candidate_hit_rate`
- `multiple_candidate_rate`
- `needs_user_confirmation_rate`
- `api_failure_rate`
- `cache_hit_rate`
- `avg_lookup_latency`
- `nutrition_field_completeness`
- `score_available_rate`

## Provider Contract

Experiment providers return a normalized lookup result with:

- `query`
- `normalized_query`
- `provider`
- `status`
- `matched_food_name`
- `matched_food_code`
- `candidate_count`
- `top_candidates`
- `energy_kcal`
- `carbohydrate_g`
- `protein_g`
- `fat_g`
- `sodium_mg`
- `serving_size`
- `source`
- `latency_seconds`
- `error_message`

## Env / API Keys

The current stub provider does not require an API key.

Future MFDS/RDA providers should read API keys from environment variables such as:

```env
MFDS_SERVICE_KEY=<MFDS_API_KEY>
MFDS_SERVICE_KEY_ENCODED=<MFDS_ENCODED_API_KEY>
RDA_FOOD_NUTRITION_API_KEY=<RDA_API_KEY>
```

Do not commit real API keys or generated API response dumps.

## Integration Policy

This experiment is intentionally not connected to production.

Recommended next steps:

1. Validate query cleanup and lookup success rate with the stub provider.
2. Add MFDS/RDA clients under this experiment directory.
3. Measure public API latency, top candidate quality, failure rate, and cache effectiveness.
4. Only after the provider is stable, introduce a production `NutritionProvider` adapter behind a feature flag.
