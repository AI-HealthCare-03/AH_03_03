# CV Food3165 YOLO Baseline Plan

이 문서는 현재 업로드된 음식 CV 모델의 저장 위치, Git 포함/제외 기준, 평가 지표를 정리한다. 이번 모델은 AI-Hub 1000종 classification 모델이 아니라 YOLOv3-tiny 기반 3165-class 음식 detection/classification 모델로 본다.

실제 모델 학습, 대용량 weight 커밋, 외부 API 호출은 이 문서 범위에 포함하지 않는다.

## 1. Repository Boundary

실험 자료는 `experiment/ai/cv/food3165_yolo/` 아래에 둔다.

```text
experiment/ai/cv/food3165_yolo/
  README.md
  UPSTREAM_README.md
  yolo/
    config/
      food3165-yolov3-tiny_3l-v3-2.cfg
      food3165-darknet-v1.data
    data/
      food/
        food3165-classes.names
        food3165-classes.codes
        Standard_Code.txt
        Standard_CodeName.txt
        weights/
  images_rec/
  font/
  nutrition/
    raw/
```

서비스 런타임 코드는 `ai_runtime/cv/food/` 아래에 둔다.

```text
ai_runtime/cv/food/
  labels/
  configs/
  inference/
    food_yolo_predictor.py
```

런타임 영역에는 label map, class names, inference config, predictor/preprocess/postprocess 코드만 둔다. 대용량 weight 파일은 Git에 추가하지 않는다.

## 2. Experiment Directory Policy

| 경로 | 설명 | Git 포함 여부 |
|---|---|---|
| `README.md` | 현재 repo 기준 정리 문서 | 포함 가능 |
| `UPSTREAM_README.md` | 업로드된 원본 설명 문서 보존 | 포함 가능 |
| `yolo/config/*.cfg` | Darknet YOLO 모델 구조 설정 | 포함 가능 |
| `yolo/config/*.data` | Darknet data 설정 | 포함 가능 |
| `yolo/data/food/*classes.*` | food3165 label/code mapping | 포함 가능 |
| `yolo/data/food/Standard_Code*.txt` | 표준 코드 reference | 포함 가능 |
| `yolo/data/food/weights/` | YOLO weight 파일 | 제외 |
| `images_rec/` | 예측 샘플 이미지/결과 | 제외 |
| `font/` | 로컬 표시용 폰트 | 제외 |
| `nutrition/raw/` | 대용량 영양성분 원본 엑셀 | 제외 |

## 3. Git Ignore Policy

Git에서 제외한다.

- `*.weights`
- `*.xlsx`
- `*.ttc`
- `*.pt`, `*.pth`, `*.onnx`, `*.ckpt`, `*.safetensors`, `*.zip`
- `experiment/**/images_rec/**`
- `experiment/**/font/**`
- `experiment/**/*.log`
- `experiment/**/raw/**`, `experiment/**/processed/**`, `experiment/**/models/**`

Git에 포함 가능하다.

- `experiment/**/README.md`
- `experiment/**/configs/**`
- `experiment/**/reports/**/*.md`
- `experiment/**/reports/**/*.json`
- `food3165` label/config 소형 텍스트 파일

## 4. Baseline Metrics

Food3165 YOLO baseline 평가는 detection/classification 성격을 함께 고려한다.

- class Top-1 Accuracy 또는 top prediction accuracy
- Top-k hit rate
- macro F1
- confidence distribution
- class-wise precision/recall
- detection box sanity check
- low-confidence sample review count

AUC는 이 태스크의 1차 지표로 두지 않는다.

## 5. Runtime Handoff

학습/평가가 끝난 뒤 런타임에 넘길 수 있는 것은 다음으로 제한한다.

- `food3165` class id to class name mapping
- `food3165` label normalization table
- inference threshold config
- preprocess/postprocess code
- 모델 artifact 위치 또는 registry id

모델 binary 자체를 repo에 직접 넣는 방식은 피한다.

## 6. Deferred Work

GPT Vision fallback 기준은 후속 작업으로 분리한다.

- confidence threshold
- Top-k margin
- OOD 음식 판단 기준
- 사용자 confirm UI와 연결되는 후보 표시 정책
