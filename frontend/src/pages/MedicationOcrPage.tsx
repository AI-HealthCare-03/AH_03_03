import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import {
  confirmMedicationOcr,
  type MedicationOcrItem,
  type MedicationOcrRequest,
  type MedicationOcrResponse,
  runMedicationOcr,
} from "../api/medications";
import { normalizeImageForPreview } from "../api/uploads";
import Card from "../components/Card";
import ConfirmDialog from "../components/ConfirmDialog";
import ErrorMessage from "../components/ErrorMessage";
import { useAsyncJobPolling } from "../hooks/useAsyncJobPolling";
import { isHeicFile } from "../utils/files";

type FeedbackDialog = {
  message: string;
  title: string;
  tone?: "default" | "danger";
};

type Step = 1 | 2 | 3;

function StepIndicator({ current }: { current: Step }) {
  const steps: { label: string; num: Step }[] = [
    { num: 1, label: "파일 업로드" },
    { num: 2, label: "복약정보 확인" },
    { num: 3, label: "저장 완료" },
  ];
  return (
    <div className="step-indicator">
      {steps.map((step, i) => (
        <div key={step.num} className="step-indicator__item">
          <div
            className={[
              "step-indicator__circle",
              current > step.num ? "step-indicator__circle--done" : "",
              current === step.num ? "step-indicator__circle--active" : "",
            ]
              .filter(Boolean)
              .join(" ")}
          >
            {current > step.num ? "✓" : step.num}
          </div>
          <span
            className={[
              "step-indicator__label",
              current === step.num ? "step-indicator__label--active" : "",
            ]
              .filter(Boolean)
              .join(" ")}
          >
            {step.label}
          </span>
          {i < steps.length - 1 && <div className="step-indicator__line" />}
        </div>
      ))}
    </div>
  );
}

export default function MedicationOcrPage() {
  const [items, setItems] = useState<MedicationOcrItem[]>([]);
  const [sourceType, setSourceType] = useState("PRESCRIPTION");
  const [imageFilename, setImageFilename] = useState("");
  const [selectedImageFile, setSelectedImageFile] = useState<File | null>(null);
  const [selectedPreviewUrl, setSelectedPreviewUrl] = useState("");
  const [previewMessage, setPreviewMessage] = useState("");
  const [isMobileDevice, setIsMobileDevice] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [feedbackDialog, setFeedbackDialog] = useState<FeedbackDialog | null>(null);
  const [ocrJobId, setOcrJobId] = useState<number | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [canRetryOcr, setCanRetryOcr] = useState(false);

  const currentStep: Step = isSaved ? 3 : items.length > 0 ? 2 : 1;

  useEffect(() => {
    return () => {
      if (selectedPreviewUrl) URL.revokeObjectURL(selectedPreviewUrl);
    };
  }, [selectedPreviewUrl]);

  useEffect(() => {
    const checkMobile = () => {
      const coarsePointer = window.matchMedia?.("(pointer: coarse)").matches ?? false;
      const mobileUserAgent = /Android|iPhone|iPad|iPod/i.test(window.navigator.userAgent);
      setIsMobileDevice(coarsePointer || mobileUserAgent);
    };
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  useAsyncJobPolling({
    jobId: ocrJobId,
    enabled: isRunning && ocrJobId !== null,
    intervalMs: 1500,
    timeoutMs: 120000,
    onSuccess: (job) => {
      setIsRunning(false);
      setOcrJobId(null);
      const result = medicationResultFromPayload(job.result_payload);
      setItems(result?.items ?? []);
      setCanRetryOcr(false);
      setFeedbackDialog({
        title: "복약 정보 인식이 완료되었습니다.",
        message: "인식된 항목을 확인하고 필요한 경우 수정해 주세요.",
      });
    },
    onFailure: (job) => {
      setFeedbackDialog({
        title: "복약 정보 인식에 실패했습니다.",
        message:
          job.status === "CANCELED"
            ? "복약 정보 인식 작업이 취소되었습니다."
            : "이미지를 다시 확인한 뒤 업로드해 주세요.",
        tone: "danger",
      });
      setCanRetryOcr(true);
      setIsRunning(false);
      setOcrJobId(null);
    },
    onTimeout: () => {
      setFeedbackDialog({
        title: "복약 정보 인식에 실패했습니다.",
        message: "이미지를 다시 확인한 뒤 업로드해 주세요.",
        tone: "danger",
      });
      setCanRetryOcr(true);
      setIsRunning(false);
      setOcrJobId(null);
    },
  });

  const runMedicationRecognition = async () => {
    setError("");
    setMessage("");
    setFeedbackDialog(null);
    setItems([]);
    setIsSaved(false);
    setCanRetryOcr(false);
    if (!selectedImageFile) {
      setError("처방전 또는 약봉투 이미지 파일을 먼저 선택해주세요.");
      return;
    }
    setIsRunning(true);
    try {
      const job = await runMedicationOcr(
        buildMedicationOcrPayload(sourceType, selectedImageFile, imageFilename),
      );
      setOcrJobId(job.id);
    } catch {
      setError("분석 요청을 시작하지 못했습니다. 파일을 확인한 뒤 다시 시도해주세요.");
      setCanRetryOcr(true);
      setIsRunning(false);
    }
  };

  const updateItem = (
    index: number,
    key: keyof MedicationOcrItem,
    value: string | number | string[] | null,
  ) => {
    setIsSaved(false);
    setItems((prev) =>
      prev.map((item, i) => (i === index ? { ...item, [key]: value } : item)),
    );
  };

  const handleImageSelection = async (file: File | null) => {
    if (selectedPreviewUrl) URL.revokeObjectURL(selectedPreviewUrl);
    setSelectedPreviewUrl("");
    setPreviewMessage("");
    if (!file) return;
    setSelectedImageFile(file);
    setImageFilename(file.name);
    if (!isHeicFile(file)) {
      setSelectedPreviewUrl(URL.createObjectURL(file));
      return;
    }
    setPreviewMessage("HEIC 이미지를 미리보기용 JPG로 변환 중입니다.");
    try {
      const previewBlob = await normalizeImageForPreview(file);
      setSelectedPreviewUrl(URL.createObjectURL(previewBlob));
      setPreviewMessage("");
    } catch (err) {
      setPreviewMessage(
        err instanceof Error
          ? err.message
          : "HEIC 미리보기를 생성하지 못했습니다. 분석은 업로드 후 다시 시도해주세요.",
      );
    }
  };

  const save = async () => {
    setError("");
    setMessage("");
    setIsSaving(true);
    try {
      const response = await confirmMedicationOcr({
        items: items.map((item) => ({
          name: item.name,
          dosage: item.dosage,
          frequency: item.frequency,
          time_slots: item.time_slots,
          duration_days: item.duration_days,
          memo: item.memo,
        })),
      });
      setIsSaved(true);
      setMessage(
        `확인한 복약 정보가 저장되었습니다. 생성 ${response.created_count}건, 건너뜀 ${response.skipped_count}건`,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "복약정보 저장에 실패했습니다.");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="page-stack">
      {/* 헤더 */}
      <div className="page-header">
        <div>
          <h1>복약/처방전 정보 확인</h1>
          <p>처방전 또는 약봉투 이미지 기반 복약 정보 후보를 생성합니다.</p>
        </div>
        <Link className="button secondary" to="/medications">
          등록 선택으로 돌아가기
        </Link>
      </div>

      {/* 스텝 인디케이터 */}
      <StepIndicator current={currentStep} />

      {/* 에러 / 재시도 */}
      {error && <ErrorMessage message={error} />}
      {canRetryOcr && (
        <div className="button-row">
          <button disabled={isRunning} onClick={runMedicationRecognition} type="button">
            다시 시도
          </button>
        </div>
      )}
      {message && <div className="state-box">{message}</div>}

      {/* 피드백 다이얼로그 */}
      {feedbackDialog && (
        <ConfirmDialog
          confirmLabel="확인"
          message={feedbackDialog.message}
          onConfirm={() => setFeedbackDialog(null)}
          showCancel={false}
          title={feedbackDialog.title}
          tone={feedbackDialog.tone}
        />
      )}

      {/* ── STEP 1: 파일 업로드 ── */}
      <Card title="처방전/약봉투 업로드">
        <div className="upload-box">
          <div className="upload-action-grid">
            <label className="upload-action-button">
              파일에서 선택
              <input
                accept="image/*,.heic,.heif"
                type="file"
                onChange={(e) => handleImageSelection(e.currentTarget.files?.[0] ?? null)}
              />
            </label>
            {isMobileDevice ? (
              <label className="upload-action-button">
                카메라로 촬영
                <input
                  accept="image/*,.heic,.heif"
                  capture="environment"
                  type="file"
                  onChange={(e) => handleImageSelection(e.currentTarget.files?.[0] ?? null)}
                />
              </label>
            ) : (
              <span className="upload-action-button upload-action-button--disabled">
                <span style={{ fontSize: "14px", fontWeight: 600 }}>카메라 촬영</span>
                <span style={{ fontSize: "11px", fontWeight: 400, opacity: 0.7 }}>
                  카메라 촬영은 모바일에서 사용할 수 있습니다.
                </span>
              </span>
            )}
          </div>

          <span className="muted">선택된 파일: {imageFilename || "없음"}</span>

          {previewMessage && (
            <div className="state-box heic-preview-notice">{previewMessage}</div>
          )}
          {selectedPreviewUrl && (
            <img
              alt="선택한 처방전 또는 약봉투 이미지 미리보기"
              className="upload-preview"
              src={selectedPreviewUrl}
            />
          )}
        </div>

        <p className="warning-text" style={{ marginTop: 8 }}>
          업로드한 이미지는 건강정보 추출에만 사용됩니다. 자동 인식 결과에 오류가 있을 수
          있으니 저장 전 반드시 확인하세요. 복약 정보는 민감한 정보이므로 본인 자료만
          업로드해주세요.
        </p>

        <label style={{ marginTop: 12 }}>
          인식 유형
          <select className="input" value={sourceType} onChange={(e) => setSourceType(e.target.value)}>
            <option value="PRESCRIPTION">처방전</option>
            <option value="MEDICATION_BAG">약봉투</option>
            <option value="SUPPLEMENT">영양제</option>
          </select>
        </label>

        <div className="button-row" style={{ marginTop: 12 }}>
          <button disabled={isRunning || !selectedImageFile} onClick={runMedicationRecognition} type="button">
            {isRunning ? "후보 생성 중..." : "복약 정보 인식 시작"}
          </button>
        </div>
      </Card>

      {/* ── STEP 2~3: 복약 정보 후보 ── */}
      <Card title="복약 정보 후보">
        <div className="ocr-result-table">
          {items.length === 0 ? (
            <div className="state-box">
              아직 추출 결과가 없습니다. 파일을 업로드하고 복약 정보 후보를 생성해주세요.
            </div>
          ) : (
            items.map((item, index) => (
              <div className="ocr-medication-card" key={item.temp_id ?? `${item.name}-${index}`}>
                <label>
                  약 이름
                  <input value={item.name} onChange={(e) => updateItem(index, "name", e.target.value)} />
                </label>
                <label>
                  용량
                  <input value={item.dosage ?? ""} onChange={(e) => updateItem(index, "dosage", e.target.value)} />
                </label>
                <label>
                  복용 횟수
                  <input value={item.frequency ?? ""} onChange={(e) => updateItem(index, "frequency", e.target.value)} />
                </label>
                <label>
                  복용 시간
                  <input
                    value={item.time_slots.join(", ")}
                    onChange={(e) =>
                      updateItem(
                        index,
                        "time_slots",
                        e.target.value.split(",").map((v) => v.trim()).filter(Boolean),
                      )
                    }
                  />
                </label>
                <label>
                  복용 기간
                  <input
                    value={item.duration_days ?? ""}
                    onChange={(e) =>
                      updateItem(index, "duration_days", e.target.value ? Number(e.target.value) : null)
                    }
                  />
                </label>
                <label>
                  메모
                  <input value={item.memo ?? ""} onChange={(e) => updateItem(index, "memo", e.target.value)} />
                </label>
                {item.confidence !== null && item.confidence !== undefined && (
                  <span className="badge">신뢰도 {(item.confidence * 100).toFixed(0)}%</span>
                )}
              </div>
            ))
          )}
        </div>

        {/* 하단 액션 */}
        <div
          className="button-row"
          style={{ marginTop: 16, justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}
        >
          <div className="button-row" style={{ margin: 0 }}>
            <Link className="button secondary" to="/medications">
              복약정보 화면으로 이동
            </Link>
          </div>
          <button
            disabled={items.length === 0 || isSaving || isSaved}
            onClick={save}
            type="button"
          >
            {isSaving ? "저장 중..." : isSaved ? "저장 완료 ✓" : "확인/저장"}
          </button>
        </div>
      </Card>
    </div>
  );
}

function buildMedicationOcrPayload(
  sourceType: string,
  file: File | null,
  imageFilename: string,
): MedicationOcrRequest | FormData {
  if (!file) {
    return {
      source_type: sourceType,
      image_filename: imageFilename || undefined,
      memo: "medication ocr request",
    };
  }
  const formData = new FormData();
  formData.append("image", file);
  formData.append("source_type", sourceType);
  formData.append("image_filename", file.name);
  return formData;
}

function medicationResultFromPayload(
  payload: Record<string, unknown> | null | undefined,
): MedicationOcrResponse | null {
  if (!payload || !Array.isArray(payload.items)) return null;
  return payload as MedicationOcrResponse;
}