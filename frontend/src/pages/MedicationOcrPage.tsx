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
import ErrorMessage from "../components/ErrorMessage";
import { useAsyncJobPolling } from "../hooks/useAsyncJobPolling";
import { getAsyncJobStatusMessage } from "../utils/asyncJobStatus";
import { isHeicFile } from "../utils/files";

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
  const [ocrJobId, setOcrJobId] = useState<number | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [canRetryOcr, setCanRetryOcr] = useState(false);

  useEffect(() => {
    return () => {
      if (selectedPreviewUrl) {
        URL.revokeObjectURL(selectedPreviewUrl);
      }
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

  const { latestJob: latestOcrJob } = useAsyncJobPolling({
    jobId: ocrJobId,
    enabled: isRunning && ocrJobId !== null,
    intervalMs: 1500,
    timeoutMs: 120000,
    onSuccess: (job) => {
      const result = medicationResultFromPayload(job.result_payload);
      const nextItems = result?.items ?? [];
      setItems(nextItems);
      setCanRetryOcr(false);
      setMessage(
        nextItems.length > 0
          ? `${getAsyncJobStatusMessage("SUCCESS")} 저장 전 약 이름과 복용 정보를 반드시 확인해주세요.`
          : `${getAsyncJobStatusMessage("SUCCESS")} 인식된 복약 정보 후보가 없습니다. 파일을 다시 확인해주세요.`,
      );
      setIsRunning(false);
      setOcrJobId(null);
    },
    onFailure: (job) => {
      setError(getAsyncJobStatusMessage(job.status === "CANCELED" ? "CANCELED" : "FAILED"));
      setCanRetryOcr(true);
      setIsRunning(false);
      setOcrJobId(null);
    },
    onTimeout: () => {
      setError(getAsyncJobStatusMessage("TIMEOUT"));
      setCanRetryOcr(true);
      setIsRunning(false);
      setOcrJobId(null);
    },
  });

  const ocrStatusMessage =
    isRunning && ocrJobId !== null ? getAsyncJobStatusMessage(latestOcrJob?.status ?? "PENDING") : "";

  const runMedicationRecognition = async () => {
    setError("");
    setMessage("");
    setItems([]);
    setCanRetryOcr(false);
    if (!selectedImageFile) {
      setError("처방전 또는 약봉투 이미지 파일을 먼저 선택해주세요.");
      return;
    }
    setIsRunning(true);
    try {
      const job = await runMedicationOcr(buildMedicationOcrPayload(sourceType, selectedImageFile, imageFilename));
      setMessage("");
      setOcrJobId(job.id);
    } catch {
      setError("분석 요청을 시작하지 못했습니다. 파일을 확인한 뒤 다시 시도해주세요.");
      setCanRetryOcr(true);
      setIsRunning(false);
    }
  };

  const updateItem = (index: number, key: keyof MedicationOcrItem, value: string | number | string[] | null) => {
    setItems((prev) => prev.map((item, itemIndex) => (itemIndex === index ? { ...item, [key]: value } : item)));
  };

  const handleImageSelection = async (file: File | null) => {
    if (selectedPreviewUrl) {
      URL.revokeObjectURL(selectedPreviewUrl);
    }
    setSelectedPreviewUrl("");
    setPreviewMessage("");
    if (!file) {
      return;
    }
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
      setMessage(`확인한 복약 정보가 저장되었습니다. 생성 ${response.created_count}건, 건너뜀 ${response.skipped_count}건`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "복약정보 저장에 실패했습니다.");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="page-stack">
      <div className="page-header">
        <div>
          <h1>복약/처방전 정보 확인</h1>
          <p>처방전 또는 약봉투 이미지/텍스트 기반 복약 정보 후보를 생성합니다.</p>
        </div>
        <Link className="button secondary" to="/ocr">
          등록 선택으로 돌아가기
        </Link>
      </div>
      {error && <ErrorMessage message={error} />}
      {canRetryOcr ? (
        <div className="button-row">
          <button disabled={isRunning} onClick={runMedicationRecognition} type="button">
            다시 시도
          </button>
        </div>
      ) : null}
      {ocrStatusMessage && <div className="state-box">{ocrStatusMessage}</div>}
      {message && <div className="state-box">{message}</div>}
      <div className="page-grid">
        <Card title="처방전/약봉투 업로드">
          <div className="upload-box">
            <strong>이미지 업로드 영역</strong>
            <span>촬영/업로드 후 생성된 후보 정보를 확인하고 저장해주세요.</span>
            <p className="warning-text">
              업로드한 이미지는 건강정보 추출 및 분석을 위해 사용됩니다. 자동 인식 결과는 오류가 있을 수 있으므로
              저장 또는 분석 전에 내용을 확인해주세요. 복약 정보는 민감한 건강정보일 수 있으므로 본인 자료만
              업로드해주세요.
            </p>
            <div className="upload-action-grid">
              <label className="upload-action-button">
                파일에서 선택
                <input
                  accept="image/*,.heic,.heif"
                  type="file"
                  onChange={(event) => handleImageSelection(event.currentTarget.files?.[0] ?? null)}
                />
              </label>
              {isMobileDevice ? (
                <label className="upload-action-button">
                  카메라로 촬영
                  <input
                    accept="image/*,.heic,.heif"
                    capture="environment"
                    type="file"
                    onChange={(event) => handleImageSelection(event.currentTarget.files?.[0] ?? null)}
                  />
                </label>
              ) : (
                <span className="upload-mobile-hint">카메라 촬영은 모바일에서 사용할 수 있습니다.</span>
              )}
            </div>
            <span className="muted">선택된 파일: {imageFilename || "없음"}</span>
            {previewMessage ? (
              <div className="state-box heic-preview-notice">
                {previewMessage}
              </div>
            ) : null}
            {selectedPreviewUrl ? (
              <img alt="선택한 처방전 또는 약봉투 이미지 미리보기" className="upload-preview" src={selectedPreviewUrl} />
            ) : null}
          </div>
          <label>
            인식 유형
            <select className="input" value={sourceType} onChange={(event) => setSourceType(event.target.value)}>
              <option value="PRESCRIPTION">처방전</option>
              <option value="MEDICATION_BAG">약봉투</option>
              <option value="SUPPLEMENT">영양제</option>
            </select>
          </label>
          <button disabled={isRunning} onClick={runMedicationRecognition} type="button">
            {isRunning ? "후보 생성 중..." : "복약 정보 후보 생성"}
          </button>
        </Card>
        <Card title="확인 안내">
          <p className="warning-text">약 정보는 반드시 사용자가 직접 확인해야 합니다. 치료 변경은 의료진과 상담해주세요.</p>
          <button disabled={items.length === 0 || isSaving} onClick={save} type="button">
            {isSaving ? "저장 중..." : "확인/저장"}
          </button>
          <Link className="button secondary" style={{ marginTop: 12 }} to="/medications">
            복약정보 화면으로 이동
          </Link>
        </Card>
      </div>
      <Card title="복약 정보 후보">
        <div className="ocr-result-table">
          {items.length === 0 && <div className="state-box">아직 추출 결과가 없습니다.</div>}
          {items.map((item, index) => (
            <div className="ocr-medication-card" key={item.temp_id ?? `${item.name}-${index}`}>
              <label>
                약 이름
                <input value={item.name} onChange={(event) => updateItem(index, "name", event.target.value)} />
              </label>
              <label>
                용량
                <input value={item.dosage ?? ""} onChange={(event) => updateItem(index, "dosage", event.target.value)} />
              </label>
              <label>
                복용 횟수
                <input
                  value={item.frequency ?? ""}
                  onChange={(event) => updateItem(index, "frequency", event.target.value)}
                />
              </label>
              <label>
                복용 시간
                <input
                  value={item.time_slots.join(", ")}
                  onChange={(event) =>
                    updateItem(
                      index,
                      "time_slots",
                      event.target.value
                        .split(",")
                        .map((value) => value.trim())
                        .filter(Boolean),
                    )
                  }
                />
              </label>
              <label>
                복용 기간
                <input
                  value={item.duration_days ?? ""}
                  onChange={(event) =>
                    updateItem(index, "duration_days", event.target.value ? Number(event.target.value) : null)
                  }
                />
              </label>
              <label>
                메모
                <input value={item.memo ?? ""} onChange={(event) => updateItem(index, "memo", event.target.value)} />
              </label>
              {item.confidence !== null && item.confidence !== undefined && (
                <span className="badge">신뢰도 {(item.confidence * 100).toFixed(0)}%</span>
              )}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function buildMedicationOcrPayload(sourceType: string, file: File | null, imageFilename: string): MedicationOcrRequest | FormData {
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

function medicationResultFromPayload(payload: Record<string, unknown> | null | undefined): MedicationOcrResponse | null {
  if (!payload || !Array.isArray(payload.items)) {
    return null;
  }
  return payload as MedicationOcrResponse;
}
