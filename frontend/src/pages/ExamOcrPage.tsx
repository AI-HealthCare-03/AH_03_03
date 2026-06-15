import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import {
  confirmExam,
  createExam,
  listMeasurements,
  runExamOcr,
  updateMeasurement,
  type ExamMeasurement,
  type ExamReport,
} from "../api/exams";
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

// 스텝 인디케이터
type Step = 1 | 2 | 3 | 4;

function StepIndicator({ current }: { current: Step }) {
  const steps: { label: string; num: Step }[] = [
    { num: 1, label: "파일 업로드 및 인식" },
    { num: 2, label: "인식 결과 확인" },
    { num: 3, label: "검진 결과 등록" },
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

export default function ExamOcrPage() {
  const [selectedFileName, setSelectedFileName] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedPreviewUrl, setSelectedPreviewUrl] = useState("");
  const [previewMessage, setPreviewMessage] = useState("");
  const [isMobileDevice, setIsMobileDevice] = useState(false);
  const [exam, setExam] = useState<ExamReport | null>(null);
  const [measurements, setMeasurements] = useState<ExamMeasurement[]>([]);
  const [ocrJobId, setOcrJobId] = useState<number | null>(null);
  const [isRunningOcr, setIsRunningOcr] = useState(false);
  const [isConfirming, setIsConfirming] = useState(false);
  const [isAppliedToHealth, setIsAppliedToHealth] = useState(false);
  const [error, setError] = useState("");
  const [feedbackDialog, setFeedbackDialog] = useState<FeedbackDialog | null>(null);
  const [canRetryOcr, setCanRetryOcr] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);

  // 현재 스텝 계산
  const currentStep: Step = isAppliedToHealth ? 4 : measurements.length > 0 ? 3 : selectedFile ? 2 : 1;

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

  useEffect(() => {
    if (exam?.is_confirmed) setIsAppliedToHealth(true);
  }, [exam?.is_confirmed]);

  useAsyncJobPolling({
    jobId: ocrJobId,
    enabled: isRunningOcr && ocrJobId !== null,
    intervalMs: 1500,
    timeoutMs: 120000,
    onSuccess: async () => {
      setIsRunningOcr(false);
      setOcrJobId(null);
      if (!exam) {
        setFeedbackDialog({
          title: "검진표 인식에 실패했습니다.",
          message: "이미지를 다시 확인한 뒤 업로드해 주세요.",
          tone: "danger",
        });
        return;
      }
      try {
        const latestMeasurements = await listMeasurements(exam.id);
        setMeasurements(latestMeasurements);
        setCanRetryOcr(false);
        setFeedbackDialog({
          title: "검진표 인식이 완료되었습니다.",
          message: "인식 결과에 오타가 없는지 반드시 확인해 주세요. 잘못 입력된 항목은 직접 수정하실 수 있습니다.",
        });
      } catch {
        setFeedbackDialog({
          title: "검진표 인식에 실패했습니다.",
          message: "이미지를 다시 확인한 뒤 업로드해 주세요.",
          tone: "danger",
        });
        setCanRetryOcr(true);
      }
    },
    onFailure: (job) => {
      setFeedbackDialog({
        title: "검진표 인식에 실패했습니다.",
        message:
          job.status === "CANCELED"
            ? "검진표 인식 작업이 취소되었습니다."
            : "이미지를 다시 확인한 뒤 업로드해 주세요.",
        tone: "danger",
      });
      setCanRetryOcr(true);
      setIsRunningOcr(false);
      setOcrJobId(null);
    },
    onTimeout: () => {
      setFeedbackDialog({
        title: "검진표 인식에 실패했습니다.",
        message: "이미지를 다시 확인한 뒤 업로드해 주세요.",
        tone: "danger",
      });
      setCanRetryOcr(true);
      setIsRunningOcr(false);
      setOcrJobId(null);
    },
  });

  const handleFileSelection = async (file: File | null) => {
    if (selectedPreviewUrl) URL.revokeObjectURL(selectedPreviewUrl);
    setSelectedPreviewUrl("");
    setPreviewMessage("");
    setIsAppliedToHealth(false);
    setExam(null);
    setMeasurements([]);
    setError("");
    if (!file) {
      setSelectedFile(null);
      setSelectedFileName("");
      return;
    }
    setSelectedFile(file);
    setSelectedFileName(file.name);
    if (isPdfFile(file)) return;
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

  const startExamOcr = async () => {
    setError("");
    setFeedbackDialog(null);
    setMeasurements([]);
    setCanRetryOcr(false);
    setIsAppliedToHealth(false);
    if (!selectedFile) {
      setError("검진표 이미지 또는 PDF 파일을 먼저 선택해주세요.");
      return;
    }
    setIsRunningOcr(true);
    try {
      const report =
        exam ??
        (await createExam({
          original_filename: selectedFileName,
          file_path: `exam-upload/${selectedFileName}`,
          uploaded_at: new Date().toISOString(),
        }));
      setExam(report);
      const job = await runExamOcr(report.id, selectedFile);
      setOcrJobId(job.id);
    } catch {
      setError("분석 요청을 시작하지 못했습니다. 파일을 확인한 뒤 다시 시도해주세요.");
      setCanRetryOcr(true);
      setIsRunningOcr(false);
    }
  };

  const updateLocalMeasurement = (measurementId: number, value: string) => {
    setIsAppliedToHealth(false);
    setMeasurements((prev) =>
      prev.map((m) => (m.id === measurementId ? { ...m, value } : m)),
    );
  };

  const saveAndConfirm = async () => {
    if (!exam) {
      setError("먼저 측정값 후보를 생성해주세요.");
      return;
    }
    setError("");
    setIsConfirming(true);
    try {
      await Promise.all(
        measurements.map((m) =>
          updateMeasurement(m.id, {
            value: m.value,
            unit: m.unit,
            is_user_confirmed: true,
          }),
        ),
      );
      const confirmedExam = await confirmExam(exam.id);
      setExam(confirmedExam);
      setMeasurements(await listMeasurements(exam.id));
      setIsAppliedToHealth(true);
      setFeedbackDialog({
        title: "검진 결과가 반영되었습니다.",
        message: "이제 정밀분석이 가능합니다!",
      });
    } catch (err) {
      setIsAppliedToHealth(false);
      setFeedbackDialog({
        title: "검진 결과 반영에 실패했습니다.",
        message: "잠시 후 다시 시도해 주세요.",
        tone: "danger",
      });
      setError(err instanceof Error ? err.message : "검진 결과 반영에 실패했습니다.");
    } finally {
      setIsConfirming(false);
    }
  };

  return (
    <div className="page-stack">
      {/* 헤더 */}
      <div className="page-header">
        <div>
          <h1>건강검진표 사진/PDF 등록</h1>
          <p>검진표 이미지/PDF를 업로드하면 결과값을 자동으로 인식합니다.</p>
        </div>
        <Link className="button secondary" to="/health">
          건강 분석으로 돌아가기
        </Link>
      </div>

      {/* 스텝 인디케이터 */}
      <StepIndicator current={currentStep} />

      {/* 파일 업로드 도움말 모달 */}
      {isHelpOpen && (
        <div
          onClick={() => setIsHelpOpen(false)}
          style={{
            position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)",
            zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center",
            padding: "16px",
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              background: "var(--color-surface)", borderRadius: "var(--radius-lg)",
              padding: "28px", maxWidth: "560px", width: "100%",
              maxHeight: "80vh", overflowY: "auto",
              boxShadow: "0 8px 32px rgba(0,0,0,0.2)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
              <h2 style={{ margin: 0, fontSize: "18px" }}>건강검진 결과 PDF 다운로드 방법</h2>
              <button
                onClick={() => setIsHelpOpen(false)}
                type="button"
                style={{
                  background: "none", border: "none", fontSize: "20px",
                  cursor: "pointer", color: "var(--color-text-secondary)", lineHeight: 1,
                }}
              >
                ✕
              </button>
            </div>
            <ol style={{ paddingLeft: "20px", lineHeight: "2", margin: "0 0 16px" }}>
              <li>
                <a
                  href="https://www.nhis.or.kr/nhis/etc/personalLoginPage.do"
                  rel="noreferrer"
                  style={{ color: "var(--color-primary)", wordBreak: "break-all" }}
                  target="_blank"
                >
                  https://www.nhis.or.kr/nhis/etc/personalLoginPage.do
                </a>
                {" "}(국민건강보험) 사이트에 접속합니다.
              </li>
              <li>건강모아 → 건강검진 결과조회</li>
              <li>
                <img
                  alt="건강검진 결과조회 화면"
                  src="/images/nhis-guide.png"
                  style={{ marginTop: "8px", width: "100%", borderRadius: "8px", border: "1px solid var(--color-border)" }}
                  onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                />
              </li>
            </ol>
          </div>
        </div>
      )}

      {/* 에러 / 재시도 */}
      {error && <ErrorMessage message={error} />}
      {canRetryOcr && (
        <div className="button-row">
          <button disabled={isRunningOcr} onClick={startExamOcr} type="button">
            다시 시도
          </button>
        </div>
      )}

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
      <Card>
        <div className="card-header" style={{ marginBottom: "4px" }}>
          <h2 style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            파일 업로드
            <button
              aria-label="파일 업로드 도움말"
              onClick={() => setIsHelpOpen(true)}
              type="button"
              style={{
                width: "20px", height: "20px", borderRadius: "50%",
                border: "none", background: "var(--color-primary)",
                color: "#fff", fontSize: "12px", fontWeight: 800,
                cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
                flexShrink: 0, lineHeight: 1,
              }}
            >
              ?
            </button>
          </h2>
        </div>
        <p className="muted" style={{ marginBottom: "12px", fontSize: "14px" }}>
          국민건강보험공단 웹사이트의 [건강검진 결과조회] 메뉴에서 최근 검진 내역을 <strong>'PDF로 저장'</strong>하여 다운로드하실 수 있습니다. (자세한 방법은 우측의 ? 버튼을 참고해 주세요.)
        </p>
        <div className="upload-box">
          <div className="upload-action-grid">
            <label className="upload-action-button">
              파일에서 선택
              <input
                accept="image/*,.heic,.heif,.pdf"
                onChange={(e) => handleFileSelection(e.target.files?.[0] ?? null)}
                type="file"
              />
            </label>
            {isMobileDevice ? (
              <label className="upload-action-button">
                카메라로 촬영
                <input
                  accept="image/*,.heic,.heif"
                  capture="environment"
                  onChange={(e) => handleFileSelection(e.target.files?.[0] ?? null)}
                  type="file"
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

          <span className="muted">선택된 파일: {selectedFileName || "없음"}</span>

          {previewMessage && (
            <div className="state-box heic-preview-notice">{previewMessage}</div>
          )}
          {selectedPreviewUrl && (
            <img
              alt="선택한 검진표 이미지 미리보기"
              className="upload-preview"
              src={selectedPreviewUrl}
            />
          )}
        </div>

        {/* 경고 문구: 여기 한 번만 */}
        <p className="warning-text" style={{ marginTop: 8 }}>
          업로드한 이미지는 건강정보 추출에만 사용됩니다. 자동 인식 결과에 오류가 있을 수
          있으니 저장 전 반드시 확인하세요. 건강검진 결과는 민감한 정보이므로 본인 자료만
          업로드해주세요.
        </p>

        <div className="button-row" style={{ marginTop: 12 }}>
          <button disabled={isRunningOcr || !selectedFile} onClick={startExamOcr} type="button">
            {isRunningOcr ? "검진표 인식 중..." : "검진표 인식 시작"}
          </button>
        </div>
      </Card>

      {/* ── STEP 2~3: 측정값 후보 (파일 선택 후 항상 노출) ── */}
      <Card>
        <div className="card-header" style={{ marginBottom: "4px" }}>
          <h2 style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            검진표 인식 결과
            {measurements.length > 0 && !isAppliedToHealth && (
              <em className="badge badge-required">확인 필요</em>
            )}
          </h2>
        </div>
        <p className="muted" style={{ marginBottom: "12px", fontSize: "16px" }}>인식된 결과는 반드시 직접 확인해 주세요.</p>
        <div className="ocr-result-table">
          {measurements.length === 0 ? (
            <div className="state-box">
              아직 인식된 검진표가 없습니다. 파일을 업로드하고 '검진표 인식 시작'버튼을 눌러주세요.
            </div>
          ) : (
            measurements.map((m) => (
              <label className="ocr-result-row" key={m.id}>
                <span>
                  {m.measurement_name}
                </span>
                <input
                  onChange={(e) => updateLocalMeasurement(m.id, e.target.value)}
                  value={m.value ?? ""}
                />
                <strong>{formatUnit(m.unit)}</strong>
              </label>
            ))
          )}
        </div>

        {/* 하단 액션 영역 */}
        <div className="button-row" style={{ marginTop: 16, justifyContent: "flex-end", flexWrap: "wrap", gap: 8 }}>
          {isAppliedToHealth && (
            <>
              <Link className="button secondary" to="/health/profile">
                등록된 정보 확인
              </Link>
              <Link className="button secondary" to="/analysis">
                분석 결과 확인
              </Link>
            </>
          )}
          <button
            disabled={measurements.length === 0 || isConfirming || isAppliedToHealth}
            onClick={saveAndConfirm}
            type="button"
          >
            {isConfirming
              ? "검진 결과 등록 중..."
              : isAppliedToHealth
                ? "검진 결과 등록 완료"
                : "검진 결과 등록"}
          </button>
        </div>
      </Card>
    </div>
  );
}

function isPdfFile(file: File): boolean {
  return file.type.toLowerCase() === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
}

function formatUnit(unit: string | null | undefined): string {
  if (!unit) return "-";
  return unit.replace("1.73m2", "1.73m²");
}