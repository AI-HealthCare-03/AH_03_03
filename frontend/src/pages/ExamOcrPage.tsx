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
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [feedbackDialog, setFeedbackDialog] = useState<FeedbackDialog | null>(null);
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

  useAsyncJobPolling({
    jobId: ocrJobId,
    enabled: isRunningOcr && ocrJobId !== null,
    intervalMs: 1500,
    timeoutMs: 120000,
    onSuccess: async () => {
      if (!exam) {
        setFeedbackDialog({
          title: "검진표 인식에 실패했습니다.",
          message: "이미지를 다시 확인한 뒤 업로드해 주세요.",
          tone: "danger",
        });
        setIsRunningOcr(false);
        setOcrJobId(null);
        return;
      }
      try {
        const latestMeasurements = await listMeasurements(exam.id);
        setMeasurements(latestMeasurements);
        setCanRetryOcr(false);
        setFeedbackDialog({
          title: "검진표 인식이 완료되었습니다.",
          message: "인식된 항목을 확인하고 필요한 경우 수정해 주세요.",
        });
      } catch {
        setFeedbackDialog({
          title: "검진표 인식에 실패했습니다.",
          message: "이미지를 다시 확인한 뒤 업로드해 주세요.",
          tone: "danger",
        });
        setCanRetryOcr(true);
      } finally {
        setIsRunningOcr(false);
        setOcrJobId(null);
      }
    },
    onFailure: (job) => {
      setFeedbackDialog({
        title: "검진표 인식에 실패했습니다.",
        message: job.status === "CANCELED" ? "검진표 인식 작업이 취소되었습니다." : "이미지를 다시 확인한 뒤 업로드해 주세요.",
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
    if (selectedPreviewUrl) {
      URL.revokeObjectURL(selectedPreviewUrl);
    }
    setSelectedPreviewUrl("");
    setPreviewMessage("");
    if (!file) {
      return;
    }
    setSelectedFile(file);
    setSelectedFileName(file.name);
    if (isPdfFile(file)) {
      return;
    }
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
    setMessage("");
    setMeasurements([]);
    setCanRetryOcr(false);
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
      setMessage("");
      setOcrJobId(job.id);
    } catch (err) {
      setError("분석 요청을 시작하지 못했습니다. 파일을 확인한 뒤 다시 시도해주세요.");
      setCanRetryOcr(true);
      setIsRunningOcr(false);
    }
  };

  const updateLocalMeasurement = (measurementId: number, value: string) => {
    setMeasurements((prev) =>
      prev.map((measurement) => (measurement.id === measurementId ? { ...measurement, value } : measurement)),
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
        measurements.map((measurement) =>
          updateMeasurement(measurement.id, {
            value: measurement.value,
            unit: measurement.unit,
            is_user_confirmed: true,
          }),
        ),
      );
      await confirmExam(exam.id);
      setMeasurements(await listMeasurements(exam.id));
      setMessage("건강정보에 반영되었습니다.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "건강정보 반영에 실패했습니다.");
    } finally {
      setIsConfirming(false);
    }
  };

  return (
    <div className="page-stack">
      <div className="page-header">
        <div>
          <h1>건강검진표 측정값 확인</h1>
          <p>검진표 이미지/PDF 기반 측정값 후보를 생성하고 확인 후 건강정보에 반영합니다.</p>
        </div>
        <Link className="button secondary" to="/ocr">
          등록 선택으로 돌아가기
        </Link>
      </div>
      {error && <ErrorMessage message={error} />}
      {canRetryOcr ? (
        <div className="button-row">
          <button disabled={isRunningOcr} onClick={startExamOcr} type="button">
            다시 시도
          </button>
        </div>
      ) : null}
      {message && <div className="state-box">{message}</div>}
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
      <div className="page-grid">
        <Card title="파일 업로드">
          <div className="upload-box">
            <strong>검진표 이미지/PDF 업로드</strong>
            <span>촬영/업로드 후 생성된 후보 값을 확인하고 저장해주세요.</span>
            <p className="warning-text">
              업로드한 이미지는 건강정보 추출 및 분석을 위해 사용됩니다. 자동 인식 결과는 오류가 있을 수 있으므로
              저장 또는 분석 전에 내용을 확인해주세요. 건강검진 결과는 민감한 건강정보일 수 있으므로 본인 자료만
              업로드해주세요.
            </p>
            <div className="upload-action-grid">
              <label className="upload-action-button">
                파일에서 선택
                <input
                  accept="image/*,.heic,.heif,.pdf"
                  onChange={(event) => handleFileSelection(event.target.files?.[0] ?? null)}
                  type="file"
                />
              </label>
              {isMobileDevice ? (
                <label className="upload-action-button">
                  카메라로 촬영
                  <input
                    accept="image/*,.heic,.heif"
                    capture="environment"
                    onChange={(event) => handleFileSelection(event.target.files?.[0] ?? null)}
                    type="file"
                  />
                </label>
              ) : (
                <span className="upload-mobile-hint">카메라 촬영은 모바일에서 사용할 수 있습니다.</span>
              )}
            </div>
            <span className="muted">선택된 파일: {selectedFileName || "없음"}</span>
            {previewMessage ? (
              <div className="state-box heic-preview-notice">
                {previewMessage}
              </div>
            ) : null}
            {selectedPreviewUrl ? (
              <img alt="선택한 검진표 이미지 미리보기" className="upload-preview" src={selectedPreviewUrl} />
            ) : null}
          </div>
          <button disabled={isRunningOcr} onClick={startExamOcr} type="button">
            {isRunningOcr ? "검진표 분석 중..." : "측정값 후보 생성"}
          </button>
        </Card>
        <Card title="저장 전 확인">
          <p className="warning-text">자동 인식으로 생성된 후보값입니다. 값과 단위를 확인한 뒤 저장해주세요.</p>
          <p className="warning-text">
            확인/저장 시 아래 인식 후보값이 최신 건강정보에 반영됩니다. 기존에 직접 입력한 건강정보와 다를 수
            있으므로, 검진일 기준 수치가 맞는지 확인해주세요.
          </p>
          <div className="button-row" style={{ marginTop: 12 }}>
            <Link className="button secondary" to="/health/profile">
              건강정보 확인
            </Link>
            <Link className="button secondary" to="/analysis">
              분석 화면 이동
            </Link>
          </div>
        </Card>
      </div>
      <Card title="측정값 후보">
        <div className="ocr-result-table">
          {measurements.length === 0 && <div className="state-box">아직 측정값 후보가 없습니다.</div>}
          {measurements.map((measurement) => (
            <label className="ocr-result-row" key={measurement.id}>
              <span>
                {measurement.measurement_name}
                <em className="badge badge-required">확인 필요</em>
              </span>
              <input
                onChange={(event) => updateLocalMeasurement(measurement.id, event.target.value)}
                value={measurement.value ?? ""}
              />
              <strong>{measurement.unit ?? "-"}</strong>
            </label>
          ))}
          <div className="state-box">
            <p className="warning-text">
              확인/저장 시 아래 인식 후보값이 최신 건강정보에 반영됩니다. 기존에 직접 입력한 건강정보와 다를 수
              있으므로, 검진일 기준 수치가 맞는지 확인해주세요.
            </p>
            {measurements.length === 0 ? (
              <p className="muted">측정값 후보가 생성되면 건강정보 반영 버튼을 사용할 수 있습니다.</p>
            ) : null}
            <button disabled={measurements.length === 0 || isConfirming} onClick={saveAndConfirm} type="button">
              {isConfirming ? "건강정보에 반영 중..." : "선택한 후보값을 건강정보에 반영"}
            </button>
          </div>
        </div>
      </Card>
    </div>
  );
}

function isPdfFile(file: File): boolean {
  return file.type.toLowerCase() === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
}
