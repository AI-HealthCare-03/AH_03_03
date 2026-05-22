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
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

export default function ExamOcrPage() {
  const [selectedFileName, setSelectedFileName] = useState("health-exam-sample.pdf");
  const [isMobileDevice, setIsMobileDevice] = useState(false);
  const [exam, setExam] = useState<ExamReport | null>(null);
  const [measurements, setMeasurements] = useState<ExamMeasurement[]>([]);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

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

  const startExamOcr = async () => {
    setError("");
    setMessage("");
    try {
      const report =
        exam ??
        (await createExam({
          original_filename: selectedFileName,
          file_path: `exam-upload/${selectedFileName}`,
          uploaded_at: new Date().toISOString(),
        }));
      setExam(report);
      const result = await runExamOcr(report.id);
      setMeasurements(result.measurements);
      setMessage(toUserMessage(result.message));
    } catch (err) {
      setError(err instanceof Error ? toUserMessage(err.message) : "자동 인식 실행에 실패했습니다.");
    }
  };

  const updateLocalMeasurement = (measurementId: number, value: string) => {
    setMeasurements((prev) =>
      prev.map((measurement) => (measurement.id === measurementId ? { ...measurement, value } : measurement)),
    );
  };

  const saveAndConfirm = async () => {
    if (!exam) {
      setError("먼저 자동 인식을 실행해주세요.");
      return;
    }
    setError("");
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
      setMessage("OCR 결과를 확인 처리했습니다. 건강정보에 반영하려면 필수 건강정보 관리 화면에서 값을 확인해주세요.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "OCR 결과 저장에 실패했습니다.");
    }
  };

  return (
    <div className="page-stack">
      <div className="page-header">
        <div>
          <h1>건강검진표 OCR</h1>
          <p>검진표 이미지/PDF에서 주요 건강 측정값을 자동 인식합니다.</p>
        </div>
        <Link className="button secondary" to="/ocr">
          OCR 선택으로 돌아가기
        </Link>
      </div>
      {error && <ErrorMessage message={error} />}
      {message && <div className="state-box">{message}</div>}
      <div className="page-grid">
        <Card title="파일 업로드">
          <div className="upload-box">
            <strong>검진표 이미지/PDF 업로드</strong>
            <span>촬영/업로드 후 자동 인식 결과를 확인하고 저장해주세요.</span>
            <div className="upload-action-grid">
              <label className="upload-action-button">
                파일에서 선택
                <input
                  accept="image/*,.pdf"
                  onChange={(event) => setSelectedFileName(event.target.files?.[0]?.name ?? selectedFileName)}
                  type="file"
                />
              </label>
              {isMobileDevice ? (
                <label className="upload-action-button">
                  카메라로 촬영
                  <input
                    accept="image/*"
                    capture="environment"
                    onChange={(event) => setSelectedFileName(event.target.files?.[0]?.name ?? selectedFileName)}
                    type="file"
                  />
                </label>
              ) : (
                <span className="upload-mobile-hint">카메라 촬영은 모바일에서 사용할 수 있습니다.</span>
              )}
            </div>
            <span className="muted">선택된 파일: {selectedFileName || "없음"}</span>
          </div>
          <button onClick={startExamOcr} type="button">
            자동 인식 실행
          </button>
        </Card>
        <Card title="저장 전 확인">
          <p className="warning-text">자동 인식 결과입니다. 값과 단위를 확인한 뒤 저장해주세요.</p>
          <button disabled={measurements.length === 0} onClick={saveAndConfirm} type="button">
            확인/저장
          </button>
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
      <Card title="OCR 결과 측정값">
        <div className="ocr-result-table">
          {measurements.length === 0 && <div className="state-box">아직 OCR 결과가 없습니다.</div>}
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
        </div>
      </Card>
    </div>
  );
}

function toUserMessage(message: string): string {
  return message;
}
