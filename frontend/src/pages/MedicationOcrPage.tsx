import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import {
  confirmMedicationOcr,
  type MedicationOcrItem,
  runMedicationDummyOcr,
} from "../api/medications";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

export default function MedicationOcrPage() {
  const [items, setItems] = useState<MedicationOcrItem[]>([]);
  const [sourceType, setSourceType] = useState("PRESCRIPTION");
  const [imageFilename, setImageFilename] = useState("");
  const [isMobileDevice, setIsMobileDevice] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

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

  const runDummyOcr = async () => {
    setError("");
    setMessage("");
    setIsRunning(true);
    try {
      const response = await runMedicationDummyOcr({
        source_type: sourceType,
        image_filename: imageFilename || undefined,
        memo: "MVP medication OCR demo request",
      });
      setItems(response.items);
      setMessage(`${toUserMessage(response.message)} 저장 전 약 이름과 복용 정보를 확인해주세요.`);
    } catch (err) {
      setError(err instanceof Error ? toUserMessage(err.message) : "복약정보 자동 인식에 실패했습니다.");
    } finally {
      setIsRunning(false);
    }
  };

  const updateItem = (index: number, key: keyof MedicationOcrItem, value: string | number | string[] | null) => {
    setItems((prev) => prev.map((item, itemIndex) => (itemIndex === index ? { ...item, [key]: value } : item)));
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
      setMessage(`${response.message} 생성 ${response.created_count}건, 건너뜀 ${response.skipped_count}건`);
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
          <h1>복약/처방전 OCR</h1>
          <p>처방전 또는 약봉투 이미지에서 복약 정보를 자동 인식합니다.</p>
        </div>
        <Link className="button secondary" to="/ocr">
          OCR 선택으로 돌아가기
        </Link>
      </div>
      {error && <ErrorMessage message={error} />}
      {message && <div className="state-box">{message}</div>}
      <div className="page-grid">
        <Card title="처방전/약봉투 업로드">
          <div className="upload-box">
            <strong>이미지 업로드 영역</strong>
            <span>촬영/업로드 후 자동 인식 결과를 확인하고 저장해주세요.</span>
            <div className="upload-action-grid">
              <label className="upload-action-button">
                파일에서 선택
                <input
                  accept="image/*"
                  type="file"
                  onChange={(event) => setImageFilename(event.currentTarget.files?.[0]?.name ?? "")}
                />
              </label>
              {isMobileDevice ? (
                <label className="upload-action-button">
                  카메라로 촬영
                  <input
                    accept="image/*"
                    capture="environment"
                    type="file"
                    onChange={(event) => setImageFilename(event.currentTarget.files?.[0]?.name ?? "")}
                  />
                </label>
              ) : (
                <span className="upload-mobile-hint">카메라 촬영은 모바일에서 사용할 수 있습니다.</span>
              )}
            </div>
            <span className="muted">선택된 파일: {imageFilename || "없음"}</span>
          </div>
          <label>
            인식 유형
            <select className="input" value={sourceType} onChange={(event) => setSourceType(event.target.value)}>
              <option value="PRESCRIPTION">처방전</option>
              <option value="MEDICATION_BAG">약봉투</option>
              <option value="SUPPLEMENT">영양제</option>
            </select>
          </label>
          <button disabled={isRunning} onClick={runDummyOcr} type="button">
            {isRunning ? "자동 인식 실행 중..." : "자동 인식 실행"}
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
      <Card title="자동 인식 결과">
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

function toUserMessage(message: string): string {
  return message.replaceAll("더미 OCR", "자동 인식").replaceAll("더미", "예시").replaceAll("dummy", "demo");
}
