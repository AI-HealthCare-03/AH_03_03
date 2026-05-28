import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import Card from "../components/Card";

export default function OcrPage() {
  const [isMobileDevice, setIsMobileDevice] = useState(false);

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

  return (
    <div className="page-stack">
      <div className="page-header">
        <div>
          <h1>검진·복약 등록</h1>
          <p>건강검진표 또는 복약/처방전 이미지를 인식해 건강정보를 빠르게 입력할 수 있습니다.</p>
        </div>
      </div>
      <div className="state-box warning-card">자동 인식 결과는 저장 전 반드시 직접 확인해주세요.</div>
      <div className="state-box">
        {isMobileDevice
          ? "모바일에서는 파일 선택과 카메라 촬영을 모두 사용할 수 있습니다."
          : "PC에서는 파일 선택을 기본으로 사용합니다. 카메라 촬영은 모바일에서 사용할 수 있습니다."}{" "}
        촬영/업로드 후 자동 인식 결과를 확인하고 저장해주세요.
      </div>
      <div className="page-grid">
        <Card title="검진표 등록">
          <div className="ocr-card-icon">📄</div>
          <p>혈압, 혈당, 콜레스테롤, 중성지방, HDL(좋은), LDL(나쁜), HbA1c 등을 추출합니다.</p>
          <Link className="button" to="/ocr/exam">
            검진표 등록하기
          </Link>
        </Card>
        <Card title="복약 정보 등록">
          <div className="ocr-card-icon">💊</div>
          <p>약 이름, 복용 시간, 복용 횟수, 복용 기간을 추출합니다.</p>
          <Link className="button" to="/ocr/medication">
            복약 정보 등록하기
          </Link>
        </Card>
      </div>
    </div>
  );
}
