import { useEffect } from "react";

export const occupationHelpItems = [
  ["관리·전문직", "관리자, 전문가, 연구직, 의료/법률/교육 전문직"],
  ["사무직", "회사원, 행정, 회계, 기획, 개발자, 일반 사무"],
  ["서비스·판매직", "매장, 영업, 고객응대, 요식업, 서비스업"],
  ["농림어업", "농업, 임업, 어업 종사자"],
  ["기능·노무직", "생산, 제조, 운전, 건설, 정비, 단순노무"],
  ["주부·학생", "전업주부, 학생"],
  ["무직·기타", "구직 중, 은퇴, 기타"],
] as const;

type OccupationHelpModalProps = {
  onClose: () => void;
};

export default function OccupationHelpModal({ onClose }: OccupationHelpModalProps) {
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  return (
    <div className="dialog-backdrop" onClick={onClose} role="presentation">
      <div
        aria-modal="true"
        className="help-dialog"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
      >
        <div className="help-dialog-header">
          <div>
            <h2>직업군 선택 도움말</h2>
            <p>현재 생활 패턴과 가장 가까운 항목을 선택해주세요.</p>
          </div>
          <button aria-label="직업군 도움말 닫기" className="help-close-button" onClick={onClose} type="button">
            닫기
          </button>
        </div>
        <div className="occupation-help-list">
          {occupationHelpItems.map(([title, description]) => (
            <div className="occupation-help-item" key={title}>
              <strong>{title}</strong>
              <span>{description}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
