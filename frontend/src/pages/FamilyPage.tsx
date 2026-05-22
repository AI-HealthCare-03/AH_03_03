import { useState } from "react";

import Card from "../components/Card";

const summaryItems = [
  { label: "연결된 가족", value: "0명", helper: "아직 연결된 가족이 없습니다." },
  { label: "초대 대기", value: "0건", helper: "대기 중인 초대가 없습니다." },
  { label: "공유 알림", value: "0건", helper: "공유된 건강 알림이 없습니다." },
  { label: "이상 수치 알림", value: "0건", helper: "확인할 이상 수치 알림이 없습니다." },
];

const shareSettings = [
  "건강기록 공유",
  "분석결과 공유",
  "복약정보 공유",
  "식단기록 공유",
  "검진표/OCR 결과 공유",
  "이상 수치 알림 받기",
];

const familyAlerts = ["가족 건강분석 결과 알림", "혈압/혈당 이상 알림", "복약 미수행 알림", "챌린지 미수행 알림"];

export default function FamilyPage() {
  const [notice, setNotice] = useState("");

  const showUpcomingNotice = () => {
    setNotice("가족 연결 기능은 다음 단계에서 활성화됩니다.");
  };

  return (
    <div className="page-stack">
      <div className="page-header">
        <div>
          <span className="badge badge-reference">Full Service</span>
          <h1>가족 관리</h1>
          <p>가족과 건강 정보를 안전하게 공유하고, 보호자 알림을 관리합니다.</p>
        </div>
      </div>

      {notice && <div className="state-box">{notice}</div>}

      <div className="metric-grid">
        {summaryItems.map((item) => (
          <div className="metric-card" key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
            <p className="muted">{item.helper}</p>
          </div>
        ))}
      </div>

      <div className="family-section-grid">
        <Card title="가족 목록">
          <div className="empty-state">
            <strong>아직 연결된 가족이 없습니다.</strong>
            <p>가족을 연결하면 건강정보 공유와 보호자 알림을 설정할 수 있습니다.</p>
          </div>
          <div className="button-row" style={{ marginTop: 16 }}>
            <button onClick={showUpcomingNotice} type="button">
              가족 초대하기
            </button>
            <button className="secondary" onClick={showUpcomingNotice} type="button">
              미가입 가족 등록
            </button>
          </div>
        </Card>

        <Card title="가족 초대">
          <div className="family-invite-grid">
            <div className="mini-card">
              <span className="badge badge-reference">초대 코드</span>
              <strong>초대 코드 생성</strong>
              <p className="muted">코드로 가족을 연결할 수 있습니다.</p>
              <button className="secondary" onClick={showUpcomingNotice} type="button">
                코드 생성
              </button>
            </div>
            <div className="mini-card">
              <span className="badge badge-reference">코드 입력</span>
              <strong>초대 코드로 연결</strong>
              <p className="muted">받은 초대 코드를 입력해 가족과 연결합니다.</p>
              <button className="secondary" onClick={showUpcomingNotice} type="button">
                코드 입력
              </button>
            </div>
            <div className="mini-card">
              <span className="badge badge-reference">검색 초대</span>
              <strong>이메일/휴대폰으로 초대</strong>
              <p className="muted">가입한 가족을 찾아 초대합니다.</p>
              <button className="secondary" onClick={showUpcomingNotice} type="button">
                초대 보내기
              </button>
            </div>
          </div>
        </Card>

        <Card title="공유 권한">
          <p className="muted">가족 연결 후 공유 범위를 선택할 수 있습니다.</p>
          <div className="settings-list">
            {shareSettings.map((setting) => (
              <label className="toggle-row family-toggle-row" key={setting}>
                <span>{setting}</span>
                <input disabled type="checkbox" />
              </label>
            ))}
          </div>
        </Card>

        <Card title="가족 알림">
          <div className="card-list">
            {familyAlerts.map((alert) => (
              <div className="mini-card family-alert-row" key={alert}>
                <div>
                  <strong>{alert}</strong>
                  <p className="muted">아직 표시할 알림이 없습니다.</p>
                </div>
                <span className="badge badge-reference">0건</span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

