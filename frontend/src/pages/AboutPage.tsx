import { useState } from "react";
import { Link } from "react-router-dom";
import Card from "../components/Card";
import { HeartPulse, FileText, Salad, Dumbbell, Pill, BotMessageSquare, ClipboardList, ChartBar, Trophy, Bell, TrendingUp } from "lucide-react";

const aboutFeatures = [
  {
    icon: "🧭",
    title: "AI 위험도 분석",
    description: "건강정보를 기반으로 당뇨, 고혈압, 이상지질혈증 이상 위험도를 확인합니다.",
    to: "/analysis",
  },
  {
    icon: "📄",
    title: "검진표 등록",
    description: "검진표 이미지나 PDF에서 주요 건강 수치를 빠르게 입력합니다.",
    to: "/ocr/exam",
  },
  {
    icon: "🥗",
    title: "식단 이미지 분석",
    description: "식단 사진을 기록하고 영양 요약과 개선 포인트를 확인합니다.",
    to: "/diets",
  },
  {
    icon: "🚶",
    title: "맞춤 챌린지",
    description: "위험도와 생활습관에 맞춘 작은 건강 습관을 실천합니다.",
    to: "/challenges",
  },
  {
    icon: "💊",
    title: "복약/영양제 관리",
    description: "복약 정보와 기록을 한 곳에서 관리합니다.",
    to: "/medications",
  },
  {
    icon: "💬",
    title: "AI 건강 상담",
    description: "건강 분석, 식단, 운동, 복약 관련 질문을 편하게 남깁니다.",
    to: "/chatbot",
  },
];

const aboutPersonas = [
  {
    id: "exam",
    icon: "📋",
    title: "검진 결과가 걱정되는 직장인",
    quote: "검진표는 받았는데 수치가 뭘 의미하는지 모르겠어요.",
    features: ["검진표 등록", "위험도 분석", "AI 코멘트"],
    flow: [
      { icon: "📄", title: "검진표 업로드", description: "촬영하거나 파일로 올립니다." },
      { icon: "📊", title: "위험도 분석", description: "주요 질환 위험도를 확인합니다." },
      { icon: "✅", title: "챌린지 추천", description: "생활습관 액션을 이어갑니다." },
    ],
    to: "/ocr/exam",
    buttonLabel: "검진표 등록하러 가기",
  },
  {
    id: "habit",
    icon: "🚶",
    title: "생활습관을 바꾸고 싶은 사용자",
    quote: "식단과 운동 습관으로 혈당, 혈압, 체중을 관리하고 싶어요.",
    features: ["식단 분석", "챌린지", "건강 리포트"],
    flow: [
      { icon: "🥗", title: "식단 기록", description: "사진과 기본 정보를 입력합니다." },
      { icon: "📈", title: "변화 추적", description: "건강 점수와 추이를 봅니다." },
      { icon: "🚶", title: "챌린지 실천", description: "추천 습관을 시작합니다." },
    ],
    to: "/challenges",
    buttonLabel: "챌린지 시작하기",
  },
  {
    id: "record",
    icon: "💊",
    title: "복약/건강기록을 관리하는 사용자",
    quote: "복약, 영양제, 건강기록을 놓치지 않고 관리하고 싶어요.",
    features: ["복약 정보 등록", "복약 기록", "알림", "AI 상담"],
    flow: [
      { icon: "💊", title: "복약 입력", description: "처방전과 약봉투를 정리합니다." },
      { icon: "🔔", title: "알림 확인", description: "기록과 알림을 관리합니다." },
      { icon: "🤖", title: "AI 상담", description: "궁금한 점을 이어서 묻습니다." },
    ],
    to: "/medications",
    buttonLabel: "복약 관리 시작하기",
  },
];

const featureIcons: Record<string, React.ReactNode> = {
  "🧭": <HeartPulse size={24} />,
  "📄": <FileText size={24} />,
  "🥗": <Salad size={24} />,
  "🚶": <Dumbbell size={24} />,
  "💊": <Pill size={24} />,
  "💬": <BotMessageSquare size={24} />,
};

const flowIcons: Record<string, React.ReactNode> = {
  "📄": <FileText size={24} />,
  "📊": <ChartBar size={24} />,
  "✅": <Trophy size={24} />,
  "🥗": <Salad size={24} />,
  "📈": <TrendingUp size={24} />,
  "🚶": <Dumbbell size={24} />,
  "💊": <Pill size={24} />,
  "🔔": <Bell size={24} />,
  "🤖": <BotMessageSquare size={24} />,
};

const personaIcons: Record<string, React.ReactNode> = {
  exam: <ClipboardList size={24} />,
  habit: <Dumbbell size={24} />,
  record: <Pill size={24} />,
};

export default function AboutPage() {
  const [selectedPersonaId, setSelectedPersonaId] = useState("exam");
  const selectedPersona = aboutPersonas.find((p) => p.id === selectedPersonaId) ?? aboutPersonas[0];

  return (
    <div className="landing-page">
      {/* 히어로 */}
      <section className="hero-panel">
        <div>
          <span className="eyebrow">Health Ladder</span>
          <h1>AI 만성질환 예측과 생활습관 관리까지, 나를 위한 건강 사다리를 시작하세요</h1>
          <p>만성질환 위험도를 예측하고 식단 관리, 복약, 생활습관 챌린지를 스마트하게 케어하세요.</p>
          <div className="button-row">
            <Link className="button" to="/health">
              건강정보 입력하기
            </Link>
            <Link className="button secondary" to="/analysis">
              분석 결과 보기
            </Link>
          </div>
        </div>
        <div className="hero-image-area">
          <img
            src="/images/hero-preview.png"
            alt="서비스 미리보기"
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              borderRadius: "var(--border-radius-lg)",
              display: "block",
            }}
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        </div>
      </section>

      {/* 기능 소개 */}
      <section className="landing-section" id="features">
        <div className="section-heading">
          <h2>제공 기능</h2>
          <p>Health Ladder에서 사용할 수 있는 기능을 확인하고 바로 이용해보세요.</p>
        </div>
        <div className="landing-feature-grid">
          {aboutFeatures.map((feature) => (
            <Link
              className="landing-feature-card"
              key={feature.title}
              to={feature.to}
            >
              <em className="badge badge-saved" style={{ position: "absolute", top: "12px", right: "12px" }}>
                바로 이용
              </em>
              <span className="landing-feature-icon">{featureIcons[feature.icon] ?? feature.icon}</span>
              <strong>{feature.title}</strong>
              <p>{feature.description}</p>
            </Link>
          ))}
        </div>
      </section>

      {/* 내 상황에 맞는 시작 방법 */}
      <section className="landing-section">
        <div className="section-heading">
          <h2>나에게 맞는 건강 관리 시작하기</h2>
          <p>현재 나의 상황과 가장 비슷한 항목을 선택하면 추천 흐름을 안내해드립니다.</p>
        </div>
        <div className="persona-grid">
          {aboutPersonas.map((persona) => (
            <button
              className={`persona-card ${persona.id === selectedPersona.id ? "active" : ""}`}
              key={persona.id}
              type="button"
              onClick={() => setSelectedPersonaId(persona.id)}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span className="persona-icon">{personaIcons[persona.id]}</span>
                <strong>{persona.title}</strong>
              </div>
              <p>{persona.quote}</p>
              <em>{persona.features.join(" · ")}</em>
            </button>
          ))}
        </div>
      </section>

      {/* 추천 흐름 */}
      <section className="persona-flow-panel">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <span className="eyebrow">추천 흐름</span>
            <h2>{selectedPersona.title}</h2>
            <p>{selectedPersona.quote}</p>
          </div>
          <div className="button-row">
            <Link className="button" to={selectedPersona.to}>
              {selectedPersona.buttonLabel}
            </Link>
            <Link className="button secondary" to="/">
              홈으로 돌아가기
            </Link>
          </div>
        </div>
        <div className="persona-timeline">
          {selectedPersona.flow.map((step, index) => (
            <div className="persona-timeline-item" key={step.title}>
              <span className="timeline-icon">{flowIcons[step.icon] ?? step.icon}</span>
              <div>
                <strong>{step.title}</strong>
                <p>{step.description}</p>
              </div>
              {index < selectedPersona.flow.length - 1 ? <em aria-hidden="true">→</em> : null}
            </div>
          ))}
        </div>

      </section>

      {/* 하단 CTA */}
      <section className="landing-cta">
        <div>
          <h2>지금 바로 건강 관리를 시작하세요.</h2>
          <p>기본 건강정보를 입력하면 만성질환 위험도 예측과 건강 챌린지 서비스를 바로 이용할 수 있습니다.</p>
        </div>
        <div className="button-row">
          <Link className="button" to="/health">
            건강정보 입력하기
          </Link>
          <Link className="button secondary" to="/faqs">
            FAQ 보기
          </Link>
        </div>
      </section>
    </div>
  );
}
