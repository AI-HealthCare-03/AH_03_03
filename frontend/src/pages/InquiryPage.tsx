import { FormEvent, useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { createInquiry, listFaqs, listMyInquiries } from "../api/faqs";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

type Inquiry = Record<string, unknown>;
type Item = Record<string, unknown>;

const inquiryStatusLabels: Record<string, string> = {
  PENDING: "답변 대기",
  ANSWERED: "답변 완료",
  CLOSED: "종료",
  CANCELED: "취소됨",
  CANCELLED: "취소됨",
};

function getInquiryStatusLabel(value: unknown): string {
  const status = String(value ?? "").trim().toUpperCase();
  return inquiryStatusLabels[status] ?? "상태 확인 중";
}

const inquiryCategoryLabels: Record<string, string> = {
  GENERAL: "일반 문의",
  HEALTH: "건강 분석",
  ACCOUNT: "계정",
  ETC: "기타",
};

function getInquiryCategoryLabel(value: unknown): string {
  return inquiryCategoryLabels[String(value ?? "").trim().toUpperCase()] ?? String(value ?? "-");
}

function getStatusBadgeClass(value: unknown): string {
  const status = String(value ?? "").trim().toUpperCase();
  if (status === "ANSWERED") return "badge badge-saved";
  if (status === "PENDING") return "badge badge-required";
  return "badge badge-missing";
}

const FAQ_CATEGORIES = ["전체", "회원/로그인", "건강분석", "챌린지", "식단/복약", "개인정보/보안"];

export default function InquiryPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const isNew = location.pathname.endsWith("/new");
  const [items, setItems] = useState<Inquiry[]>([]);
  const [category, setCategory] = useState("GENERAL");
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [error, setError] = useState("");
  const [faqs, setFaqs] = useState<Item[]>([]);
  const [keyword, setKeyword] = useState("");
  const [faqCategory, setFaqCategory] = useState("전체");

  const load = async () => {
    setItems(await listMyInquiries<Inquiry[]>());
    setFaqs(await listFaqs<Item[]>());
  };

  useEffect(() => {
    void load().catch(() => undefined);
  }, []);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    try {
      await createInquiry({ category, title, content });
      setTitle("");
      setContent("");
      await load();
      navigate("/inquiries");
    } catch (err) {
      setError(err instanceof Error ? err.message : "문의 등록에 실패했습니다.");
    }
  };

  const filteredFaqs = faqs
    .filter((faq) =>
      `${String(faq.question)} ${String(faq.answer)}`
        .toLowerCase()
        .includes(keyword.toLowerCase()),
    )
    .filter((faq) => faqCategory === "전체" || String(faq.category) === faqCategory);

  return (
    <div className="page-stack">
      {/* 헤더 */}
      <div className="page-header">
        <div>
          <h1>고객센터</h1>
          <p>문의하기 및 자주 묻는 질문을 확인하세요.</p>
        </div>
      </div>

      {/* 안내 notice */}
      <div className="notice-box">
        <i className="ti ti-info-circle" aria-hidden="true" />
        문의 답변은 관리자 확인 후 상태가 변경됩니다. FAQ를 먼저 확인하면 더 빠르게 해결할 수 있습니다.
      </div>

      {/* ── 1:1 문의 ── */}
      <Card
        title="1:1 문의"
        actions={
          <button
            className="button"
            onClick={() => navigate(isNew ? "/inquiries" : "/inquiries/new")}
            type="button"
          >
            {isNew ? "문의 목록" : "+ 새 문의"}
          </button>
        }
      >
        <div className="filter-tabs" style={{ marginBottom: 16 }}>
          <button
            className={!isNew ? "filter-tab active" : "filter-tab"}
            onClick={() => navigate("/inquiries")}
            type="button"
          >
            상담 내역
          </button>
          <button
            className={isNew ? "filter-tab active" : "filter-tab"}
            onClick={() => navigate("/inquiries/new")}
            type="button"
          >
            문의 작성
          </button>
        </div>

        {isNew ? (
          <form className="form" onSubmit={submit}>
            {error && <ErrorMessage message={error} />}
            <label>
              문의 유형
              <select value={category} onChange={(e) => setCategory(e.target.value)}>
                <option value="GENERAL">일반 문의</option>
                <option value="HEALTH">건강 분석</option>
                <option value="ACCOUNT">계정</option>
                <option value="ETC">기타</option>
              </select>
            </label>
            <label>
              제목
              <input value={title} onChange={(e) => setTitle(e.target.value)} required />
            </label>
            <label>
              내용
              <textarea value={content} onChange={(e) => setContent(e.target.value)} required />
            </label>
            <label className="toggle-row">
              <span>답변 알림 받기</span>
              <input defaultChecked type="checkbox" />
            </label>
            <button type="submit">문의 등록</button>
          </form>
        ) : (
          <div className="table-list">
            {items.length === 0 && (
              <p className="placeholder">등록된 문의가 없습니다.</p>
            )}
            {items.map((item) => (
              <details className="mini-card" key={String(item.id)}>
                <summary style={{ display: "flex", alignItems: "center", gap: 12, cursor: "pointer", listStyle: "none" }}>
                  <span className="muted" style={{ fontSize: 12, minWidth: 64, flexShrink: 0 }}>
                    {getInquiryCategoryLabel(item.category)}
                  </span>
                  <strong style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {String(item.title)}
                  </strong>
                  <span className={getStatusBadgeClass(item.status)} style={{ flexShrink: 0 }}>
                    {getInquiryStatusLabel(item.status)}
                  </span>
                </summary>
                <div style={{ marginTop: 12, paddingTop: 12, borderTop: "0.5px solid var(--color-border)", fontSize: 13, color: "var(--color-text-secondary)", lineHeight: 1.6 }}>
                  <p style={{ marginBottom: 6 }}><strong style={{ color: "var(--color-text-primary)" }}>문의 내용</strong></p>
                  <p>{String(item.content ?? "내용 없음")}</p>
                  {item.answer && (
                    <div style={{ marginTop: 12, padding: "10px 14px", background: "var(--color-muted-surface)", borderRadius: "var(--border-radius-md)" }}>
                      <p style={{ marginBottom: 4 }}><strong style={{ color: "var(--color-text-primary)" }}>답변</strong></p>
                      <p>{String(item.answer)}</p>
                    </div>
                  )}
                </div>
              </details>
            ))}
          </div>
        )}
      </Card>

      {/* ── FAQ ── */}
      <Card title="FAQ">
        <input
          className="search-input"
          placeholder="키워드로 검색..."
          style={{ width: "100%", marginBottom: 12 }}
          value={keyword}
          onChange={(e) => setKeyword(e.target.value)}
        />
        <div className="pill-tabs" style={{ marginBottom: 12 }}>
          {FAQ_CATEGORIES.map((cat) => (
            <button
              className={faqCategory === cat ? "pill-tab active" : "pill-tab"}
              key={cat}
              onClick={() => setFaqCategory(cat)}
              type="button"
            >
              {cat}
            </button>
          ))}
        </div>
        <div className="card-list">
          {filteredFaqs.length === 0 && (
            <div className="state-box">등록된 FAQ가 없습니다.</div>
          )}
          {filteredFaqs.map((faq) => (
            <details className="mini-card" key={String(faq.id)}>
              <summary>
                <strong>{String(faq.question)}</strong>
              </summary>
              <p style={{ marginTop: 8, color: "var(--color-text-secondary)", fontSize: 13, lineHeight: 1.6 }}>
                {String(faq.answer)}
              </p>
            </details>
          ))}
        </div>
      </Card>
    </div>
  );
}