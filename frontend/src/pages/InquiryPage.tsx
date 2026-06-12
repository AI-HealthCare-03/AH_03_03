import { FormEvent, useEffect, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

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

  return (
    <div className="page-grid">
      <Card
        title={isNew ? "1:1 문의 작성" : "1:1 문의"}
        actions={
          <Link className="button" to={isNew ? "/inquiries" : "/inquiries/new"}>
            {isNew ? "문의 목록" : "새 문의"}
          </Link>
        }
      >
        <div className="filter-tabs">
          <Link className={!isNew ? "filter-tab active" : "filter-tab"} to="/inquiries">
            상담 내역
          </Link>
          <Link className={isNew ? "filter-tab active" : "filter-tab"} to="/inquiries/new">
            문의 작성
          </Link>
        </div>
        {isNew ? (
          <form className="form" onSubmit={submit}>
            {error && <ErrorMessage message={error} />}
            <label>
              문의 유형
              <select value={category} onChange={(event) => setCategory(event.target.value)}>
                <option value="GENERAL">일반 문의</option>
                <option value="HEALTH">건강 분석</option>
                <option value="ACCOUNT">계정</option>
                <option value="ETC">기타</option>
              </select>
            </label>
            <label>
              제목
              <input value={title} onChange={(event) => setTitle(event.target.value)} required />
            </label>
            <label>
              내용
              <textarea value={content} onChange={(event) => setContent(event.target.value)} required />
            </label>
            <label className="toggle-row">
              <span>답변 알림 받기</span>
              <input defaultChecked type="checkbox" />
            </label>
            <button type="submit">문의 등록</button>
          </form>
        ) : (
          <div className="table-list">
            {items.map((item) => (
              <div className="table-row" key={String(item.id)}>
                <span>{String(item.category)}</span>
                <strong>{String(item.title)}</strong>
                <span>{getInquiryStatusLabel(item.status)}</span>
              </div>
            ))}
            {items.length === 0 && <p className="placeholder">등록된 문의가 없습니다.</p>}
          </div>
        )}
      </Card>
      <Card title="답변 안내">
        <p>문의 답변은 관리자 확인 후 상태가 변경됩니다.</p>
        <p className="muted">자주 묻는 질문을 먼저 확인하면 더 빠르게 해결할 수 있습니다.</p>
      </Card>
      <Card title="FAQ">
        <div style={{ marginBottom: "12px" }}>
          <p className="muted" style={{ marginBottom: "6px" }}>FAQ 검색</p>
          <input className="search-input" value={keyword} onChange={(event) => setKeyword(event.target.value)} placeholder="키워드 입력" style={{ width: "100%", padding: "8px 12px", borderRadius: "8px", border: "1px solid var(--color-border)" }} />
        </div>
        <div className="filter-tabs" style={{ marginTop: "12px" }}>
          {["전체", "회원/로그인", "건강분석", "챌린지", "식단/복약", "개인정보/보안"].map((item) => (
            <button className={faqCategory === item ? "filter-tab active" : "filter-tab"} key={item} onClick={() => setFaqCategory(item)}>
              {item}
            </button>
          ))}
        </div>
        <div className="card-list" style={{ marginTop: "12px" }}>
          {faqs
            .filter((faq) =>
              `${String(faq.question)} ${String(faq.answer)}`.toLowerCase().includes(keyword.toLowerCase()),
            )
            .filter((faq) => faqCategory === "전체" || String(faq.category) === faqCategory)
            .map((faq) => (
              <details className="mini-card" key={String(faq.id)}>
                <summary>
                  <strong>{String(faq.question)}</strong>
                </summary>
                <p>{String(faq.answer)}</p>
              </details>
            ))}
          {faqs.length === 0 && <div className="state-box">등록된 FAQ가 없습니다.</div>}
        </div>
      </Card>
    </div>
  );
}
