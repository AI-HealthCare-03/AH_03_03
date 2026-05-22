import { FormEvent, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { createInquiry, listFaqs, listMyInquiries } from "../api/faqs";
import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

type Item = Record<string, unknown>;

export default function FAQPage() {
  const { isAuthenticated } = useAuth();
  const [faqs, setFaqs] = useState<Item[]>([]);
  const [inquiries, setInquiries] = useState<Item[]>([]);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [keyword, setKeyword] = useState("");
  const [category, setCategory] = useState("전체");
  const [error, setError] = useState("");

  const formatDate = (value: unknown) => {
    if (!value) {
      return "-";
    }
    const date = new Date(String(value));
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleDateString("ko-KR");
  };

  const load = async () => {
    setError("");
    try {
      setFaqs(await listFaqs<Item[]>());
      if (isAuthenticated) {
        setInquiries(await listMyInquiries<Item[]>());
      } else {
        setInquiries([]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "FAQ 또는 문의 내역을 불러오지 못했습니다.");
    }
  };

  useEffect(() => {
    void load();
  }, [isAuthenticated]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    try {
      await createInquiry({ category: "GENERAL", title, content });
      setTitle("");
      setContent("");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "문의 등록에 실패했습니다.");
    }
  };

  return (
    <div className="page-grid">
      {error && <ErrorMessage message={error} />}
      <Card title="FAQ">
        <label className="search-box">
          FAQ 검색
          <input value={keyword} onChange={(event) => setKeyword(event.target.value)} placeholder="키워드 입력" />
        </label>
        <div className="filter-tabs">
          {["전체", "회원/로그인", "건강분석", "챌린지", "식단/복약", "개인정보/보안"].map((item) => (
            <button className={category === item ? "filter-tab active" : "filter-tab"} key={item} onClick={() => setCategory(item)}>
              {item}
            </button>
          ))}
        </div>
        <div className="card-list">
          {faqs
            .filter((faq) =>
              `${String(faq.question)} ${String(faq.answer)}`.toLowerCase().includes(keyword.toLowerCase()),
            )
            .filter((faq) => category === "전체" || String(faq.category) === category)
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
      <Card
        title="문의 작성"
        actions={
          <Link className="button secondary" to="/inquiries">
            1:1 문의 화면
          </Link>
        }
      >
        <form className="form" onSubmit={submit}>
          <input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="제목" required />
          <textarea value={content} onChange={(event) => setContent(event.target.value)} placeholder="내용" required />
          <button type="submit" disabled={!isAuthenticated}>
            문의 등록
          </button>
        </form>
      </Card>
      <Card title="내 문의">
        <div className="card-list">
          {!isAuthenticated && <div className="state-box">로그인 후 내 문의 내역을 확인할 수 있습니다.</div>}
          {isAuthenticated && inquiries.length === 0 && <div className="state-box">등록된 문의가 없습니다.</div>}
          {inquiries.map((inquiry) => (
            <div className="mini-card" key={String(inquiry.id)}>
              <div className="record-row">
                <div>
                  <strong>{String(inquiry.title ?? "문의")}</strong>
                  <p className="muted">{String(inquiry.category ?? "GENERAL")}</p>
                </div>
                <span className="badge badge-reference">{String(inquiry.status ?? "PENDING")}</span>
              </div>
              <div className="record-meta-grid">
                <span>작성일 {formatDate(inquiry.created_at)}</span>
                <span>{inquiry.answer ? "답변 완료" : "답변 대기"}</span>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
