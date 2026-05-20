import { FormEvent, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { createInquiry, listFaqs, listMyInquiries } from "../api/faqs";
import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";

type Item = Record<string, unknown>;

export default function FAQPage() {
  const { isAuthenticated } = useAuth();
  const [faqs, setFaqs] = useState<Item[]>([]);
  const [inquiries, setInquiries] = useState<Item[]>([]);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [keyword, setKeyword] = useState("");
  const [category, setCategory] = useState("전체");

  const load = async () => {
    setFaqs(await listFaqs<Item[]>());
    if (isAuthenticated) {
      setInquiries(await listMyInquiries<Item[]>());
    }
  };

  useEffect(() => {
    void load().catch(() => undefined);
  }, [isAuthenticated]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    await createInquiry({ category: "GENERAL", title, content });
    setTitle("");
    setContent("");
    await load();
  };

  return (
    <div className="page-grid">
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
        <pre>{JSON.stringify(inquiries, null, 2)}</pre>
      </Card>
    </div>
  );
}
