import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { listFaqs } from "../api/faqs";
import Card from "../components/Card";

type Item = Record<string, unknown>;

export default function FAQPage() {
  const [faqs, setFaqs] = useState<Item[]>([]);
  const [keyword, setKeyword] = useState("");
  const [faqCategory, setFaqCategory] = useState("전체");

  useEffect(() => {
    void listFaqs<Item[]>().then(setFaqs).catch(() => undefined);
  }, []);

  return (
    <div className="page-grid">
      <Card title="FAQ">
        <div style={{ marginBottom: "12px" }}>
          <p className="muted" style={{ marginBottom: "6px" }}>FAQ 검색</p>
          <input
            className="search-input"
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            placeholder="키워드 입력"
            style={{ width: "100%", padding: "8px 12px", borderRadius: "8px", border: "1px solid var(--color-border)" }}
          />
        </div>
        <div className="filter-tabs" style={{ marginTop: "12px" }}>
          {["전체", "회원/로그인", "건강분석", "챌린지", "식단/복약", "개인정보/보안"].map((item) => (
            <button
              className={faqCategory === item ? "filter-tab active" : "filter-tab"}
              key={item}
              onClick={() => setFaqCategory(item)}
            >
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
              <details className="mini-card faq-item" key={String(faq.id)}>
                <summary className="faq-summary">
                  <strong>{String(faq.question)}</strong>
                  <span className="faq-summary-hint">펼치기</span>
                </summary>
                <p className="faq-answer">{String(faq.answer)}</p>
              </details>
            ))}
          {faqs.length === 0 && <div className="state-box">등록된 FAQ가 없습니다.</div>}
        </div>
      </Card>
      <Card title="더 궁금한 게 있으신가요?">
        <p className="muted">FAQ에서 해결되지 않은 문의는 1:1 문의로 남겨주세요.</p>
        <Link className="button" to="/inquiries/new" style={{ marginTop: "12px", display: "inline-block" }}>
          1:1 문의하기
        </Link>
      </Card>
    </div>
  );
}
