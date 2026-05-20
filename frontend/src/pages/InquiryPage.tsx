import { FormEvent, useEffect, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

import { createInquiry, listMyInquiries } from "../api/faqs";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

type Inquiry = Record<string, unknown>;

export default function InquiryPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const isNew = location.pathname.endsWith("/new");
  const [items, setItems] = useState<Inquiry[]>([]);
  const [category, setCategory] = useState("GENERAL");
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [error, setError] = useState("");

  const load = async () => setItems(await listMyInquiries<Inquiry[]>());

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
            <label>
              첨부파일
              <input disabled type="file" />
              <span className="muted">파일 첨부는 후속 구현 예정입니다.</span>
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
                <span>{String(item.status)}</span>
              </div>
            ))}
            {items.length === 0 && <p className="placeholder">등록된 문의가 없습니다.</p>}
          </div>
        )}
      </Card>
      <Card title="답변 안내">
        <p>문의 답변은 관리자 확인 후 상태가 변경됩니다.</p>
        <p className="placeholder">실시간 상담/파일 첨부는 후속 구현 예정입니다.</p>
      </Card>
    </div>
  );
}
