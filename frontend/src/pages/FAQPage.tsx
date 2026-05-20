import { FormEvent, useEffect, useState } from "react";

import { createInquiry, listFaqs, listMyInquiries } from "../api/faqs";
import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";

type Item = Record<string, unknown>;

export default function FAQPage() {
  const { firebaseUser } = useAuth();
  const [faqs, setFaqs] = useState<Item[]>([]);
  const [inquiries, setInquiries] = useState<Item[]>([]);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");

  const load = async () => {
    setFaqs(await listFaqs<Item[]>());
    if (firebaseUser) {
      setInquiries(await listMyInquiries<Item[]>());
    }
  };

  useEffect(() => {
    void load().catch(() => undefined);
  }, [firebaseUser]);

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
        <div className="card-list">
          {faqs.map((faq) => (
            <div className="mini-card" key={String(faq.id)}>
              <strong>{String(faq.question)}</strong>
              <p>{String(faq.answer)}</p>
            </div>
          ))}
        </div>
      </Card>
      <Card title="문의 작성">
        <form className="form" onSubmit={submit}>
          <input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="제목" required />
          <textarea value={content} onChange={(event) => setContent(event.target.value)} placeholder="내용" required />
          <button type="submit" disabled={!firebaseUser}>
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
