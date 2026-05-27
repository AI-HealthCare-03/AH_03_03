import { FormEvent, useEffect, useMemo, useState } from "react";

import {
  AdminFaq,
  AdminFaqPayload,
  createAdminFaq,
  deactivateAdminFaq,
  listAdminFaqs,
  updateAdminFaq,
} from "../../api/admin";
import { useAuth } from "../../auth/AuthContext";
import { isOperatorRole } from "../../auth/AdminRoute";

const emptyDraft: AdminFaqPayload = {
  category: "GENERAL",
  question: "",
  answer: "",
  display_order: 0,
  is_active: true,
};

function formatDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return new Intl.DateTimeFormat("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export default function AdminFaqPage() {
  const { backendUser } = useAuth();
  const canManage = isOperatorRole(backendUser?.role);
  const [items, setItems] = useState<AdminFaq[]>([]);
  const [draft, setDraft] = useState<AdminFaqPayload>(emptyDraft);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [categoryFilter, setCategoryFilter] = useState("");
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      setItems(await listAdminFaqs({ category: categoryFilter || undefined }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "FAQ 목록을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (canManage) {
      void load();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canManage, categoryFilter]);

  const filteredItems = useMemo(() => {
    const keyword = search.trim().toLowerCase();
    if (!keyword) return items;
    return items.filter((item) => `${item.category} ${item.question} ${item.answer}`.toLowerCase().includes(keyword));
  }, [items, search]);

  const resetDraft = () => {
    setDraft(emptyDraft);
    setEditingId(null);
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setSaving(true);
    setError("");
    setNotice("");
    try {
      if (editingId) {
        await updateAdminFaq(editingId, draft);
        setNotice("FAQ가 수정되었습니다.");
      } else {
        await createAdminFaq(draft);
        setNotice("FAQ가 생성되었습니다.");
      }
      resetDraft();
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "FAQ 저장에 실패했습니다.");
    } finally {
      setSaving(false);
    }
  };

  const edit = (item: AdminFaq) => {
    setEditingId(item.id);
    setDraft({
      category: item.category,
      question: item.question,
      answer: item.answer,
      display_order: item.display_order,
      is_active: item.is_active,
    });
  };

  const deactivate = async (item: AdminFaq) => {
    if (!window.confirm("이 FAQ를 비활성화하시겠습니까? 사용자 화면에서 숨겨집니다.")) return;
    setSaving(true);
    setError("");
    setNotice("");
    try {
      await deactivateAdminFaq(item.id);
      setNotice("FAQ가 비활성화되었습니다.");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "FAQ 비활성화에 실패했습니다.");
    } finally {
      setSaving(false);
    }
  };

  if (!canManage) {
    return (
      <section className="admin-section">
        <div className="card">
          <span className="badge badge-muted">403</span>
          <h1>운영 권한이 필요합니다</h1>
          <p>FAQ 관리는 OPERATOR 이상 권한에서 사용할 수 있습니다.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="admin-section">
      <div className="page-header">
        <div>
          <h1>FAQ 관리</h1>
          <p>사용자에게 노출되는 도움말을 생성하고 공개 상태를 관리합니다.</p>
        </div>
      </div>

      {error && <div className="error-message">{error}</div>}
      {notice && <div className="state-box">{notice}</div>}

      <div className="admin-two-column">
        <article className="card">
          <h2>{editingId ? "FAQ 수정" : "FAQ 생성"}</h2>
          <form className="form" onSubmit={submit}>
            <label>
              카테고리
              <input
                value={draft.category}
                onChange={(event) => setDraft((prev) => ({ ...prev, category: event.target.value }))}
                required
              />
            </label>
            <label>
              질문
              <input
                value={draft.question}
                onChange={(event) => setDraft((prev) => ({ ...prev, question: event.target.value }))}
                required
              />
            </label>
            <label>
              답변
              <textarea
                value={draft.answer}
                onChange={(event) => setDraft((prev) => ({ ...prev, answer: event.target.value }))}
                required
              />
            </label>
            <div className="form-grid">
              <label>
                표시 순서
                <input
                  type="number"
                  value={draft.display_order ?? 0}
                  onChange={(event) => setDraft((prev) => ({ ...prev, display_order: Number(event.target.value) }))}
                />
              </label>
              <label className="toggle-row">
                <span>공개</span>
                <input
                  checked={Boolean(draft.is_active)}
                  onChange={(event) => setDraft((prev) => ({ ...prev, is_active: event.target.checked }))}
                  type="checkbox"
                />
              </label>
            </div>
            <div className="button-row">
              <button disabled={saving} type="submit">
                {editingId ? "수정 저장" : "생성"}
              </button>
              {editingId && (
                <button className="secondary" onClick={resetDraft} type="button">
                  취소
                </button>
              )}
            </div>
          </form>
        </article>

        <article className="card">
          <h2>필터</h2>
          <div className="form">
            <label>
              카테고리
              <input value={categoryFilter} onChange={(event) => setCategoryFilter(event.target.value)} />
            </label>
            <label>
              검색
              <input value={search} onChange={(event) => setSearch(event.target.value)} />
            </label>
          </div>
        </article>
      </div>

      <article className="card">
        <div className="section-title-row">
          <h2>FAQ 목록</h2>
          <span className="badge badge-muted">{filteredItems.length.toLocaleString()}건</span>
        </div>
        {loading ? (
          <div className="empty-state">FAQ 목록을 불러오는 중입니다...</div>
        ) : filteredItems.length === 0 ? (
          <div className="empty-state">등록된 FAQ가 없습니다.</div>
        ) : (
          <div className="admin-table-wrap">
            <table className="admin-table">
              <thead>
                <tr>
                  <th>질문</th>
                  <th>카테고리</th>
                  <th>상태</th>
                  <th>수정일</th>
                  <th>관리</th>
                </tr>
              </thead>
              <tbody>
                {filteredItems.map((item) => (
                  <tr key={item.id}>
                    <td>
                      <strong>{item.question}</strong>
                      <p className="muted admin-table-summary">{item.answer}</p>
                    </td>
                    <td>{item.category}</td>
                    <td>
                      <span className={item.is_active ? "badge badge-saved" : "badge badge-muted"}>
                        {item.is_active ? "공개" : "비공개"}
                      </span>
                    </td>
                    <td>{formatDateTime(item.updated_at)}</td>
                    <td>
                      <div className="button-row">
                        <button className="secondary" onClick={() => edit(item)} type="button">
                          수정
                        </button>
                        {item.is_active && (
                          <button className="btn-danger-outline" onClick={() => void deactivate(item)} type="button">
                            비활성화
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </article>
    </section>
  );
}
