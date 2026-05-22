import { FormEvent, useEffect, useMemo, useState } from "react";

import {
  AdminInquiry,
  answerAdminInquiry,
  getAdminInquiry,
  listAdminInquiries,
} from "../../api/admin";
import { useAuth } from "../../auth/AuthContext";
import { isOperatorRole } from "../../auth/AdminRoute";

function formatDateTime(value: string | null | undefined): string {
  if (!value) return "-";
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

function statusLabel(value: string): string {
  const normalized = value.toUpperCase();
  if (normalized === "ANSWERED") return "답변 완료";
  if (normalized === "CLOSED") return "종료";
  return "대기";
}

export default function AdminInquiryPage() {
  const { backendUser } = useAuth();
  const canManage = isOperatorRole(backendUser?.role);
  const [items, setItems] = useState<AdminInquiry[]>([]);
  const [selected, setSelected] = useState<AdminInquiry | null>(null);
  const [answer, setAnswer] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const nextItems = await listAdminInquiries({ status: statusFilter || undefined });
      setItems(nextItems);
      if (selected) {
        setSelected(nextItems.find((item) => item.id === selected.id) ?? null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "문의 목록을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (canManage) {
      void load();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canManage, statusFilter]);

  const filteredItems = useMemo(() => {
    const keyword = search.trim().toLowerCase();
    if (!keyword) return items;
    return items.filter((item) => `${item.category} ${item.title} ${item.status}`.toLowerCase().includes(keyword));
  }, [items, search]);

  const selectInquiry = async (item: AdminInquiry) => {
    setError("");
    setNotice("");
    try {
      const detail = await getAdminInquiry(item.id);
      setSelected(detail);
      setAnswer(detail.answer ?? "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "문의 상세를 불러오지 못했습니다.");
    }
  };

  const submitAnswer = async (event: FormEvent) => {
    event.preventDefault();
    if (!selected) return;
    setSaving(true);
    setError("");
    setNotice("");
    try {
      const updated = await answerAdminInquiry(selected.id, answer);
      setSelected(updated);
      setNotice("문의 답변이 저장되었습니다.");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "문의 답변 저장에 실패했습니다.");
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
          <p>문의 관리는 OPERATOR 이상 권한에서 사용할 수 있습니다.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="admin-section">
      <div className="page-header">
        <div>
          <h1>문의 관리</h1>
          <p>1:1 문의를 확인하고 사용자에게 답변합니다.</p>
        </div>
      </div>

      {error && <div className="error-message">{error}</div>}
      {notice && <div className="state-box">{notice}</div>}

      <div className="admin-two-column admin-management-grid">
        <article className="card">
          <div className="section-title-row">
            <h2>문의 목록</h2>
            <span className="badge badge-muted">{filteredItems.length.toLocaleString()}건</span>
          </div>
          <div className="form-grid">
            <label>
              상태
              <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
                <option value="">전체</option>
                <option value="PENDING">대기</option>
                <option value="ANSWERED">답변 완료</option>
              </select>
            </label>
            <label>
              검색
              <input value={search} onChange={(event) => setSearch(event.target.value)} />
            </label>
          </div>
          {loading ? (
            <div className="empty-state">문의 목록을 불러오는 중입니다...</div>
          ) : filteredItems.length === 0 ? (
            <div className="empty-state">조회된 문의가 없습니다.</div>
          ) : (
            <div className="admin-list">
              {filteredItems.map((item) => (
                <button
                  className={selected?.id === item.id ? "admin-list-item active" : "admin-list-item"}
                  key={item.id}
                  onClick={() => void selectInquiry(item)}
                  type="button"
                >
                  <span className="badge badge-reference">{item.category}</span>
                  <strong>{item.title}</strong>
                  <span className={item.status === "ANSWERED" ? "badge badge-saved" : "badge badge-missing"}>
                    {statusLabel(item.status)}
                  </span>
                  <small>
                    작성자 #{item.user_id} · {formatDateTime(item.created_at)}
                  </small>
                </button>
              ))}
            </div>
          )}
        </article>

        <article className="card">
          <div className="section-title-row">
            <h2>문의 상세/답변</h2>
            {selected && <span className="badge badge-muted">#{selected.id}</span>}
          </div>
          {!selected ? (
            <div className="empty-state">답변할 문의를 선택하세요.</div>
          ) : (
            <div className="admin-detail-stack">
              <div className="admin-inquiry-detail">
                <span className="badge badge-reference">{selected.category}</span>
                <h3>{selected.title}</h3>
                <p className="muted">
                  작성자 #{selected.user_id} · {formatDateTime(selected.created_at)}
                </p>
                <div className="state-box">{selected.content}</div>
              </div>
              <form className="form" onSubmit={submitAnswer}>
                <label>
                  답변
                  <textarea value={answer} onChange={(event) => setAnswer(event.target.value)} required />
                </label>
                <div className="state-box">
                  의료 진단을 단정하지 말고, 필요한 경우 의료기관 상담을 권장하는 표현을 사용하세요. 개인
                  건강정보가 포함된 답변은 최소한으로 작성하세요.
                </div>
                <button disabled={saving} type="submit">
                  답변 저장
                </button>
              </form>
            </div>
          )}
        </article>
      </div>
    </section>
  );
}
