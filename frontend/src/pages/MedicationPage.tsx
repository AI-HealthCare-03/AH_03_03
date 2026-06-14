import { FormEvent, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import {
  createMedication,
  deactivateMedication,
  deleteMedication,
  getMedication,
  listMedicationRecords,
  listMedications,
  updateMedication,
  updateMedicationRecord,
  type MedicationPayload,
} from "../api/medications";
import Card from "../components/Card";
import ConfirmDialog from "../components/ConfirmDialog";
import ErrorMessage from "../components/ErrorMessage";

type Item = Record<string, unknown>;

const MEDICATION_TYPE_LABEL: Record<string, string> = {
  MEDICATION: "경구약",
  SUPPLEMENT: "영양제",
  PRESCRIPTION: "처방약",
};

function getMedicationTypeLabel(type: unknown): string {
  return MEDICATION_TYPE_LABEL[String(type ?? "")] ?? String(type ?? "-");
}

export default function MedicationPage() {
  const [name, setName] = useState("");
  const [items, setItems] = useState<Item[]>([]);
  const [records, setRecords] = useState<Item[]>([]);
  const [error, setError] = useState("");
  const [selectedMedicationId, setSelectedMedicationId] = useState<number | null>(null);
  const [editDraft, setEditDraft] = useState<MedicationPayload>({});
  const [pendingAction, setPendingAction] = useState<null | { type: "deactivate" | "delete"; medicationId: number }>(
    null,
  );
  const [isMutating, setIsMutating] = useState(false);

  const load = async () => {
    setError("");
    try {
      const medications = await listMedications<Item[]>();
      setItems(medications);
      if (medications[0]?.id) {
        setRecords(await listMedicationRecords<Item[]>(Number(medications[0].id)));
      } else {
        setRecords([]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "복약 정보를 불러오지 못했습니다.");
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (isMutating) return;
    setError("");
    try {
      setIsMutating(true);
      await createMedication({
        name,
        medication_type: "SUPPLEMENT",
        frequency: "매일 1회",
        is_active: true,
      });
      setName("");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "복약/영양제 등록에 실패했습니다.");
    } finally {
      setIsMutating(false);
    }
  };

  const getMedicationName = (record: Item) => {
    const medication = items.find((item) => Number(item.id) === Number(record.medication_id));
    return String(medication?.name ?? "복약 기록");
  };

  const startEdit = async (medicationId: number) => {
    setError("");
    try {
      const medication = await getMedication<Item>(medicationId);
      setSelectedMedicationId(medicationId);
      setEditDraft({
        name: String(medication.name ?? ""),
        medication_type: String(medication.medication_type ?? "SUPPLEMENT"),
        dosage: medication.dosage ? String(medication.dosage) : "",
        frequency: medication.frequency ? String(medication.frequency) : "",
        reminder_time: medication.reminder_time ? String(medication.reminder_time) : null,
        memo: medication.memo ? String(medication.memo) : "",
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "복약/영양제 정보를 불러오지 못했습니다.");
    }
  };

  const saveEdit = async () => {
    if (!selectedMedicationId || isMutating) return;
    setError("");
    try {
      setIsMutating(true);
      await updateMedication(selectedMedicationId, editDraft);
      setSelectedMedicationId(null);
      setEditDraft({});
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "복약/영양제 수정에 실패했습니다.");
    } finally {
      setIsMutating(false);
    }
  };

  const confirmPendingAction = async () => {
    if (!pendingAction || isMutating) return;
    setError("");
    try {
      setIsMutating(true);
      if (pendingAction.type === "deactivate") {
        await deactivateMedication(pendingAction.medicationId);
      } else {
        await deleteMedication(pendingAction.medicationId);
      }
      if (selectedMedicationId === pendingAction.medicationId) {
        setSelectedMedicationId(null);
        setEditDraft({});
      }
      setPendingAction(null);
      await load();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : pendingAction.type === "deactivate"
            ? "복약/영양제 중단 처리에 실패했습니다."
            : "복약/영양제 삭제에 실패했습니다.",
      );
    } finally {
      setIsMutating(false);
    }
  };

  return (
    <div className="page-stack">
      {/* 헤더 */}
      <div className="page-header">
        <div>
          <h1>복약 관리</h1>
          <p>복용 중인 약과 영양제를 관리합니다.</p>
        </div>
      </div>

      {error && <ErrorMessage message={error} />}

      {/* ── 등록 카드 ── */}
      <Card title="등록">
        <div className="button-row" style={{ marginBottom: 12 }}>
          <Link className="button" to="/ocr/medication">
            처방전/약봉투로 등록
          </Link>
        </div>
        <form className="form" onSubmit={submit}>
          <div style={{ display: "flex", gap: 8 }}>
            <input
              placeholder="약 이름 직접 입력"
              style={{ flex: 1 }}
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
            <button disabled={isMutating} type="submit" style={{ whiteSpace: "nowrap" }}>
              추가
            </button>
          </div>
        </form>
      </Card>

      {/* ── 복약 목록 ── */}
      <Card title="복약 목록">
        {items.length === 0 && (
          <div className="state-box">등록된 복약 정보가 없습니다.</div>
        )}
        <div className="med-grid">
          {items.map((item) => {
            const isActive = Boolean(item.is_active);
            const isEditing = selectedMedicationId === Number(item.id);
            return (
              <div
                className="mini-card"
                key={String(item.id)}
                style={{ opacity: isActive ? 1 : 0.55 }}
              >
                <div className="record-row">
                  <div>
                    <strong>{String(item.name ?? "복약/영양제")}</strong>
                    <p className="muted">{getMedicationTypeLabel(item.medication_type)}</p>
                  </div>
                  <span className={isActive ? "badge badge-saved" : "badge badge-missing"}>
                    {isActive ? "복용 중" : "중단"}
                  </span>
                </div>
                <div className="record-meta-grid">
                  <span>용량 {String(item.dosage ?? "-")}</span>
                  <span>복용 횟수 {String(item.frequency ?? "-")}</span>
                  <span>복용 시간 {String(item.reminder_time ?? "-")}</span>
                  <span>메모 {String(item.memo ?? "-")}</span>
                </div>
                <div className="button-row">
                  <button
                    className="secondary"
                    disabled={isMutating}
                    onClick={() => void startEdit(Number(item.id))}
                    type="button"
                  >
                    {isEditing ? "수정 중..." : "수정"}
                  </button>
                  {isActive ? (
                    <button
                      className="secondary"
                      disabled={isMutating}
                      onClick={() =>
                        setPendingAction({ type: "deactivate", medicationId: Number(item.id) })
                      }
                      type="button"
                    >
                      중단
                    </button>
                  ) : (
                    <button
                      className="secondary"
                      disabled={isMutating}
                      onClick={() =>
                        void updateMedication(Number(item.id), { is_active: true }).then(load)
                      }
                      type="button"
                    >
                      재개
                    </button>
                  )}
                  <button
                    className="danger-ghost"
                    disabled={isMutating}
                    onClick={() =>
                      setPendingAction({ type: "delete", medicationId: Number(item.id) })
                    }
                    type="button"
                    aria-label="삭제"
                  >
                    🗑
                  </button>
                </div>

                {/* 인라인 수정 폼 */}
                {isEditing && (
                  <div style={{ marginTop: 12, borderTop: "0.5px solid var(--color-border)", paddingTop: 12 }}>
                    <div className="form two-col">
                      <label>
                        이름
                        <input
                          value={String(editDraft.name ?? "")}
                          onChange={(e) => setEditDraft((prev) => ({ ...prev, name: e.target.value }))}
                        />
                      </label>
                      <label>
                        구분
                        <select
                          value={String(editDraft.medication_type ?? "SUPPLEMENT")}
                          onChange={(e) =>
                            setEditDraft((prev) => ({ ...prev, medication_type: e.target.value }))
                          }
                        >
                          <option value="MEDICATION">경구약</option>
                          <option value="SUPPLEMENT">영양제</option>
                          <option value="PRESCRIPTION">처방약</option>
                        </select>
                      </label>
                      <label>
                        용량
                        <input
                          value={String(editDraft.dosage ?? "")}
                          onChange={(e) => setEditDraft((prev) => ({ ...prev, dosage: e.target.value }))}
                        />
                      </label>
                      <label>
                        복용 횟수
                        <input
                          value={String(editDraft.frequency ?? "")}
                          onChange={(e) =>
                            setEditDraft((prev) => ({ ...prev, frequency: e.target.value }))
                          }
                        />
                      </label>
                      <label>
                        메모
                        <input
                          value={String(editDraft.memo ?? "")}
                          onChange={(e) => setEditDraft((prev) => ({ ...prev, memo: e.target.value }))}
                        />
                      </label>
                    </div>
                    <div className="button-row" style={{ marginTop: 10 }}>
                      <button disabled={isMutating} onClick={() => void saveEdit()} type="button">
                        저장
                      </button>
                      <button
                        className="secondary"
                        disabled={isMutating}
                        onClick={() => { setSelectedMedicationId(null); setEditDraft({}); }}
                        type="button"
                      >
                        취소
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </Card>

      {/* ── 복약 기록 ── */}
      <Card title="복약 기록">
        <div className="card-list">
          {records.length === 0 && (
            <div className="state-box">최근 복약 기록이 없습니다.</div>
          )}
          {records.map((record) => (
            <div className="mini-card" key={String(record.id)}>
              <div className="record-row">
                <div>
                  <strong>{getMedicationName(record)}</strong>
                  <p className="muted">{String(record.scheduled_at ?? record.created_at ?? "일정 없음")}</p>
                </div>
                <span className={record.is_taken ? "badge badge-saved" : "badge badge-missing"}>
                  {record.is_taken ? "복용 완료" : "복용 대기"}
                </span>
              </div>
              <button
                onClick={() =>
                  void updateMedicationRecord(Number(record.id), {
                    is_taken: true,
                    status: "TAKEN",
                    taken_at: new Date().toISOString(),
                  }).then(load)
                }
              >
                복약 완료
              </button>
            </div>
          ))}
        </div>
      </Card>

      {/* 확인 다이얼로그 */}
      {pendingAction && (
        <ConfirmDialog
          cancelLabel="취소"
          confirmLabel={pendingAction.type === "deactivate" ? "중단하기" : "삭제하기"}
          message={
            pendingAction.type === "deactivate"
              ? "이 복약/영양제 정보를 중단 처리하시겠습니까?"
              : "이 복약/영양제 정보를 삭제하시겠습니까? 삭제 후에는 목록에서 제거됩니다."
          }
          onCancel={() => setPendingAction(null)}
          onConfirm={() => void confirmPendingAction()}
          title={pendingAction.type === "deactivate" ? "복용 중단 확인" : "복약 정보 삭제 확인"}
          tone={pendingAction.type === "delete" ? "danger" : "default"}
        />
      )}
    </div>
  );
}