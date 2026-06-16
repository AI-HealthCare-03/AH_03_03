import { FormEvent, useEffect, useState } from "react";

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
type MedicationFormErrors = Partial<Record<"name" | "medication_type" | "reminder_time", string>>;

const MEDICATION_TYPE_LABEL: Record<string, string> = {
  MEDICATION: "경구약",
  SUPPLEMENT: "영양제",
  PRESCRIPTION: "처방약",
};

function getMedicationTypeLabel(type: unknown): string {
  return MEDICATION_TYPE_LABEL[String(type ?? "")] ?? String(type ?? "-");
}

function cleanOptionalText(value: unknown): string | null {
  const text = String(value ?? "").trim();
  return text.length > 0 ? text : null;
}

function validateMedicationDraft(draft: MedicationPayload): MedicationFormErrors {
  const errors: MedicationFormErrors = {};
  const name = String(draft.name ?? "").trim();
  const medicationType = String(draft.medication_type ?? "").trim();
  const reminderTime = String(draft.reminder_time ?? "").trim();

  if (!name) {
    errors.name = "약 이름을 입력해 주세요.";
  }
  if (!medicationType || !Object.prototype.hasOwnProperty.call(MEDICATION_TYPE_LABEL, medicationType)) {
    errors.medication_type = "구분을 선택해 주세요.";
  }
  if (reminderTime && !/^([01]\d|2[0-3]):[0-5]\d(:[0-5]\d)?$/.test(reminderTime)) {
    errors.reminder_time = "복용 시간은 08:00처럼 입력해 주세요.";
  }

  return errors;
}

function buildMedicationPayload(draft: MedicationPayload): MedicationPayload {
  return {
    name: String(draft.name ?? "").trim(),
    medication_type: String(draft.medication_type ?? "SUPPLEMENT").trim(),
    dosage: cleanOptionalText(draft.dosage),
    frequency: cleanOptionalText(draft.frequency),
    reminder_time: cleanOptionalText(draft.reminder_time),
    memo: cleanOptionalText(draft.memo),
    is_active: draft.is_active !== false,
  };
}

export default function MedicationPage() {
  const [registerDraft, setRegisterDraft] = useState<MedicationPayload>({
    name: "",
    medication_type: "SUPPLEMENT",
    dosage: "",
    frequency: "",
    reminder_time: null,
    memo: "",
    is_active: true,
  });
  const [items, setItems] = useState<Item[]>([]);
  const [records, setRecords] = useState<Item[]>([]);
  const [error, setError] = useState("");
  const [registerErrors, setRegisterErrors] = useState<MedicationFormErrors>({});
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
    await submitRegisterDraft();
  };

  const submitRegisterDraft = async () => {
    if (isMutating) return;
    setError("");
    const validationErrors = validateMedicationDraft(registerDraft);
    setRegisterErrors(validationErrors);
    if (Object.keys(validationErrors).length > 0) {
      setError("필수 입력값을 확인해 주세요.");
      return;
    }

    const payload = buildMedicationPayload(registerDraft);
    try {
      setIsMutating(true);
      const created = await createMedication<Item>(payload);
      setItems((prev) => [created, ...prev.filter((item) => Number(item.id) !== Number(created.id))]);
      setRecords([]);
      setRegisterDraft({
        name: "",
        medication_type: "SUPPLEMENT",
        dosage: "",
        frequency: "",
        reminder_time: null,
        memo: "",
        is_active: true,
      });
      setRegisterErrors({});
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
        <div className="state-box" style={{ marginBottom: 12 }}>
          <p style={{ marginBottom: 6 }}>
            약물명 확인은 아래 사이트를 참고해 주세요.
          </p>
          <ul style={{ paddingLeft: 16, marginBottom: 8 }}>
            <li>
              <a href="https://www.health.kr" target="_blank" rel="noopener noreferrer">
                약학정보원 (health.kr)
              </a>
            </li>
            <li>
              <a href="https://www.druginfo.co.kr" target="_blank" rel="noopener noreferrer">
                드럭인포 (druginfo.co.kr)
              </a>
            </li>
          </ul>
          <p style={{ color: "var(--color-danger, #e53e3e)", fontWeight: 600, fontSize: "0.85rem" }}>
            ⚠️ 본 서비스는 처방·복용 판단을 제공하지 않습니다.
            복약 관련 결정은 반드시 의사 또는 약사와 상담하세요.
          </p>
        </div>

        <form className="form" noValidate onSubmit={submit}>
          <div className="form two-col">
            <label>
              <span style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 4 }}>
                약 이름
                <span style={{ color: "var(--color-danger, #e53e3e)" }}>*</span>
              </span>
              <input
                placeholder="약 이름 직접 입력"
                value={String(registerDraft.name ?? "")}
                onChange={(e) => setRegisterDraft((prev) => ({ ...prev, name: e.target.value }))}
              />
              {registerErrors.name && <span className="muted" style={{ color: "var(--color-danger, #e53e3e)" }}>{registerErrors.name}</span>}
            </label>
            <label>
              구분
              <select
                value={String(registerDraft.medication_type ?? "SUPPLEMENT")}
                onChange={(e) => setRegisterDraft((prev) => ({ ...prev, medication_type: e.target.value }))}
              >
                <option value="MEDICATION">경구약</option>
                <option value="SUPPLEMENT">영양제</option>
                <option value="PRESCRIPTION">처방약</option>
              </select>
              {registerErrors.medication_type && <span className="muted" style={{ color: "var(--color-danger, #e53e3e)" }}>{registerErrors.medication_type}</span>}
            </label>
            <label>
              용량
              <input
                placeholder="예: 500mg"
                value={String(registerDraft.dosage ?? "")}
                onChange={(e) => setRegisterDraft((prev) => ({ ...prev, dosage: e.target.value }))}
              />
            </label>
            <label>
              복용 횟수
              <input
                placeholder="예: 매일 1회"
                value={String(registerDraft.frequency ?? "")}
                onChange={(e) => setRegisterDraft((prev) => ({ ...prev, frequency: e.target.value }))}
              />
            </label>
            <label>
              복용 시간
              <input
                placeholder="예: 08:00"
                value={String(registerDraft.reminder_time ?? "")}
                onChange={(e) =>
                  setRegisterDraft((prev) => ({
                    ...prev,
                    reminder_time: e.target.value || null,
                  }))
                }
              />
              {registerErrors.reminder_time && <span className="muted" style={{ color: "var(--color-danger, #e53e3e)" }}>{registerErrors.reminder_time}</span>}
            </label>
            <label>
              메모
              <input
                placeholder="예: 식후 복용"
                value={String(registerDraft.memo ?? "")}
                onChange={(e) => setRegisterDraft((prev) => ({ ...prev, memo: e.target.value }))}
              />
            </label>
          </div>
          <div style={{ marginTop: 10, display: "flex", justifyContent: "flex-end" }}>
            <button disabled={isMutating} type="submit">
              {isMutating ? "추가 중..." : "추가"}
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
