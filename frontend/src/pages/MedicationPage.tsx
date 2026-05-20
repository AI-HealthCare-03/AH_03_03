import { FormEvent, useEffect, useState } from "react";

import { createMedication, listMedicationRecords, listMedications, updateMedicationRecord } from "../api/medications";
import Card from "../components/Card";

type Item = Record<string, unknown>;

export default function MedicationPage() {
  const [name, setName] = useState("");
  const [items, setItems] = useState<Item[]>([]);
  const [records, setRecords] = useState<Item[]>([]);

  const load = async () => {
    const medications = await listMedications<Item[]>();
    setItems(medications);
    if (medications[0]?.id) {
      setRecords(await listMedicationRecords<Item[]>(Number(medications[0].id)));
    }
  };

  useEffect(() => {
    void load().catch(() => undefined);
  }, []);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    await createMedication({
      name,
      medication_type: "SUPPLEMENT",
      frequency: "매일 1회",
      is_active: true,
    });
    setName("");
    await load();
  };

  return (
    <div className="page-grid">
      <Card title="복약/영양제 등록">
        <form className="form" onSubmit={submit}>
          <label>
            이름
            <input value={name} onChange={(event) => setName(event.target.value)} required />
          </label>
          <button type="submit">등록</button>
        </form>
      </Card>
      <Card title="복약 목록">
        <pre>{JSON.stringify(items, null, 2)}</pre>
      </Card>
      <Card title="복약 기록">
        <div className="card-list">
          {records.map((record) => (
            <div className="mini-card" key={String(record.id)}>
              <pre>{JSON.stringify(record, null, 2)}</pre>
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
    </div>
  );
}
