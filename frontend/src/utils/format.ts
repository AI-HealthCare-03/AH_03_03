const MEAL_TYPE_MAP: Record<string, string> = {
  BREAKFAST: "아침",
  LUNCH: "점심",
  DINNER: "저녁",
  SNACK: "간식",
};

export function formatDateTime(value: unknown): string {
  if (!value || value === "") return "-";
  const d = new Date(String(value));
  if (isNaN(d.getTime())) return "-";
  const parts = new Intl.DateTimeFormat("ko-KR", {
    timeZone: "Asia/Seoul",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).formatToParts(d);
  const get = (type: string) => parts.find((p) => p.type === type)?.value ?? "";
  return `${get("year")}.${get("month")}.${get("day")} ${get("hour")}:${get("minute")}`;
}

export function mealTypeLabel(value: unknown): string {
  if (!value) return "알 수 없음";
  return MEAL_TYPE_MAP[String(value).toUpperCase()] ?? "알 수 없음";
}

export function scoreBadgeClass(score: number): string {
  if (score >= 80) return "risk-low";
  if (score >= 60) return "risk-medium";
  return "risk-high";
}
