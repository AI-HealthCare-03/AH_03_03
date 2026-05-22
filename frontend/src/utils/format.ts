const MEAL_TYPE_MAP: Record<string, string> = {
  BREAKFAST: "아침",
  LUNCH: "점심",
  DINNER: "저녁",
  SNACK: "간식",
  LATE_NIGHT: "야식",
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

export function formatRelativeTime(value: unknown): string {
  if (!value || value === "") return "-";
  const date = new Date(String(value));
  if (Number.isNaN(date.getTime())) return "-";

  const diffMs = Date.now() - date.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);
  if (diffSeconds < 60) return "방금";

  const diffMinutes = Math.floor(diffSeconds / 60);
  if (diffMinutes < 60) return `${diffMinutes}분 전`;

  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}시간 전`;

  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) return `${diffDays}일 전`;

  return formatDateTime(value);
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
