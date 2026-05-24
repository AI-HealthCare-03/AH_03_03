import re
from datetime import date

from ai_worker.ocr.medication.schemas import MedicationOcrItem, MedicationOcrParseResult

DATE_PATTERN = re.compile(r"(?P<year>20\d{2})[.\-/년\s]+(?P<month>\d{1,2})[.\-/월\s]+(?P<day>\d{1,2})")
DOSAGE_PATTERN = re.compile(r"(?P<dosage>\d+(?:\.\d+)?\s?(?:mg|g|mcg|㎎|정|캡슐|mL|ml|IU))", re.IGNORECASE)
FREQUENCY_PATTERN = re.compile(r"(?:하루|1일|일\s?일)\s*(?P<count>\d+)\s*회")
DURATION_PATTERN = re.compile(r"(?P<days>\d+)\s*일(?:분|간)?")
PHARMACY_PATTERN = re.compile(r"(?P<name>[가-힣A-Za-z0-9\s]{2,30}약국)")
INSTRUCTION_KEYWORDS = ("식후", "식전", "취침", "아침", "점심", "저녁", "필요시", "복용")
HEADER_KEYWORDS = ("처방", "조제", "약국", "환자", "성명", "일자", "전화", "주소")


def parse_medication_text(raw_text: str | None) -> MedicationOcrParseResult:
    text = (raw_text or "").strip()
    if not text:
        return MedicationOcrParseResult(raw_text="", warnings=["empty_raw_text"])

    pharmacy_name = _extract_pharmacy_name(text)
    prescribed_date = _extract_prescribed_date(text)
    items = [
        item
        for line in _candidate_lines(text)
        if (item := _parse_medication_line(line, pharmacy_name=pharmacy_name, prescribed_date=prescribed_date))
        is not None
    ]
    warnings = [] if items else ["no_medication_item_detected"]
    return MedicationOcrParseResult(
        items=items,
        pharmacy_name=pharmacy_name,
        prescribed_date=prescribed_date,
        raw_text=text,
        warnings=warnings,
    )


def _parse_medication_line(
    line: str,
    pharmacy_name: str | None,
    prescribed_date: date | None,
) -> MedicationOcrItem | None:
    normalized = " ".join(line.split())
    if not normalized or _is_header_line(normalized):
        return None

    dosage = _match_text(DOSAGE_PATTERN, normalized, "dosage")
    frequency = _extract_frequency(normalized)
    duration_days = _extract_duration_days(normalized)
    instruction = _extract_instruction(normalized)
    medication_name = _extract_medication_name(normalized, dosage, frequency)

    if not medication_name or (dosage is None and frequency is None and duration_days is None):
        return None

    return MedicationOcrItem(
        medication_name=medication_name,
        dosage=dosage,
        frequency=frequency,
        duration_days=duration_days,
        instruction=instruction,
        pharmacy_name=pharmacy_name,
        prescribed_date=prescribed_date,
        confidence=None,
        raw_text=normalized,
    )


def _candidate_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        normalized = line.strip(" -•\t")
        if normalized:
            lines.append(normalized)
    return lines


def _is_header_line(line: str) -> bool:
    return any(keyword in line for keyword in HEADER_KEYWORDS) and DOSAGE_PATTERN.search(line) is None


def _extract_medication_name(line: str, dosage: str | None, frequency: str | None) -> str:
    end_positions = [len(line)]
    if dosage:
        dosage_match = DOSAGE_PATTERN.search(line)
        if dosage_match:
            end_positions.append(dosage_match.start())
    if frequency:
        frequency_match = FREQUENCY_PATTERN.search(line)
        if frequency_match:
            end_positions.append(frequency_match.start())
    for pattern in (DURATION_PATTERN,):
        match = pattern.search(line)
        if match:
            end_positions.append(match.start())
    name = line[: min(end_positions)].strip(" :,/-")
    name = re.sub(r"^\d+\s*", "", name).strip()
    return name


def _extract_frequency(line: str) -> str | None:
    match = FREQUENCY_PATTERN.search(line)
    if not match:
        return None
    return f"하루 {match.group('count')}회"


def _extract_duration_days(line: str) -> int | None:
    matches = list(DURATION_PATTERN.finditer(line))
    if not matches:
        return None
    try:
        return int(matches[-1].group("days"))
    except ValueError:
        return None


def _extract_instruction(line: str) -> str | None:
    found = [keyword for keyword in INSTRUCTION_KEYWORDS if keyword in line]
    return " / ".join(dict.fromkeys(found)) if found else None


def _extract_pharmacy_name(text: str) -> str | None:
    match = PHARMACY_PATTERN.search(text)
    return match.group("name").strip() if match else None


def _extract_prescribed_date(text: str) -> date | None:
    match = DATE_PATTERN.search(text)
    if not match:
        return None
    try:
        return date(int(match.group("year")), int(match.group("month")), int(match.group("day")))
    except ValueError:
        return None


def _match_text(pattern: re.Pattern[str], text: str, group_name: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    return match.group(group_name).replace(" ", "")
