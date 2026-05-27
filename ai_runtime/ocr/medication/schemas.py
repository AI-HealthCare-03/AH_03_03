from datetime import date

from pydantic import BaseModel, Field


class MedicationOcrItem(BaseModel):
    medication_name: str
    dosage: str | None = None
    frequency: str | None = None
    duration_days: int | None = None
    instruction: str | None = None
    pharmacy_name: str | None = None
    prescribed_date: date | None = None
    confidence: float | None = None
    raw_text: str


class MedicationOcrParseResult(BaseModel):
    items: list[MedicationOcrItem] = Field(default_factory=list)
    pharmacy_name: str | None = None
    prescribed_date: date | None = None
    raw_text: str = ""
    source: str = "rule_based_medication_ocr"
    warnings: list[str] = Field(default_factory=list)
