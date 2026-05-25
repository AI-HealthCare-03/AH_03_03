from ai_runtime.ocr.medication.parser import parse_medication_text
from ai_runtime.ocr.medication.schemas import MedicationOcrParseResult


def extract_medication_ocr_result(raw_text: str | None) -> MedicationOcrParseResult:
    """Parse already-recognized medication package text into service fields.

    External OCR provider calls are intentionally outside this skeleton.
    """
    return parse_medication_text(raw_text)
