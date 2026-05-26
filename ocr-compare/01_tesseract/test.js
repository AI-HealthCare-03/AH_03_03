/**
 * 01_tesseract/test.js
 *
 * Tesseract.js OCR 테스트 (이미지 + PDF 지원)
 * 실행: node test.js
 */

const Tesseract = require("tesseract.js");
const fs = require("fs");
const path = require("path");
const { createCanvas } = require("canvas");
const pdfjsLib = require("pdfjs-dist/legacy/build/pdf.js");

// ── 경로 설정 ──────────────────────────────────────────────────────────────────

const IMAGES_DIR = path.join(__dirname, "../results/images/checkup");
const RESULTS_DIR = path.join(__dirname, "../results");
const COLUMNS_FILE = path.join(RESULTS_DIR, "columns.json");
const GROUND_TRUTH_FILE = path.join(RESULTS_DIR, "ground_truth.json");
const OCR_NAME = "tesseract";

// ── 데이터 로드 ────────────────────────────────────────────────────────────────

const columnsData = JSON.parse(fs.readFileSync(COLUMNS_FILE, "utf-8"));
const groundTruth = JSON.parse(fs.readFileSync(GROUND_TRUTH_FILE, "utf-8"));

const NOT_MEASURED_VALUE = columnsData.not_measured_value;
const NOT_MEASURED_KEYWORDS = columnsData.not_measured_keywords;

const NULLABLE_FIELDS = new Set();
for (const category of Object.values(columnsData.columns)) {
  for (const field of category.fields) {
    if (field.nullable) NULLABLE_FIELDS.add(field.key);
  }
}

const FIELD_KEYS = [];
for (const category of Object.values(columnsData.columns)) {
  for (const field of category.fields) {
    FIELD_KEYS.push(field.key);
  }
}

// ── 키워드 매핑 ────────────────────────────────────────────────────────────────

const FIELD_KEYWORDS = {
  systolic_bp:       ["수축기", "수축기혈압", "최고혈압", "SBP", "mmHg"],
  diastolic_bp:      ["이완기", "이완기혈압", "최저혈압", "DBP"],
  fasting_glucose:   ["공복혈당(mg", "공복혈당", "공복 혈당", "당뇨병", "GLU", "Glucose"],
  total_cholesterol: ["총콜레스테롤", "총 콜레스테롤", "콜레스테롤(mg", "TC", "T-CHO"],
  triglyceride:      ["중성지방(mg", "중성지방", "TG"],
  hdl:               ["고밀도 콜레스테롤", "고밀도콜레스테롤", "HDL"],
  ldl:               ["저밀도 콜레스테롤", "저밀도콜레스테롤", "LDL"],
  height_cm:         ["키(cm)", "키(Cm)", "신장(cm)", "신장", "Height"],
  weight_kg:         ["몸무게(kg)", "체중(kg)", "체중", "몸무게", "Weight"],
  bmi:               ["체질량지수", "체질량지수(kg", "BMI", "비만도"],
  waist_cm:          ["허리둘레(cm)", "허리둘레", "허리 둘레", "복부둘레"],
};

// ── 노이즈 패턴 ────────────────────────────────────────────────────────────────

const NOISE_PATTERN = /\d+\.?\d*\s*(만만|미만|이하|이상|초과|이내)/g;

function cleanLine(text) {
  return text
    .replace(/[■□▣▪●○◆◇]/g, "")
    .replace(NOISE_PATTERN, "")
    .trim();
}

function extractNumbers(text) {
  const cleaned = cleanLine(text).replace(/,/g, "");
  const matches = cleaned.match(/\d+\.?\d*/g);
  if (!matches) return [];
  return matches.map(Number).filter((n) => !isNaN(n));
}

function isKeywordMatch(text, keywords) {
  const upper = text.toUpperCase();
  return keywords.some((kw) => upper.includes(kw.toUpperCase()));
}

function isNotMeasured(text) {
  return NOT_MEASURED_KEYWORDS.some((kw) =>
    text.toUpperCase().includes(kw.toUpperCase())
  );
}

function validateValue(field, value) {
  const ranges = {
    systolic_bp:       [60, 250],
    diastolic_bp:      [40, 150],
    fasting_glucose:   [40, 600],
    total_cholesterol: [50, 600],
    triglyceride:      [20, 2000],
    hdl:               [10, 200],
    ldl:               [20, 500],
    height_cm:         [100, 250],
    weight_kg:         [20, 300],
    bmi:               [10, 70],
    waist_cm:          [40, 200],
  };
  if (!ranges[field]) return true;
  const [min, max] = ranges[field];
  return value >= min && value <= max;
}

// ── 파싱 함수 ──────────────────────────────────────────────────────────────────

function parseBloodPressure(lines) {
  for (const line of lines) {
    if (isKeywordMatch(line, ["수축기", "mmHg", "SBP"])) {
      const nums = extractNumbers(line).filter((n) => n >= 40 && n <= 250);
      if (nums.length >= 2) return [nums[0], nums[1]];
      if (nums.length === 1) return [nums[0], null];
    }
  }
  return [null, null];
}

function parseHeightWeight(lines) {
  for (const line of lines) {
    if (
      isKeywordMatch(line, ["키(cm)", "신장", "몸무게", "체중", "Height"]) ||
      (line.includes("키") && line.includes("몸무게"))
    ) {
      const nums = extractNumbers(line);
      for (let i = 0; i < nums.length - 1; i++) {
        const h = nums[i];
        const w = nums[i + 1];
        if (h >= 100 && h <= 250 && w >= 20 && w <= 300) {
          return [h, w];
        }
      }
    }
  }
  return [null, null];
}

function parseBmi(lines) {
  const BMI_NOISE = /\(?\d+\.?\d*\s*[-~]\s*\d+\.?\d*\)?/g;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!isKeywordMatch(line, FIELD_KEYWORDS.bmi)) continue;
    const cleaned = line
      .replace(BMI_NOISE, "")
      .replace(/\d+\.?\d*미만/g, "")
      .replace(/\d+\.?\d*이상/g, "")
      .replace(/[저정과비만체중상하]+/g, "");
    const nums = extractNumbers(cleaned).filter((n) => n >= 10 && n <= 50);
    if (nums.length) return nums[0];
    if (i + 1 < lines.length) {
      const nextNums = extractNumbers(lines[i + 1]).filter((n) => n >= 10 && n <= 50);
      if (nextNums.length) return nextNums[0];
    }
  }
  return null;
}

function parseWaist(lines) {
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!isKeywordMatch(line, FIELD_KEYWORDS.waist_cm)) continue;
    const nums = extractNumbers(line).filter((n) => n >= 40 && n <= 200);
    if (nums.length) return nums[0];
    if (i + 1 < lines.length) {
      const nextNums = extractNumbers(lines[i + 1]).filter((n) => n >= 40 && n <= 200);
      if (nextNums.length) return nextNums[0];
    }
  }
  return null;
}

function parseFastingGlucose(lines) {
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!isKeywordMatch(line, FIELD_KEYWORDS.fasting_glucose)) continue;
    const nums = extractNumbers(line).filter((n) => n >= 40 && n <= 600);
    if (nums.length) return nums[0];
    if (i + 1 < lines.length) {
      const nextNums = extractNumbers(lines[i + 1]).filter((n) => n >= 40 && n <= 600);
      if (nextNums.length) return nextNums[0];
    }
  }
  return null;
}

function parseDyslipidemia(lines, field) {
  const ranges = {
    total_cholesterol: [50, 600],
    triglyceride:      [20, 2000],
    hdl:               [10, 200],
    ldl:               [20, 500],
  };
  const [min, max] = ranges[field];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!isKeywordMatch(line, FIELD_KEYWORDS[field])) continue;
    const check = [line];
    if (i + 1 < lines.length) check.push(lines[i + 1]);
    if (i + 2 < lines.length) check.push(lines[i + 2]);
    if (check.some((l) => isNotMeasured(l))) return NOT_MEASURED_VALUE;
    for (const l of check) {
      const nums = extractNumbers(l).filter((n) => n >= min && n <= max);
      if (nums.length) return nums[0];
    }
  }
  return null;
}

function extractFromLines(lines) {
  const extracted = {};
  FIELD_KEYS.forEach((k) => (extracted[k] = null));

  const [systolic, diastolic] = parseBloodPressure(lines);
  if (systolic && validateValue("systolic_bp", systolic)) extracted.systolic_bp = systolic;
  if (diastolic && validateValue("diastolic_bp", diastolic)) extracted.diastolic_bp = diastolic;

  const [height, weight] = parseHeightWeight(lines);
  if (height) extracted.height_cm = height;
  if (weight) extracted.weight_kg = weight;

  const bmi = parseBmi(lines);
  if (bmi !== null) extracted.bmi = bmi;

  const waist = parseWaist(lines);
  if (waist !== null) extracted.waist_cm = waist;

  const glucose = parseFastingGlucose(lines);
  if (glucose !== null) extracted.fasting_glucose = glucose;

  for (const field of ["total_cholesterol", "triglyceride", "hdl", "ldl"]) {
    extracted[field] = parseDyslipidemia(lines, field);
  }

  return extracted;
}

// ── 정확도 측정 ────────────────────────────────────────────────────────────────

function evaluate(extracted, subject) {
  const gt = groundTruth.subjects[subject]?.ground_truth;
  if (!gt) {
    console.error(`정답 데이터 없음: ${subject}`);
    return null;
  }

  let correct = 0;
  let total = 0;
  const details = {};

  for (const key of FIELD_KEYS) {
    const gtVal = gt[key];
    const extVal = extracted[key];

    if (gtVal === null || gtVal === undefined) {
      details[key] = { gt: null, extracted: extVal, match: null };
      continue;
    }

    total++;
    let match = false;
    if (gtVal === NOT_MEASURED_VALUE) {
      match = extVal === NOT_MEASURED_VALUE;
    } else if (extVal !== null && extVal !== NOT_MEASURED_VALUE) {
      const tolerance = Math.abs(Number(gtVal)) * 0.05;
      match = Math.abs(Number(extVal) - Number(gtVal)) <= tolerance;
    }
    if (match) correct++;
    details[key] = { gt: gtVal, extracted: extVal, match };
  }

  return {
    correct,
    total,
    accuracy: total > 0 ? Math.round((correct / total) * 1000) / 10 : 0,
    details,
  };
}

// ── 결과 출력/저장 ─────────────────────────────────────────────────────────────

function printResult(subject, fileType, evalResult, elapsedMs) {
  console.log(`\n${"=".repeat(50)}`);
  console.log(`  ${OCR_NAME.toUpperCase()} | ${subject} | ${fileType}`);
  console.log("=".repeat(50));
  console.log(`  정확도: ${evalResult.correct}/${evalResult.total} (${evalResult.accuracy}%)`);
  console.log(`  속도:   ${elapsedMs}ms`);
  console.log("-".repeat(50));
  for (const [key, detail] of Object.entries(evalResult.details)) {
    const status = detail.match === null ? "⬜" : detail.match ? "✅" : "❌";
    const ext = detail.extracted === NOT_MEASURED_VALUE ? "비해당" : detail.extracted;
    console.log(`  ${status} ${key}: 정답=${detail.gt} / 추출=${ext}`);
  }
  console.log("=".repeat(50));
}

function saveResult(subject, fileType, evalResult, elapsedMs) {
  const output = {
    ocr_engine: OCR_NAME,
    subject,
    file_type:  fileType,
    accuracy:   evalResult.accuracy,
    correct:    evalResult.correct,
    total:      evalResult.total,
    elapsed_ms: elapsedMs,
    details:    evalResult.details,
  };
  const filename = path.join(RESULTS_DIR, `${OCR_NAME}_${subject}_${fileType}.json`);
  fs.writeFileSync(filename, JSON.stringify(output, null, 2), "utf-8");
  console.log(`  결과 저장: ${filename}`);
}

// ── PDF → 이미지 변환 ─────────────────────────────────────────────────────────

async function pdfToImages(pdfPath) {
  const data = new Uint8Array(fs.readFileSync(pdfPath));
  const loadingTask = pdfjsLib.getDocument({ data });
  const pdf = await loadingTask.promise;
  const images = [];

  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const viewport = page.getViewport({ scale: 2.0 });
    const canvas = createCanvas(viewport.width, viewport.height);
    const context = canvas.getContext("2d");
    await page.render({ canvasContext: context, viewport }).promise;
    images.push(canvas.toBuffer("image/jpeg", { quality: 0.95 }));
    console.log(`  PDF 페이지 ${i}/${pdf.numPages} 변환 완료`);
  }

  return images;
}

// ── README 자동 생성 ───────────────────────────────────────────────────────────

function generateReadme(allResults) {
  if (allResults.length === 0) return;

  const avgAccuracy =
    Math.round(
      (allResults.reduce((s, r) => s + r.evalResult.accuracy, 0) / allResults.length) * 10
    ) / 10;
  const avgSpeed = Math.round(
    allResults.reduce((s, r) => s + r.elapsedMs, 0) / allResults.length
  );
  const subjects = [...new Set(allResults.map((r) => r.subject))];

  const jpgRows = allResults
    .filter((r) => r.fileType === "jpg")
    .map((r) => `| ${r.subject} | ${r.evalResult.correct}/${r.evalResult.total} (${r.evalResult.accuracy}%) | ${r.elapsedMs}ms |`);

  const pdfRows = allResults
    .filter((r) => r.fileType === "pdf")
    .map((r) => `| ${r.subject} | ${r.evalResult.correct}/${r.evalResult.total} (${r.evalResult.accuracy}%) | ${r.elapsedMs}ms |`);

  const colHeader = `| 컬럼 | ${subjects.join(" | ")} | 평균 |`;
  const colSep = `|------|${"--------|".repeat(subjects.length)}--------|`;
  const colRows = FIELD_KEYS.map((key) => {
    const cells = subjects.map((subj) => {
      const r = allResults.find((x) => x.subject === subj && x.fileType === "jpg");
      if (!r) return "-";
      const detail = r.evalResult.details[key];
      if (!detail || detail.match === null) return "⬜";
      if (detail.match) return detail.extracted === NOT_MEASURED_VALUE ? "✅ 비해당" : "✅";
      return `❌ (${detail.extracted})`;
    });
    const matchCount = subjects.filter((subj) => {
      const r = allResults.find((x) => x.subject === subj && x.fileType === "jpg");
      return r && r.evalResult.details[key]?.match === true;
    }).length;
    return `| ${key} | ${cells.join(" | ")} | ${matchCount}/${subjects.length} |`;
  });

  const now = new Date().toLocaleString("ko-KR");
  const content = `# Tesseract.js

## 테스트 결과 요약
- **평균 정확도 (JPG)**: ${avgAccuracy}%
- **평균 속도**: ${avgSpeed}ms
- **테스트 일시**: ${now}

## 설치 방법
\`\`\`bash
npm install tesseract.js pdfjs-dist canvas
\`\`\`

## JPG 테스트
| 대상 | 정확도 | 속도 |
|------|--------|------|
${jpgRows.join("\n")}

## PDF 테스트
| 대상 | 정확도 | 속도 |
|------|--------|------|
${pdfRows.length ? pdfRows.join("\n") : "_해당 없음_"}

## 컬럼별 인식 결과 (JPG 기준)
${colHeader}
${colSep}
${colRows.join("\n")}

## 범례
- ✅ 정답 일치
- ✅ 비해당 정확히 인식
- ❌ 오인식
- ⬜ 정답 데이터 없음

## 장점
- 오픈소스, 완전 무료
- 브라우저 / Node.js 모두 지원
- 별도 서버 불필요

## 단점
- 한국어 인식률 낮음
- PDF 직접 미지원 (pdfjs-dist 변환 필요)
- 표 구조 인식 약함

## 결론
- 추후 작성
`;

  const readmePath = path.join(__dirname, "README.md");
  fs.writeFileSync(readmePath, content, "utf-8");
  console.log(`\n📝 README.md 자동 생성: ${readmePath}`);
}

// ── 테스트 실행 ────────────────────────────────────────────────────────────────

async function runTest(filePath, subject, fileType) {
  console.log(`\n분석 중: ${path.basename(filePath)}`);
  const start = Date.now();
  const result = await Tesseract.recognize(filePath, "kor+eng", { logger: () => {} });
  const elapsed = Date.now() - start;

  const lines = result.data.text
    .split("\n")
    .map((l) => cleanLine(l))
    .filter((l) => l.length > 0);

  const extracted = extractFromLines(lines);
  const evalResult = evaluate(extracted, subject);
  if (!evalResult) return null;

  printResult(subject, fileType, evalResult, elapsed);
  saveResult(subject, fileType, evalResult, elapsed);
  return { subject, fileType, evalResult, elapsedMs: elapsed };
}

async function runPdfTest(filePath, subject) {
  console.log(`\n분석 중 (PDF): ${path.basename(filePath)}`);
  try {
    const images = await pdfToImages(filePath);
    let allLines = [];
    let totalElapsed = 0;

    for (let i = 0; i < images.length; i++) {
      console.log(`  페이지 ${i + 1}/${images.length} OCR 실행 중...`);
      const start = Date.now();
      const result = await Tesseract.recognize(images[i], "kor+eng", { logger: () => {} });
      totalElapsed += Date.now() - start;
      const lines = result.data.text
        .split("\n")
        .map((l) => cleanLine(l))
        .filter((l) => l.length > 0);
      allLines = allLines.concat(lines);
    }

    const extracted = extractFromLines(allLines);
    const evalResult = evaluate(extracted, subject);
    if (!evalResult) return null;

    printResult(subject, "pdf", evalResult, totalElapsed);
    saveResult(subject, "pdf", evalResult, totalElapsed);
    return { subject, fileType: "pdf", evalResult, elapsedMs: totalElapsed };

  } catch (err) {
    console.error(`  PDF 처리 실패: ${err.message}`);
    return null;
  }
}

async function main() {
  console.log("🔍 Tesseract.js OCR 테스트 시작 (이미지 + PDF)");
  console.log(`이미지 경로: ${IMAGES_DIR}\n`);

  const allResults = [];
  const files = fs.readdirSync(IMAGES_DIR).map((f) => ({
    filePath: path.join(IMAGES_DIR, f),
    filename: f,
  }));

  for (const { filePath, filename } of files) {
    let subject = null;
    for (const name of Object.keys(groundTruth.subjects)) {
      if (filename.includes(name)) { subject = name; break; }
    }

    if (!subject) {
      console.log(`⚠️  대상자 매핑 실패: ${filename}`);
      continue;
    }

    const fileType = path.extname(filename).toLowerCase().replace(".", "");

    if (fileType === "pdf") {
      const result = await runPdfTest(filePath, subject);
      if (result) allResults.push(result);
    } else {
      const result = await runTest(filePath, subject, fileType);
      if (result) allResults.push(result);
    }
  }

  generateReadme(allResults);
  console.log("\n✅ 테스트 완료");
}

main().catch(console.error);