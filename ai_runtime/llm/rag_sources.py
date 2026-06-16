from urllib.parse import urlparse

ALLOWED_RAG_SOURCES = [
    "질병관리청 국가건강정보포털",
    "국민건강보험공단",
    "대한고혈압학회",
    "대한당뇨병학회",
    "대한비만학회",
    "대한지질·동맥경화학회",
]

ALLOWED_RAG_DOMAINS = [
    "health.kdca.go.kr",
    "nhis.or.kr",
    "diabetes.or.kr",
    "koreanhypertension.org",
    # 대한지질·동맥경화학회 guideline source is managed in docs/rag_sources/index.json.
    "lipid.or.kr",
    # TODO: 대한비만학회 공식 도메인 확인 후 추가
]


def is_allowed_rag_source(source_name: str | None = None, url: str | None = None) -> bool:
    if source_name and is_allowed_source_name(source_name):
        return True

    if url and is_allowed_domain(url):
        return True

    return False


def is_allowed_source_name(source_name: str) -> bool:
    normalized_source = source_name.strip()
    if not normalized_source:
        return False

    return any(allowed_source in normalized_source for allowed_source in ALLOWED_RAG_SOURCES)


def is_allowed_domain(url: str) -> bool:
    domain = extract_domain(url)
    if not domain:
        return False

    return any(
        domain == allowed_domain or domain.endswith(f".{allowed_domain}") for allowed_domain in ALLOWED_RAG_DOMAINS
    )


def extract_domain(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return parsed.netloc.lower().removeprefix("www.")
