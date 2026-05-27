from pathlib import Path


def convert_pdf_first_page_to_image(pdf_path: str, output_image_path: str, zoom: float = 2.0) -> str:
    """Convert the first page of a PDF to a JPG image.

    This PoC requires PyMuPDF. Install it locally when running the experiment:
        uv pip install pymupdf

    Dependency files (`pyproject.toml`, `uv.lock`) are intentionally not updated
    in this PoC step.
    """

    source = Path(pdf_path)
    if not source.is_file():
        raise FileNotFoundError(f"PDF file not found: {source}")

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF conversion. Run: uv pip install pymupdf") from exc

    output = Path(output_image_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with fitz.open(source) as document:
        if document.page_count == 0:
            raise ValueError(f"PDF has no pages: {source}")

        page = document.load_page(0)
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        pixmap.save(output)

    return str(output)


def convert_pdf_all_pages_to_images(pdf_path: str, output_dir: str, zoom: float = 2.0) -> list[str]:
    """Convert all PDF pages to JPG images named page_001.jpg, page_002.jpg, ...

    This PoC requires PyMuPDF. Install it locally when running the experiment:
        uv pip install pymupdf

    Dependency files (`pyproject.toml`, `uv.lock`) are intentionally not updated
    in this PoC step.
    """

    source = Path(pdf_path)
    if not source.is_file():
        raise FileNotFoundError(f"PDF file not found: {source}")

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF conversion. Run: uv pip install pymupdf") from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    image_paths = []
    with fitz.open(source) as document:
        if document.page_count == 0:
            raise ValueError(f"PDF has no pages: {source}")

        matrix = fitz.Matrix(zoom, zoom)
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output / f"page_{page_index + 1:03d}.jpg"
            pixmap.save(image_path)
            image_paths.append(str(image_path))

    return image_paths
