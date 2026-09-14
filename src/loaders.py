"""
Document Upload -> Parsing stage.

Each loader returns a list of `Page` objects: raw text plus metadata
(source filename, page number) so citations can point back to an exact
location in the original document.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Page:
    text: str
    source: str
    page_number: int


def load_txt_or_md(path: Path) -> List[Page]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Page(text=text, source=path.name, page_number=1)]


def load_csv(path: Path) -> List[Page]:
    import csv
    rows = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        for row in reader:
            rows.append(", ".join(f"{h}: {v}" for h, v in zip(header, row)))
    # Group rows into pseudo-pages so a 10,000-row CSV doesn't become one page.
    pages, batch, batch_size = [], [], 200
    for i, row in enumerate(rows, 1):
        batch.append(row)
        if len(batch) >= batch_size:
            pages.append(Page(text="\n".join(batch), source=path.name, page_number=len(pages) + 1))
            batch = []
    if batch:
        pages.append(Page(text="\n".join(batch), source=path.name, page_number=len(pages) + 1))
    return pages or [Page(text="", source=path.name, page_number=1)]


def load_pdf(path: Path) -> List[Page]:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("pypdf is required for PDF parsing: pip install pypdf") from e

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages.append(Page(text=text, source=path.name, page_number=i))
    return pages


def load_docx(path: Path) -> List[Page]:
    try:
        import docx
    except ImportError as e:
        raise ImportError("python-docx is required for DOCX parsing: pip install python-docx") from e

    doc = docx.Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs)
    # DOCX has no reliable page boundaries via python-docx; treat as one logical page,
    # the chunker will subdivide it anyway.
    return [Page(text=text, source=path.name, page_number=1)]


LOADERS = {
    ".txt": load_txt_or_md,
    ".md": load_txt_or_md,
    ".csv": load_csv,
    ".pdf": load_pdf,
    ".docx": load_docx,
}


def load_document(path: str) -> List[Page]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {list(LOADERS)}")
    pages = LOADERS[ext](p)
    return [pg for pg in pages if pg.text and pg.text.strip()]
