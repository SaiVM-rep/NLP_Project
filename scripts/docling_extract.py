"""Tables and formulas of a PDF -> JSON, with a PNG crop of each region (DECISION-40).

Runs in `.venv-docling` (not the main env): `python scripts/docling_extract.py PDF OUT`.
Called by `app.ingest.objects.extract_objects`; standalone, so it imports nothing from app.
Output: a JSON list of {kind, page, caption, content, image_png} in document order.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from typing import Any

SCALE = 2.0  # render at 144 dpi: readable crops of small table text
PAD = 4.0  # points around each region


def _options() -> Any:
    import torch
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    opts = PdfPipelineOptions()
    opts.accelerator_options = AcceleratorOptions(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    opts.do_ocr = False  # academic PDFs carry a text layer; OCR only costs time
    opts.do_table_structure = True
    opts.do_formula_enrichment = True  # LaTeX for display formulas
    opts.generate_picture_images = False
    return opts


def extract(pdf: Path) -> list[dict[str, Any]]:
    import pypdfium2 as pdfium
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import DocItemLabel, TableItem, TextItem

    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_options())}
    )
    doc = conv.convert(pdf).document
    src = pdfium.PdfDocument(str(pdf))
    pages: dict[int, Any] = {}

    def crop(item: Any) -> str | None:
        if not item.prov:
            return None
        prov = item.prov[0]
        if prov.page_no not in pages:
            pages[prov.page_no] = src[prov.page_no - 1].render(scale=SCALE).to_pil()
        img = pages[prov.page_no]
        h = doc.pages[prov.page_no].size.height
        b = prov.bbox.to_top_left_origin(page_height=h)
        box = (
            max(0, int((b.l - PAD) * SCALE)),
            max(0, int((b.t - PAD) * SCALE)),
            min(img.width, int((b.r + PAD) * SCALE)),
            min(img.height, int((b.b + PAD) * SCALE)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        buf = io.BytesIO()
        img.crop(box).save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    out: list[dict[str, Any]] = []
    for item, _ in doc.iterate_items():
        if isinstance(item, TableItem):
            caption = item.caption_text(doc) or None
            content = item.export_to_markdown(doc).strip()
            if caption and content.startswith(caption):  # shown separately as the title
                content = content[len(caption) :].lstrip()
            kind = "table"
        elif isinstance(item, TextItem) and item.label == DocItemLabel.FORMULA:
            kind, caption, content = "formula", None, item.text.strip()
        else:
            continue
        png = crop(item)
        if png is None:
            continue
        out.append(
            {
                "kind": kind,
                "page": item.prov[0].page_no,
                "caption": caption,
                "content": content,
                "image_png": png,
            }
        )
    return out


def main() -> None:
    pdf, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.write_text(json.dumps(extract(pdf)), encoding="utf-8")


if __name__ == "__main__":
    main()
