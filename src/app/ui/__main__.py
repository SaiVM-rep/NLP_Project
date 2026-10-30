"""`python -m app.ui` / `make ui`: serve the page on localhost only (no share link)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from app.ui.app import CSS, build_app

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-dir", type=Path, default=ROOT / "demo")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build_app(a.demo_dir, a.api).launch(
        server_name="127.0.0.1", server_port=a.port, share=False, css=CSS
    )


if __name__ == "__main__":
    main()
