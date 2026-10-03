import json
import re
import sys
from pathlib import Path


def find_bsr_root() -> Path:
    for candidate in Path("literature").iterdir():
        if candidate.is_dir() and candidate.name.startswith("BSR") and "nature" in candidate.name:
            return candidate
    raise FileNotFoundError("Could not find BSR nature reading directory under literature/")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def snippet(text: str, needle: str, radius: int = 900) -> str | None:
    folded = text.lower()
    idx = folded.find(needle.lower())
    if idx < 0:
        return None
    return clean(text[max(0, idx - radius) : min(len(text), idx + radius)])


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = find_bsr_root()
    out = root / "evidence_contexts.md"
    lines: list[str] = [
        "# BSR Evidence Contexts",
        "",
        "This file is generated from local `source_bundle.json` files. It records short page-grounded contexts for tables, datasets, methods, and evaluation protocols.",
        "",
    ]
    key_terms = [
        "Abstract",
        "Related Work",
        "3. Method",
        "Proposed Method",
        "Experiments",
        "Experimental",
        "Implementation Details",
        "Table 1",
        "Table 2",
        "Table 3",
        "Table 4",
        "PSNR",
        "BurstSR",
        "SyntheticBurst",
        "RealBSR",
        "alignment",
        "base frame",
        "uncertainty set",
        "ambiguity set",
    ]
    for paper_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if "codex-pdf-flow-test" in paper_dir.name:
            continue
        bundle_path = paper_dir / "source_bundle.json"
        if not bundle_path.exists():
            continue
        bundle = json.loads(bundle_path.read_text(encoding="utf-8", errors="replace"))
        title = bundle.get("metadata", {}).get("title") or paper_dir.name
        pages = bundle.get("pages", [])
        lines.extend([f"## {title}", "", f"- Slug: `{paper_dir.name}`", f"- Pages: {len(pages)}", ""])
        for term in key_terms:
            hits = []
            for page in pages:
                text = page.get("text", "")
                snip = snippet(text, term)
                if snip:
                    hits.append((page.get("pdf_page"), snip))
            if not hits:
                continue
            lines.append(f"### {term}")
            for page_no, snip in hits[:3]:
                lines.append(f"- PDF p.{page_no}: {snip[:1200]}")
            lines.append("")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
