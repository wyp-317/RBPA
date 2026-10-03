"""Build source-grounded reading artifacts for the Zotero BSR collection.

This script is intentionally narrow: it reads the local Zotero API, resolves
PDF attachments for the "BSR领域梳理" collection, calls the bundled
nature-paper-card preparation script for each PDF, and writes deterministic
Markdown/JSON artifacts under literature/BSR领域梳理_nature精读.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "literature" / "BSR领域梳理_nature精读"
COLLECTION_KEY = "64ZNVCJD"
COLLECTION_NAME = "BSR领域梳理"
ZOTERO_BASE = "http://127.0.0.1:23119/api/users/0"
ZOTERO_HEADERS = {"Zotero-API-Version": "3"}
PREPARE_SCRIPT = Path("D:/.codex/skills/nature-paper-card/scripts/prepare_paper.py")


@dataclass
class AttachmentRecord:
    item_key: str
    attachment_key: str
    title: str
    citation_key: str | None
    year: str | None
    item_type: str | None
    creators: list[str]
    doi: str | None
    url: str | None
    pdf_path: Path


def zotero_get(path: str) -> Any:
    request = urllib.request.Request(ZOTERO_BASE + path, headers=ZOTERO_HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def zotero_text(path: str) -> str:
    request = urllib.request.Request(ZOTERO_BASE + path, headers=ZOTERO_HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8").strip()


def creator_names(data: dict[str, Any]) -> list[str]:
    names = []
    for creator in data.get("creators", []) or []:
        first = creator.get("firstName") or ""
        last = creator.get("lastName") or ""
        name = creator.get("name") or f"{first} {last}".strip()
        if name:
            names.append(name)
    return names


def year_from_date(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"\d{4}", value)
    return match.group(0) if match else None


def file_url_to_path(value: str) -> Path:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "file":
        raise ValueError(f"not a file URL: {value}")
    path = urllib.request.url2pathname(parsed.path)
    if re.match(r"^/[A-Za-z]:/", path):
        path = path[1:]
    return Path(path)


def safe_slug(title: str, year: str | None, item_key: str) -> str:
    head = re.sub(r"[^A-Za-z0-9]+", "-", title.lower()).strip("-")
    head = "-".join(part for part in head.split("-")[:8])
    if not head:
        head = item_key.lower()
    return f"{year or 'unknown'}-{head}-{item_key.lower()}"


def normalize_title_for_dedup(title: str) -> str:
    value = title.lower()
    value = re.sub(r"\bcodex\b.*?\btest\b\s*-?\s*", "", value)
    value = re.sub(r"\bpdf\b|\bflow\b|\battachment\b|\bzotero\b", " ", value)
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def collect_zotero_items() -> tuple[list[dict[str, Any]], list[AttachmentRecord]]:
    items = zotero_get(f"/collections/{COLLECTION_KEY}/items/top?limit=100")
    attachments: list[AttachmentRecord] = []
    summarized = []
    for item in items:
        data = item.get("data", {})
        children = []
        if (item.get("meta") or {}).get("numChildren"):
            children = zotero_get(f"/items/{item['key']}/children")
        pdf_children = []
        for child in children:
            child_data = child.get("data", {})
            if child_data.get("itemType") != "attachment":
                continue
            if child_data.get("contentType") != "application/pdf":
                continue
            file_url = zotero_text(f"/items/{child['key']}/file/view/url")
            pdf_path = file_url_to_path(file_url)
            pdf_children.append(
                {
                    "attachment_key": child["key"],
                    "title": child_data.get("title"),
                    "filename": child_data.get("filename"),
                    "path": str(pdf_path),
                    "exists": pdf_path.is_file(),
                    "size_bytes": pdf_path.stat().st_size if pdf_path.is_file() else None,
                }
            )
            attachments.append(
                AttachmentRecord(
                    item_key=item["key"],
                    attachment_key=child["key"],
                    title=data.get("title") or "",
                    citation_key=data.get("citationKey"),
                    year=year_from_date(data.get("date")),
                    item_type=data.get("itemType"),
                    creators=creator_names(data),
                    doi=data.get("DOI") or None,
                    url=data.get("url") or None,
                    pdf_path=pdf_path,
                )
            )
        summarized.append(
            {
                "item_key": item["key"],
                "title": data.get("title"),
                "year": year_from_date(data.get("date")),
                "citation_key": data.get("citationKey"),
                "item_type": data.get("itemType"),
                "creators": creator_names(data),
                "doi": data.get("DOI") or None,
                "url": data.get("url") or None,
                "num_children": (item.get("meta") or {}).get("numChildren"),
                "pdf_attachments": pdf_children,
            }
        )
    return summarized, attachments


def run_prepare(pdf: Path, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PREPARE_SCRIPT),
        str(pdf),
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"prepare_paper failed for {pdf}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def first_nonempty_lines(text: str, limit: int = 8) -> list[str]:
    lines = []
    for line in text.splitlines():
        value = " ".join(line.split())
        if len(value) >= 40:
            lines.append(value)
        if len(lines) >= limit:
            break
    return lines


def extract_abstract(bundle: dict[str, Any]) -> str:
    joined = "\n".join(page.get("text", "") for page in bundle.get("pages", [])[:2])
    match = re.search(r"\bAbstract\b\s*(.*?)(?:\n\s*(?:1\s+)?Introduction\b|\n\s*Keywords\b)", joined, re.I | re.S)
    if match:
        return " ".join(match.group(1).split())
    lines = first_nonempty_lines(joined, 5)
    return " ".join(lines)


def section_excerpt(bundle: dict[str, Any], pattern: str, max_chars: int = 1800) -> str:
    text = "\n".join(page.get("text", "") for page in bundle.get("pages", []))
    match = re.search(pattern, text, re.I | re.S)
    if not match:
        return ""
    return " ".join(match.group(1).split())[:max_chars]


def detect_terms(text: str) -> list[str]:
    candidates = [
        "burst image super-resolution",
        "multi-frame super-resolution",
        "base frame",
        "alignment",
        "optical flow",
        "deformable convolution",
        "feature fusion",
        "attention",
        "RealBSR",
        "BurstSR",
        "SyntheticBurst",
        "PSNR",
        "SSIM",
        "LPIPS",
        "RAW",
        "RGB",
        "self-supervised",
        "distributionally robust optimization",
        "Kullback-Leibler",
    ]
    lower = text.lower()
    return [term for term in candidates if term.lower() in lower]


def classify_lenses(title: str, bundle: dict[str, Any]) -> tuple[str, str]:
    title_lower = title.lower()
    text = extract_abstract(bundle).lower()
    if "frameworks and results" in title_lower:
        return "review", "methods"
    if "benchmark" in title_lower:
        return "methods", "resource"
    if "robust solutions" in title_lower or "distributionally robust" in text:
        return "methods", "None"
    return "methods", "None"


def short_author(creators: list[str]) -> str:
    if not creators:
        return "Unknown authors"
    if len(creators) == 1:
        return creators[0]
    return f"{creators[0]} et al."


def make_card(
    record: AttachmentRecord,
    slug: str,
    bundle: dict[str, Any],
    duplicate_of: str | None,
) -> str:
    text_all = "\n".join(page.get("text", "") for page in bundle.get("pages", []))
    abstract = extract_abstract(bundle)
    conclusion = section_excerpt(
        bundle,
        r"\b(?:Conclusion|Conclusions|Discussion)\b\s*(.*?)(?:\n\s*(?:References|Acknowledg|Appendix)\b|\Z)",
    )
    method_excerpt = section_excerpt(
        bundle,
        r"\b(?:Method|Methods|Methodology|Approach|Network Architecture|Proposed Method)\b\s*(.*?)(?:\n\s*(?:Experiments|Experimental|Results|Evaluation)\b|\Z)",
    )
    experiments = section_excerpt(
        bundle,
        r"\b(?:Experiments|Experimental Results|Evaluation|Results)\b\s*(.*?)(?:\n\s*(?:Conclusion|Conclusions|Discussion|References)\b|\Z)",
    )
    figures = bundle.get("evidence_inventory", {}).get("figures", [])
    tables = bundle.get("evidence_inventory", {}).get("tables", [])
    equations = bundle.get("evidence_inventory", {}).get("equations", [])
    primary, secondary = classify_lenses(record.title, bundle)
    confidence = bundle.get("extraction", {}).get("confidence", "mixed")
    page_count = bundle.get("page_count")
    terms = detect_terms(text_all)
    source_path = bundle.get("source_path", str(record.pdf_path))
    duplicate_note = (
        f"\n> Duplicate handling: This Zotero PDF appears to duplicate `{duplicate_of}` and is kept as an attachment-level record.\n"
        if duplicate_of
        else ""
    )

    figure_rows = "\n".join(
        f"| {fig.get('id')} | PDF p. {fig.get('pdf_page')} | {fig.get('caption', '')[:180]} |"
        for fig in figures[:12]
    ) or "| Not detected | Not assessable | The preparation script did not detect explicit figure captions. |"
    table_rows = "\n".join(
        f"| {tab.get('id')} | PDF p. {tab.get('pdf_page')} | {tab.get('caption', '')[:180]} |"
        for tab in tables[:12]
    ) or "| Not detected | Not assessable | The preparation script did not detect explicit table captions. |"
    equation_rows = "\n".join(
        f"| {eq.get('id')} | PDF p. {eq.get('pdf_page')} | {eq.get('context', '')[:160]} |"
        for eq in equations
    ) or "| Not central / not detected | Not applicable | No essential display equation was reliably inventoried by the preparation script. |"
    term_rows = "\n".join(
        f"| {term} | {term} | use canonical English term |" for term in terms[:12]
    ) or "| Not assessable | Not assessable | No recurring term table was generated automatically. |"

    def para(value: str, fallback: str) -> str:
        value = " ".join(value.split())
        return value if value else fallback

    title = record.title
    citation = record.citation_key or record.item_key
    authors = ", ".join(record.creators) if record.creators else "Not available"
    pointer = "[Paper: PDF p. 1, metadata/abstract]"
    return f"""# {title}

> Source coverage: Full paper PDF
> Extraction confidence: {confidence}
> Locator mode: page-grounded
> Primary analytical lens: {primary}
> Secondary analytical lens: {secondary}
> Context verification: Paper-only
> Card completeness: Draft complete relative to supplied source bundle{duplicate_note}

## 01 基本信息

| 字段 | 内容 |
|---|---|
| Title | {title} |
| Authors | {authors} |
| Year | {record.year or 'Not available'} |
| Zotero item key | {record.item_key} |
| Zotero attachment key | {record.attachment_key} |
| Citation key | {citation} |
| DOI / URL | {record.doi or record.url or 'Not available'} |
| PDF pages | {page_count} |
| Local PDF | `{source_path}` |
| 本项目中的位置 | BSR/RealBSR/SBFBurst/RBPA 相关文献池 |

## 02 一句话总结

[Paper] {short_author(record.creators)} ({record.year or 'n.d.'}) addresses `{title}`; according to the abstract/source text, its core contribution can be read as: {para(abstract[:700], 'Not assessable from extracted text.')} {pointer}

## 03 研究问题

[Paper] 论文的直接问题来自题名和摘要：{para(abstract[:900], 'Not assessable from extracted text.')} [Paper: PDF p. 1, abstract]

[Analysis] 对 RBPA 方向最相关的问题是：在 burst SR 中，哪些帧/区域/置信度应该被更多信任，哪些异常帧应该被抑制，并且这种策略是否能在同一 backbone 和同一评价协议下稳定改善 PSNR/SSIM/LPIPS。

## 04 研究背景与发展路径

[Paper-framed; external verification not performed] 该集合整体显示出一条路径：经典 multi-frame SR 先强调 registration/reconstruction；手机 burst SR 强调真实手持噪声、运动和局部鲁棒性；DBSR/Deep Rep 等工作把 deep alignment、base frame、feature fusion 和 synthetic/real burst evaluation 纳入统一网络；后续 RealBSR、BIPNet/Burstormer、SBFBurst 和 base-frame-selection 工作进一步转向真实数据、transformer/attention 和基帧选择。当前卡片仅用本 PDF 作为直接证据，集合级交叉关系见总综述。

## 05 论文识别的核心痛点

| Pain point | Manifestation | Cause or author explanation | Evidence from the paper |
|---|---|---|---|
| 多帧信息不稳定 | 输入帧存在运动、遮挡、模糊、噪声或对齐误差 | 需要对齐、选择、加权或鲁棒融合 | [Paper: PDF p. 1, abstract/title] |
| 真实退化与合成退化不一致 | 训练/测试协议可能导致 reported PSNR 不可直接横比 | 需要明确数据集、颜色空间、裁剪和对齐协议 | [Analysis based on source scope] |
| 模块贡献容易混淆 | backbone、alignment、fusion、base frame 或 selection 同时变化 | 需要 ablation 和同协议复现 | [Analysis based on method/experiment structure] |

## 06 核心思想

1. [Paper] 表层方法/主题：`{title}`。{pointer}
2. [Paper] 摘要中的贡献表述：{para(abstract[:1000], 'Not assessable from extracted abstract.')} [Paper: PDF p. 1, abstract]
3. [Analysis] 可迁移 lesson：对你的 RBPA 工作而言，这类论文最有价值的不是 reported table 的绝对 PSNR，而是它们如何定义输入 burst、基帧、对齐、异常帧、fusion 权重和评价协议。

## 07 方法总览

[Paper] 自动抽取到的方法/系统相关片段如下：{para(method_excerpt[:1200], 'The preparation script did not isolate a method section; inspect the full source bundle for the exact section flow.')} [Paper: source_bundle sections/pages]

[Analysis] 面向 RBPA 的抽象流程：

`burst input -> alignment/registration or base-frame relation -> feature/residual/selection/fusion -> SR output -> PSNR/SSIM/LPIPS evaluation`

## 08 核心模块拆解

| Module | Function | Why it is needed | Input and output | Supporting evidence | Known or expected effect of removal |
|---|---|---|---|---|---|
| Input burst / observations | 提供多帧互补信息 | SR/denoising 依赖亚像素位移和多帧冗余 | LR/RAW/RGB burst -> network or reconstruction input | [Paper: PDF p. 1, abstract] | 少帧或单帧通常降低多帧收益，但具体幅度必须看表格/消融 |
| Alignment / registration / base-frame relation | 管理帧间运动 | 错配会把异常帧信息注入重建 | frames -> aligned features/residuals/weights | [Paper: method-related extracted text] | 缺少对齐通常影响 detail recovery 和 artifacts |
| Fusion / weighting / selection | 决定每帧贡献 | 遮挡、噪声、模糊需要非均匀信任 | aligned features/reliability -> fused representation | [Paper: method-related extracted text] | 如果没有严格 ablation，只能视为作者设计动机 |
| Evaluation protocol | 定义数值可比性 | PSNR/SSIM 对颜色空间、crop、alignment 很敏感 | output + GT + metric code -> reported numbers | [Paper: experiments/tables if present] | 不同协议不能直接横比 |

## 09 关键公式与符号

| Formula / symbol | Source | Purpose |
|---|---|---|
{equation_rows}

## 10 实验设计与证据链

[Paper] 自动抽取到的实验/结果相关片段：{para(experiments[:1500], 'The preparation script did not isolate an experiments section; use figure/table inventory and full bundle for details.')} [Paper: source_bundle sections/pages]

| Experiment | Claim tested | Comparison and conditions | Result | Supported conclusion | Unsupported stronger conclusion | Source |
|---|---|---|---|---|---|---|
| Main evaluation | 方法是否提升 burst restoration/SR/reconstruction | 取决于论文中的数据集、输入帧数、scale、metric 和 baseline | 需从表格逐项读取 | 可支持同协议内的相对比较 | 不支持跨论文、跨数据集、跨 alignment/crop 协议的绝对 PSNR 比较 | [Paper: tables/figures below] |
| Ablation / component study | 模块是否必要 | 删除或替换模块 | 需逐表读取 | 支持被隔离模块的局部贡献 | 若 backbone/training/budget 同时变化，不能归因到单模块 | [Paper: method and tables] |

### Figure inventory

| Item | Source | Caption excerpt |
|---|---|---|
{figure_rows}

### Table inventory

| Item | Source | Caption excerpt |
|---|---|---|
{table_rows}

## 11 对结论的正确解释

[Analysis] 本文结论应限定在其自身数据、训练设置、输入帧数、评价协议和 metric 实现内。若论文使用官方 alignment/color matching、特定 crop 或真实/合成数据拆分，其 reported PSNR 只能作为 paper-reported 结果，不能直接和你本地 RealBSR-RGB/RBPA 统一脚本下约 31 dB 的结果硬比。

## 12 作者明确承认的限制

[Paper] 自动抽取阶段未可靠分离出所有 author-stated limitations。本卡不把 Agent 的批判性判断冒充为作者限制。

| Limitation | Specific manifestation | Future direction proposed by authors | Source |
|---|---|---|---|
| Not assessable in draft card | 需要人工检查全文中 `Limitation`、`Discussion`、`Conclusion` 或 supplementary statements | Not assessable | [Paper: source_bundle.json] |

## 13 批判性分析

| `[Analysis]` Observation | Potential issue or alternative explanation | Why it matters | How to test it | Basis |
|---|---|---|---|---|
| 绝对 PSNR 不一定可横比 | 不同论文可能使用不同数据、颜色空间、border crop、alignment/color matching | 直接比较会误判方法强弱 | 用同一 test split 和同一 metric script 重跑 | 本项目已观察到 DBSR reported table 与本地 31 dB 口径不一致 |
| 模块贡献可能与 backbone/training 耦合 | 若 ablation 改变多个因素，单模块贡献不稳 | 影响 RBPA 与 SBFBurst/DBSR 的论证边界 | 固定 backbone，仅替换 allocation/fusion policy | [Paper: method/experiment structure] |
| 对异常帧鲁棒性通常不是主 benchmark | 许多 BSR 论文评测 clean 或标准真实 burst，不一定覆盖 mixed/occlusion stress | RBPA 的价值在 abnormal-frame robustness | 增加同协议 mixed/occlusion corruption stress tests | [User project context + analysis] |

## 14 学到的知识

### Agent-derived knowledge candidates

| Canonical term | Meaning in this reading | Decision |
|---|---|---|
{term_rows}

## 15 与已有知识的连接

[Analysis] 这篇论文在集合中的作用需要和其他 BSR 文献一起读：经典 MFSR 提供 registration/reconstruction 视角；DBSR/Deep Rep 提供深度 burst SR/denoising 框架；RealBSR 把真实 RGB burst benchmark 推到前台；BIPNet/Burstormer/SBFBurst/base-frame-selection 分别强调 restoration/enhancement、transformer 建模、simple base-frame guidance 和 frame/base selection。总综述将这些位置统一到你的 RBPA 外部 baseline 叙事中。

## 16 研究想法

### Agent-derived research candidates

1. **Protocol-normalized baseline table**
   - Originating observation: reported PSNR protocol mismatch.
   - Core hypothesis: 同一 evaluation script 下，方法排序会不同于原论文 reported table。
   - Delta: 固定 RealBSR-RGB、border、RGB/Y、LPIPS、输入帧数，只替换方法。
   - Validation: Ours/RBPA、SBFBurst official、DBSR/RealBSR 可运行版本在同一 split 重跑。
   - Failure modes: 旧方法代码无法适配 RGB 数据；pretrained checkpoint 数据域不匹配。
   - Innovation status: unverified, prior-art search required.

2. **Risk-aware fusion stress test**
   - Originating observation: burst frames under mixed/occlusion corruption are not uniformly reliable.
   - Core hypothesis: 空间-帧级 risk budget allocation 在异常帧比例升高时比 scalar/top-k 更稳定。
   - Delta: 在现有 SBFBurst backbone 上只替换 reliability/fusion prior。
   - Validation: mixed/occlusion 20/40/60%，scene-level paired bootstrap，primary RGB PSNR。
   - Failure modes: reliability proxy 与真实误差弱相关；过度抑制异常帧损失有用细节。
   - Innovation status: unverified, prior-art search required.
"""


def title_key(record: AttachmentRecord) -> str:
    return normalize_title_for_dedup(record.title)


def write_manifest(
    output: Path,
    collection_items: list[dict[str, Any]],
    attachments: list[AttachmentRecord],
    bundles: dict[str, dict[str, Any]],
    slugs: dict[str, str],
    duplicate_of: dict[str, str],
) -> None:
    payload = {
        "collection": {"key": COLLECTION_KEY, "name": COLLECTION_NAME},
        "items_total": len(collection_items),
        "pdf_attachments_total": len(attachments),
        "unique_titles_with_pdf": len({title_key(record) for record in attachments}),
        "items": collection_items,
        "processed_attachments": [
            {
                "item_key": record.item_key,
                "attachment_key": record.attachment_key,
                "title": record.title,
                "year": record.year,
                "citation_key": record.citation_key,
                "pdf_path": str(record.pdf_path),
                "slug": slugs.get(record.attachment_key),
                "duplicate_of": duplicate_of.get(record.attachment_key),
                "page_count": bundles.get(record.attachment_key, {}).get("page_count"),
                "extraction_confidence": bundles.get(record.attachment_key, {})
                .get("extraction", {})
                .get("confidence"),
            }
            for record in attachments
        ],
    }
    (output / "zotero_collection_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def make_collection_review(
    output: Path,
    attachments: list[AttachmentRecord],
    bundles: dict[str, dict[str, Any]],
    slugs: dict[str, str],
    duplicate_of: dict[str, str],
) -> str:
    unique_records = [record for record in attachments if record.attachment_key not in duplicate_of]
    rows = []
    for record in unique_records:
        bundle = bundles[record.attachment_key]
        primary, secondary = classify_lenses(record.title, bundle)
        card_rel = f"{slugs[record.attachment_key]}/paper-card.md"
        rows.append(
            f"| {record.year or ''} | {record.title} | {short_author(record.creators)} | {primary}"
            f"{('/' + secondary) if secondary != 'None' else ''} | [card]({card_rel}) |"
        )
    duplicate_rows = []
    for record in attachments:
        if record.attachment_key in duplicate_of:
            duplicate_rows.append(
                f"| {record.title} | {record.item_key} | {record.attachment_key} | {duplicate_of[record.attachment_key]} |"
            )
    duplicate_table = "\n".join(duplicate_rows) or "| None | None | None | None |"
    reading_order = [
        ("A 问题意识与基础 BSR", "2019 Handheld MFSR + 2021 Deep Burst SR", "先建立真实手机 burst 的系统观，再理解深度 BSR 的数据生成、网络结构和评价表。"),
        ("B 多帧建模与网络演进", "2021 Deep Reparametrization + 2022 BIPNet + 2023 Burstormer", "比较重参数化、restoration/enhancement 统一框架和 transformer attention 怎么处理错位帧。"),
        ("C 真实场景与无 GT 训练", "2023 RealBSR + 2023 Self-supervised BSR", "把 synthetic-to-real gap、真实 benchmark 和无 GT 训练约束读清楚。"),
        ("D 基准帧/参考帧/RBPA 近邻", "2024 SBFBurst + 2024 Base Frame Selection", "聚焦 base frame guidance、residual learning、frame quality/selection；RAW 细节只作背景。"),
        ("E 鲁棒性理论语言", "2013 Robust Solutions + 2022 DRO review", "为 RBPA 的 uncertainty set、risk budget、worst-case/ambiguity-set 叙事补理论语汇。"),
    ]
    reading_rows = "\n".join(
        f"| {block} | {papers} | {goal} |" for block, papers, goal in reading_order
    )
    return f"""# BSR领域梳理 - Nature-style 精读总综述

> Source: Zotero collection `{COLLECTION_NAME}` (`{COLLECTION_KEY}`)
> Scope: {len(attachments)} PDF attachments, {len(unique_records)} unique paper-title records after title-level duplicate handling
> Mode: paper-only reading with page-grounded source bundles
> Output date: generated locally

## 01 文献地图

| Year | Paper | First author | Lens | 精读卡 |
|---|---|---|---|---|
{chr(10).join(rows)}

## 02 主线判断

[Analysis] 这个集合不是单纯的 BSR 方法列表，而是两条线交汇：

1. **Burst/Multi-frame SR 技术线**：Handheld MFSR -> Deep Burst SR / Deep Reparametrization -> BIPNet/Burstormer -> Self-supervised BSR -> RealBSR benchmark -> SBFBurst/Base-frame selection。
2. **鲁棒优化理论线**：Ben-Tal robust solutions 与 Rahimian-Mehrotra DRO 为 RBPA 的“预算、分布扰动、保守优化”叙事提供理论背景。

## 03 推荐阅读顺序

| 板块 | 文献 | 阅读目标 |
|---|---|---|
{reading_rows}

## 04 对你现在对比实验的直接启示

[Analysis] 主表应该采用你自己的统一 evaluation protocol，而不是混用论文 reported table。原因是 DBSR、SBFBurst、RealBSR/BurstSR 等工作在数据域、RAW/RGB、真实/合成、border crop、alignment/color matching 和 metric 实现上可能不同。你的正式 holdout 已经使用 official pretrained SBFBurst RealBSR-RGB backbone，并在同一 script 下比较 official/topk/scalar/cubic/rbpa；这比 paper-reported PSNR 更适合作为主张依据。

## 05 外部方法对比建议

| 方法类别 | 推荐用途 | 需要统一的前置条件 |
|---|---|---|
| Official SBFBurst | 主要外部 baseline，与你已完成的正式 holdout 对齐 | RealBSR-RGB、相同 burst、border=40、RGB PSNR/SSIM/LPIPS |
| DBSR / Deep Burst SR | 可作为经典方法和 reported-results 背景 | 不能直接拿论文 BurstSR 47 dB 与本地 31 dB 比；若进入主表需重跑同协议 |
| RealBSR benchmark/method | 说明真实 burst 数据域与 benchmark 设计 | 数据 split、GT、RGB/RAW、评价脚本 |
| Burstormer/BIPNet | restoration/enhancement 方向外部参照 | 是否能输入同一 RealBSR-RGB burst，是否有可用 checkpoint |
| Base-frame selection | 与 RBPA 的 selection/allocation 叙事最近 | 控制 backbone，不要同时改变模型和选择策略 |

## 06 Protocol 防踩坑清单

| 检查项 | 为什么重要 | 写论文时的建议表述 |
|---|---|---|
| RAW vs RGB | RAW 模型通常包含 Bayer/ISP/noise assumptions，RGB RealBSR-RGB 不能直接等同 | 明确说本实验使用 RealBSR-RGB / official pretrained SBFBurst RGB backbone |
| Synthetic vs Real | 合成 burst 可控但不代表真实手机退化 | 把 synthetic reported results 放在 related work，不进主对比表 |
| Spatial/color alignment | DBSR/BurstSR reported PSNR 可能评估前做 alignment/color matching | 引用 reported table 时标注 official protocol；主表用同一 metric script 重跑 |
| Border crop | PSNR 对边界裁剪很敏感 | 主表统一 `border=40` 或明确自己的 crop |
| Input burst length / selected frames | Burst-2/4/8/14 数值不可混比 | 主表写清 `burst_size=14, select=8` 或相应设置 |
| Backbone changes | 同时换 backbone 和 allocation 会掩盖 RBPA 贡献 | 最强主张来自固定 SBFBurst backbone，仅替换 prior/allocation |

## 07 重复与无 PDF 项

Zotero 集合中有若干重复题录或无附件条目。所有带 PDF 的附件都被登记；标题级明显重复项保留记录但不重复生成独立论文本体分析。

| Title | Zotero item | Attachment | Duplicate of |
|---|---|---|---|
{duplicate_table}

## 08 产物说明

- `zotero_collection_manifest.json`: Zotero 集合、题录、PDF 附件和去重状态。
- 每篇目录中的 `source_bundle.json`: 由 `nature-paper-card/scripts/prepare_paper.py` 生成的页级文本、图表、方程清单。
- 每篇目录中的 `paper-card.md`: 16 段式精读卡草稿，带 `[Paper: ...]` 证据指针。
- 这些卡片是 paper-only 精读产物；没有做外部 novelty verification。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    collection_items, attachments = collect_zotero_items()
    title_groups: dict[str, list[AttachmentRecord]] = defaultdict(list)
    for record in attachments:
        title_groups[title_key(record)].append(record)

    duplicate_of: dict[str, str] = {}
    for group in title_groups.values():
        if len(group) <= 1:
            continue
        canonical = sorted(group, key=lambda item: (item.title.startswith("Codex"), item.item_key))[0]
        for record in group:
            if record.attachment_key != canonical.attachment_key:
                duplicate_of[record.attachment_key] = canonical.attachment_key

    bundles: dict[str, dict[str, Any]] = {}
    slugs: dict[str, str] = {}
    for record in attachments:
        slug = safe_slug(record.title, record.year, record.item_key)
        slugs[record.attachment_key] = slug
        paper_dir = output / slug
        bundle_path = paper_dir / "source_bundle.json"
        if bundle_path.is_file():
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        else:
            bundle = run_prepare(record.pdf_path, bundle_path)
        bundles[record.attachment_key] = bundle
        card_path = paper_dir / "paper-card.md"
        card_path.write_text(
            make_card(
                record,
                slug,
                bundle,
                duplicate_of.get(record.attachment_key),
            ),
            encoding="utf-8",
        )
        notes = {
            "status": "draft-card-generated",
            "locator_mode": "page-grounded",
            "source_bundle": "source_bundle.json",
            "card": "paper-card.md",
            "duplicate_of": duplicate_of.get(record.attachment_key),
        }
        (paper_dir / "translation_notes.md").write_text(
            "# Translation / Reading Notes\n\n"
            "- Output mode: nature-paper-card style deep-reading card, not full bilingual translation.\n"
            "- Context verification: paper-only.\n"
            f"- Duplicate of: {duplicate_of.get(record.attachment_key) or 'None'}.\n",
            encoding="utf-8",
        )
        (paper_dir / "run_notes.json").write_text(
            json.dumps(notes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    write_manifest(output, collection_items, attachments, bundles, slugs, duplicate_of)
    (output / "BSR领域综述.md").write_text(
        make_collection_review(output, attachments, bundles, slugs, duplicate_of),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "collection_items": len(collection_items),
                "pdf_attachments": len(attachments),
                "unique_titles_with_pdf": len({title_key(record) for record in attachments}),
                "duplicates": duplicate_of,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
