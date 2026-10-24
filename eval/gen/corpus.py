# 用与入库相同的解析和切分流程读取样例条款，供合成评测集和切分评测使用
import hashlib
from dataclasses import dataclass
from functools import lru_cache

from clausetrail.chunking.splitter import chunk_document
from clausetrail.config import get_settings
from clausetrail.ingest.pipeline import doc_id_for
from clausetrail.ingest.router import body_blocks, parse_document
from clausetrail.structure.heading_classifier import assign_levels
from clausetrail.structure.ocr_correction import correct_blocks
from clausetrail.structure.tree import build_tree
from clausetrail.types import Chunk, SectionNode
from eval.common import SAMPLES

# 样例是固定的几份条款，产品名直接写在这里；用户提问时一般说简称
PRODUCTS = {
    "chinalife-critical-illness-kangning-yuexiang.pdf": ("国寿康宁悦享终身重大疾病保险", "康宁悦享"),
    "chinalife-medical-kangyibao-yixiang.pdf": ("国寿康医保易享医疗保险", "康医保易享"),
    "chinalife-whole-life-hengyu-yisheng.pdf": ("国寿恒裕一生终身寿险", "恒裕一生"),
    "pingan-long-term-medical-anyibao-exiang.pdf": ("平安安医保e享长期医疗保险", "安医保e享"),
}
DEV_SALT = "clausetrail-dev"
DEV_FRACTION = 0.2


@dataclass
class Doc:
    doc_id: str
    file_name: str
    product: str
    short: str
    root: SectionNode
    chunks: list[Chunk]


@lru_cache(maxsize=1)
def load_corpus() -> tuple[Doc, ...]:
    """只取 PDF 条款；合成的英文扫描件样例只用来演示 OCR 分流，不参与问答评测。"""
    settings = get_settings()
    docs = []
    for path in sorted(SAMPLES.glob("*.pdf")):
        doc_id = doc_id_for(path)
        blocks = body_blocks(parse_document(path, use_layout=settings.layout_analysis))
        correct_blocks(blocks)
        assign_levels(blocks)
        root = build_tree(blocks)
        chunks = chunk_document(
            root,
            doc_id,
            path.name,
            parser="deepdoc",
            max_tokens=settings.chunk_max_tokens,
            overlap_sentences=settings.chunk_overlap_sentences,
        )
        product, short = PRODUCTS.get(path.name, (path.stem, path.stem))
        docs.append(Doc(doc_id, path.name, product, short, root, chunks))
    return tuple(docs)


def assign_splits(doc_ids: list[str]) -> dict[str, str]:
    """按文档哈希排序后取前 20%（至少 1 份）做开发集，开发集和测试集不共享文档。"""
    ranked = sorted(doc_ids, key=lambda d: hashlib.sha256(f"{DEV_SALT}:{d}".encode()).hexdigest())
    n_dev = max(1, round(len(ranked) * DEV_FRACTION))
    return {d: "dev" if i < n_dev else "test" for i, d in enumerate(ranked)}
