import logging
from functools import lru_cache

import cv2
import numpy as np

from clausetrail.ingest.native_pdf import HEADER_FOOTER_BAND
from clausetrail.types import Block, BlockKind, ParserName

log = logging.getLogger(__name__)
_warned_paddle = False

MIN_SKEW_DEGREES = 0.3
MAX_SKEW_DEGREES = 15.0


def denoise(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.medianBlur(gray, 3)
    return cv2.fastNlMeansDenoising(blurred, h=10, templateWindowSize=7, searchWindowSize=21)


def estimate_skew(gray: np.ndarray) -> float:
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    # minAreaRect 返回的角度区间随 OpenCV 版本不同，统一折到 [-45°, 45°]；
    # 超过 MAX_SKEW_DEGREES 不像扫描歪斜，按估计失败处理
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    return float(-angle) if abs(angle) <= MAX_SKEW_DEGREES else 0.0


def deskew(gray: np.ndarray, angle: float) -> np.ndarray:
    # 角度很小时不旋转，省掉一次会让字变糊的插值重采样
    if abs(angle) < MIN_SKEW_DEGREES:
        return gray
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    clean = denoise(gray)
    angle = estimate_skew(clean)
    return cv2.cvtColor(deskew(clean, angle), cv2.COLOR_GRAY2RGB), angle


@lru_cache(maxsize=1)
def _paddle():
    from paddleocr import PaddleOCR

    return PaddleOCR(lang="ch", use_textline_orientation=True)


@lru_cache(maxsize=1)
def _deepdoc_ocr():
    from deepdoc.vision import OCR

    return OCR()


def paddle_available() -> bool:
    try:
        import paddleocr  # noqa: F401
    except ImportError:
        return False
    return True


def _run_paddle(image: np.ndarray) -> list[tuple[tuple[float, float, float, float], str, float]]:
    out = []
    for page in _paddle().predict(image):
        for poly, text, score in zip(page["rec_polys"], page["rec_texts"], page["rec_scores"], strict=True):
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            out.append(((min(xs), min(ys), max(xs), max(ys)), text, float(score)))
    return out


def _run_deepdoc(image: np.ndarray) -> list[tuple[tuple[float, float, float, float], str, float]]:
    out = []
    for box, (text, score) in _deepdoc_ocr()(image):
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        out.append(((min(xs), min(ys), max(xs), max(ys)), text, float(score)))
    return out


# 和原生页同一口径：页面上下各 HEADER_FOOTER_BAND 的行算页眉页脚，由 router.body_blocks 滤掉，不进正文。
# 只用于 PDF 里的扫描页：手机拍的照片没有固定页边，第一行、最后一行往往就是条款正文
def _band_kind(bbox, height: float, bands: bool = True) -> BlockKind:
    if not bands:
        return BlockKind.TEXT
    if bbox[3] < height * HEADER_FOOTER_BAND:
        return BlockKind.HEADER
    if bbox[1] > height * (1 - HEADER_FOOTER_BAND):
        return BlockKind.FOOTER
    return BlockKind.TEXT


def ocr_image(rgb: np.ndarray, page_no: int, scale: float = 1.0, bands: bool = True) -> list[Block]:
    clean, angle = preprocess(rgb)
    if paddle_available():
        lines, parser = _run_paddle(clean), ParserName.PADDLEOCR
    else:
        global _warned_paddle
        # 只提示一次，免得每个扫描页打一条
        if not _warned_paddle:
            _warned_paddle = True
            log.warning("paddleocr 未安装，扫描页改用 DeepDoc OCR（pip install '.[ocr]' 可启用 PaddleOCR）")
        lines, parser = _run_deepdoc(clean), ParserName.DEEPDOC_OCR
    left = min((b[0] for b, _, _ in lines), default=0.0)
    height = clean.shape[0]
    # 扫描页由 router 按 SCAN_RENDER_DPI 渲染后识别，坐标除以 scale 换回 PDF 坐标；
    # OCR 没有字号信息，用行框高度近似，供标题特征使用
    blocks = [
        Block(
            page=page_no,
            bbox=tuple(v / scale for v in bbox),
            text=text,
            kind=_band_kind(bbox, height, bands),
            font_size=(bbox[3] - bbox[1]) / scale,
            indent=(bbox[0] - left) / scale,
            parser=parser,
            ocr_confidence=score,
        )
        for bbox, text, score in lines
        if text.strip()
    ]
    blocks.sort(key=lambda b: (round(b.bbox[1] / 4), b.bbox[0]))
    log.debug("page %s deskew=%.2f° lines=%d parser=%s", page_no, angle, len(blocks), parser)
    return blocks
