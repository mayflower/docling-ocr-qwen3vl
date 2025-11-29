"""Docling integration of the Qwen3-VL OCR engine."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrOptions
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from .options import Qwen3VlOcrOptions
from .qwen_runner import Qwen3VlRunner


_log = logging.getLogger(__name__)


class Qwen3VlOcrModel(BaseOcrModel):
    """Bridge Qwen3-VL OCR outputs into Docling's pipeline."""

    runner: Qwen3VlRunner

    def __init__(
        self,
        *,
        enabled: bool,
        artifacts_path: Path | None,
        options: Qwen3VlOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options = options
        self.runner = Qwen3VlRunner(options=options)
        self._debug_logged_pages: set[int] = set()
        self._page_scale = max(1.0, options.page_scale)

    def __call__(self, conv_res: ConversionResult, page_batch: Iterable[Page]) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        self.runner.ensure_loaded()

        for page in page_batch:
            if page._backend is None or not page._backend.is_valid():
                yield page
                continue

            ocr_rects = self.get_ocr_rects(page)
            if not ocr_rects:
                yield page
                continue

            with TimeRecorder(conv_res, "ocr"):
                all_ocr_cells: list[TextCell] = []
                for rect in ocr_rects:
                    width = rect.r - rect.l
                    height = rect.b - rect.t
                    if width <= 0 or height <= 0:
                        continue

                    try:
                        high_res_image = page._backend.get_page_image(
                            scale=self._page_scale, cropbox=rect
                        )
                    except Exception as exc:
                        _log.warning("Failed to rasterize OCR region: %s", exc)
                        continue

                    if high_res_image is None:
                        continue

                    result = self.runner.run(
                        high_res_image,
                        prompt_mode=self.options.prompt_mode,
                    )

                    # Use HTML elements with real bounding boxes if available
                    if result.html_elements:
                        cells = self._html_elements_to_cells(
                            result.html_elements,
                            rect=rect,
                            image_size=high_res_image.size,
                            index_offset=len(all_ocr_cells),
                        )
                    else:
                        cells = self._paragraphs_to_cells(
                            result.paragraphs,
                            rect=rect,
                            index_offset=len(all_ocr_cells),
                        )

                    all_ocr_cells.extend(cells)

                self.post_process_cells(all_ocr_cells, page)

                if page.page_no not in self._debug_logged_pages:
                    sample = ", ".join(cell.text[:50] for cell in all_ocr_cells[:3])
                    _log.info(
                        "Qwen3-VL OCR page %s produced %s cells: %s",
                        page.page_no,
                        len(all_ocr_cells),
                        sample,
                    )
                    self._debug_logged_pages.add(page.page_no)

            if settings.debug.visualize_ocr:
                self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

            yield page

    @classmethod
    def get_options_type(cls) -> type[OcrOptions]:
        return Qwen3VlOcrOptions

    @staticmethod
    def _paragraphs_to_cells(
        paragraphs: list[str],
        *,
        rect: BoundingBox,
        index_offset: int = 0,
    ) -> list[TextCell]:
        """Convert paragraphs into Docling TextCells with approximate positions.

        For non-HTML modes, we distribute paragraphs evenly across the vertical
        space of the OCR region. For accurate bounding boxes, use QWENVL_HTML mode.
        """
        cells: list[TextCell] = []

        if not paragraphs:
            return cells

        region_height = rect.b - rect.t
        region_width = rect.r - rect.l

        # Distribute paragraphs evenly across the vertical space
        num_paragraphs = len(paragraphs)
        paragraph_height = region_height / num_paragraphs

        for idx, text in enumerate(paragraphs):
            text = text.strip()
            if not text:
                continue

            # Calculate approximate vertical position for this paragraph
            top = rect.t + (idx * paragraph_height)
            bottom = rect.t + ((idx + 1) * paragraph_height)

            bbox = BoundingBox(
                l=rect.l,
                t=top,
                r=rect.l + region_width,
                b=bottom,
                coord_origin=CoordOrigin.TOPLEFT,
            )

            rect_obj = BoundingRectangle.from_bounding_box(bbox)
            cells.append(
                TextCell(
                    index=index_offset + len(cells),
                    text=text,
                    orig=text,
                    from_ocr=True,
                    confidence=1.0,
                    rect=rect_obj,
                )
            )

        return cells

    @staticmethod
    def _html_elements_to_cells(
        html_elements: list,
        *,
        rect: BoundingBox,
        image_size: tuple[int, int],
        index_offset: int = 0,
    ) -> list[TextCell]:
        """Convert HTML elements with bounding boxes into Docling TextCells.

        The bounding boxes from QwenVL HTML are in image pixel coordinates.
        We scale them to match the OCR region in document coordinates.
        """
        from .qwen_runner import HtmlElement

        cells: list[TextCell] = []

        if not html_elements:
            return cells

        img_width, img_height = image_size
        region_width = rect.r - rect.l
        region_height = rect.b - rect.t

        # Scale factors to convert image coordinates to document coordinates
        scale_x = region_width / img_width if img_width > 0 else 1.0
        scale_y = region_height / img_height if img_height > 0 else 1.0

        for element in html_elements:
            if not isinstance(element, HtmlElement):
                continue

            text = element.text.strip()
            if not text:
                continue

            if element.bbox:
                # Use real bounding box from QwenVL HTML
                x1, y1, x2, y2 = element.bbox
                bbox = BoundingBox(
                    l=rect.l + (x1 * scale_x),
                    t=rect.t + (y1 * scale_y),
                    r=rect.l + (x2 * scale_x),
                    b=rect.t + (y2 * scale_y),
                    coord_origin=CoordOrigin.TOPLEFT,
                )
            else:
                # Fall back to full region if no bbox available
                bbox = BoundingBox(
                    l=rect.l,
                    t=rect.t,
                    r=rect.r,
                    b=rect.b,
                    coord_origin=CoordOrigin.TOPLEFT,
                )

            rect_obj = BoundingRectangle.from_bounding_box(bbox)
            cells.append(
                TextCell(
                    index=index_offset + len(cells),
                    text=text,
                    orig=text,
                    from_ocr=True,
                    confidence=1.0,
                    rect=rect_obj,
                )
            )

        return cells
