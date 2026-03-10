"""Qwen3-VL Table Structure Model for Docling."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import BoundingBox, Page, Table, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.models.base_table_model import BaseTableStructureModel
from docling_core.types.doc import DocItemLabel, TableCell

try:
    from docling.datamodel.pipeline_options import BaseTableStructureOptions
except ImportError:
    from docling.datamodel.pipeline_options import (
        TableStructureOptions as BaseTableStructureOptions,
    )

from ._model_registry import extract_after_think_token, get_model, maybe_empty_cache
from .options import Qwen3VlTableStructureOptions
from .prompts import TABLE_STRUCTURE_PROMPT


_log = logging.getLogger(__name__)


class Qwen3VlTableStructureModel(BaseTableStructureModel):
    """Qwen3-VL based table structure detection model.

    Uses the Qwen3-VL vision-language model to detect table structure
    including rows, columns, cells, and spans.
    """

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: Qwen3VlTableStructureOptions,
        accelerator_options: AcceleratorOptions,
        **kwargs,
    ):
        self.enabled = enabled
        self.options = options
        self._shared = None

        if self.enabled:
            self._shared = get_model(
                model_repo_id=self.options.model_repo_id,
                device=self.options.device,
                dtype=self.options.dtype,
                trust_remote_code=self.options.trust_remote_code,
                hf_token=self.options.hf_token,
                attn_implementation=self.options.attn_implementation,
                quantization=self.options.quantization,
                bnb_4bit_quant_type=self.options.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.options.bnb_4bit_use_double_quant,
            )

    @classmethod
    def get_options_type(cls) -> type[BaseTableStructureOptions]:
        return Qwen3VlTableStructureOptions

    def predict_tables(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[TableStructurePrediction]:
        """Produce table structure predictions for the provided pages."""
        predictions = []

        for page in pages:
            table_map: dict[int, Table] = {}

            # Find table clusters from layout prediction
            if page.predictions.layout is None:
                predictions.append(TableStructurePrediction(table_map={}))
                continue

            table_clusters = [
                c for c in page.predictions.layout.clusters if c.label == DocItemLabel.TABLE
            ]

            for cluster in table_clusters:
                # Get table image crop
                table_image = page.get_image(scale=2.0, cropbox=cluster.bbox)
                if table_image is None:
                    continue

                # Run Qwen3-VL inference on table crop
                table_data = self._extract_table_structure(table_image, cluster.bbox, page)

                if table_data:
                    table_map[cluster.id] = table_data

            predictions.append(TableStructurePrediction(table_map=table_map))

        return predictions

    def _extract_table_structure(
        self,
        table_image,
        table_bbox: BoundingBox,
        page: Page,
    ) -> Table | None:
        """Extract table structure from an image using Qwen3-VL."""
        import torch

        assert self._shared is not None
        model = self._shared.model
        processor = self._shared.processor

        image_rgb = table_image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rgb},
                    {"type": "text", "text": TABLE_STRUCTURE_PROMPT},
                ],
            }
        ]

        text_input = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[text_input],
            images=[image_rgb],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.options.max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_len:]
        generated_ids = extract_after_think_token(generated_ids)

        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        maybe_empty_cache()

        # Parse JSON output
        try:
            # Extract JSON from output
            json_match = re.search(r"\{[\s\S]*\}", output_text)
            if not json_match:
                _log.warning("No JSON found in table structure output")
                return None

            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            _log.warning("Failed to parse table structure JSON: %s", e)
            return None

        # Convert to Table object
        return self._build_table(data, table_bbox, page)

    def _build_table(
        self,
        data: dict,
        table_bbox: BoundingBox,
        page: Page,
    ) -> Table | None:
        """Build a Table object from parsed JSON data."""
        num_rows = data.get("rows", 0)
        num_cols = data.get("cols", 0)
        cells_data = data.get("cells", [])

        if num_rows == 0 or num_cols == 0:
            return None

        table_cells: list[TableCell] = []
        otsl_seq: list[str] = []

        # Build OTSL sequence for table structure
        for row_idx in range(num_rows):
            if row_idx > 0:
                otsl_seq.append("nl")
            for col_idx in range(num_cols):
                if col_idx > 0:
                    otsl_seq.append("l")
                otsl_seq.append("cell")

        # Convert cells
        for cell_data in cells_data:
            row = cell_data.get("row", 0)
            col = cell_data.get("col", 0)
            row_span = cell_data.get("row_span", 1)
            col_span = cell_data.get("col_span", 1)
            text = cell_data.get("text", "")
            is_header = cell_data.get("is_header", False)
            bbox = cell_data.get("bbox", None)

            # Convert bbox from 0-1000 scale to page coordinates
            cell_bbox = None
            if bbox and len(bbox) == 4 and page.size:
                x1, y1, x2, y2 = bbox
                # Scale from 0-1000 to table bbox, then to page coordinates
                table_width = table_bbox.r - table_bbox.l
                table_height = table_bbox.b - table_bbox.t
                cell_bbox = BoundingBox(
                    l=table_bbox.l + (x1 / 1000) * table_width,
                    t=table_bbox.t + (y1 / 1000) * table_height,
                    r=table_bbox.l + (x2 / 1000) * table_width,
                    b=table_bbox.t + (y2 / 1000) * table_height,
                )

            table_cells.append(
                TableCell(
                    text=text,
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=row,
                    end_row_offset_idx=row + row_span,
                    start_col_offset_idx=col,
                    end_col_offset_idx=col + col_span,
                    col_header=is_header,
                    row_header=False,
                    bbox=cell_bbox,
                )
            )

        # Create placeholder cluster
        from docling.datamodel.base_models import Cluster

        cluster = Cluster(
            id=0,
            label=DocItemLabel.TABLE,
            bbox=table_bbox,
        )

        return Table(
            label=DocItemLabel.TABLE,
            id=0,
            page_no=page.page_no,
            cluster=cluster,
            otsl_seq=otsl_seq,
            num_rows=num_rows,
            num_cols=num_cols,
            table_cells=table_cells,
        )
