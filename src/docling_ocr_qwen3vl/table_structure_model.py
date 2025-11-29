"""Qwen3-VL Table Structure Model for Docling."""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Sequence
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import BoundingBox, Page, Table, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import BaseTableStructureOptions
from docling.models.base_table_model import BaseTableStructureModel
from docling_core.types.doc import DocItemLabel, TableCell

from .options import Qwen3VlQuantization, Qwen3VlTableStructureOptions
from .prompts import TABLE_STRUCTURE_PROMPT


_log = logging.getLogger(__name__)

# Global lock for model initialization
_model_init_lock = threading.Lock()

# Token ID for </think> in Qwen3 Thinking models
_THINK_END_TOKEN_ID = 151668


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
    ):
        self.enabled = enabled
        self.options = options
        self._processor = None
        self._model = None
        self._device = None

        if self.enabled:
            self._load_model()

    def _load_model(self) -> None:
        """Load the Qwen3-VL model."""
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Missing dependencies for Qwen3-VL. Install `torch` and `transformers>=4.51.0`."
            ) from exc

        requested_device = self.options.device or "cuda"
        if "cuda" not in requested_device:
            raise RuntimeError("Qwen3-VL requires a CUDA device.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device not available but required by Qwen3-VL.")

        torch_device = torch.device(requested_device)
        if torch_device.index is not None:
            torch.cuda.set_device(torch_device.index)

        torch_dtype = self._resolve_torch_dtype(self.options.dtype)

        with _model_init_lock:
            _log.info("Loading Qwen3-VL processor for table structure...")
            self._processor = AutoProcessor.from_pretrained(
                self.options.model_repo_id,
                trust_remote_code=self.options.trust_remote_code,
                token=self.options.hf_token,
            )

            attn_impl = self._select_attention_backend(self.options.attn_implementation)

            model_kwargs: dict = {
                "trust_remote_code": self.options.trust_remote_code,
                "device_map": "auto",
            }
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            if attn_impl:
                model_kwargs["attn_implementation"] = attn_impl
            if self.options.hf_token:
                model_kwargs["token"] = self.options.hf_token

            quantization_config = self._create_quantization_config()
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
                _log.info("Using %s quantization", self.options.quantization.value)

            _log.info("Loading Qwen3-VL model for table structure...")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.options.model_repo_id,
                **model_kwargs,
            )
            model = model.eval()

            self._model = model
            self._device = torch_device

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

        text_input = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[text_input],
            images=[image_rgb],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.options.max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_len:]
        generated_ids = self._extract_after_think_token(generated_ids)

        output_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        self._maybe_empty_cache()

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

    def _extract_after_think_token(self, generated_ids):
        """Extract tokens after </think> token for Thinking models."""
        ids = generated_ids[0].tolist()
        try:
            reversed_ids = ids[::-1]
            pos_from_end = reversed_ids.index(_THINK_END_TOKEN_ID)
            index = len(ids) - pos_from_end
        except ValueError:
            return generated_ids

        import torch

        return torch.tensor([ids[index:]], device=generated_ids.device)

    def _resolve_torch_dtype(self, dtype_name: str | None):
        if dtype_name is None or dtype_name == "auto":
            return "auto"
        try:
            import torch
        except ImportError:
            return None
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(dtype_name.lower(), getattr(torch, dtype_name.lower(), None))

    def _select_attention_backend(self, requested: str) -> str | None:
        if requested == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                _log.warning("flash-attn not installed; falling back to eager attention.")
                return "eager"
        return requested

    def _create_quantization_config(self):
        if self.options.quantization == Qwen3VlQuantization.NONE:
            return None
        try:
            import torch
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError("BitsAndBytes quantization requires `bitsandbytes` package.") from exc

        if self.options.quantization == Qwen3VlQuantization.INT8:
            return BitsAndBytesConfig(load_in_8bit=True)
        if self.options.quantization == Qwen3VlQuantization.INT4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.options.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.options.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        return None

    @staticmethod
    def _maybe_empty_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
