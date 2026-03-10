"""Constrained JSON generation for vision-language models.

Adapted from jsonformer (https://github.com/1rgs/jsonformer) to support
Qwen3-VL and similar VLMs that require image inputs alongside text prompts.

The key idea: the model only generates *values* (strings, numbers, booleans).
All structural tokens (braces, brackets, colons, commas, key names) are
inserted programmatically, guaranteeing syntactically valid JSON output.

For instruction-tuned models, the partial JSON progress is placed as the
*assistant prefix* (after the generation prompt token), so the model
continues the JSON rather than starting a new response.
"""

from __future__ import annotations

import json
import logging
from typing import Any

_log = logging.getLogger(__name__)


class VLMJsonformer:
    """Constrained JSON generator for vision-language models."""

    GENERATION_MARKER = "|GENERATION|"

    def __init__(
        self,
        model,
        processor,
        json_schema: dict[str, Any],
        prompt: str,
        image,
        *,
        max_array_length: int = 20,
        max_number_tokens: int = 6,
        max_string_token_length: int = 30,
    ):
        self.model = model
        self.processor = processor
        self.json_schema = json_schema
        self.prompt = prompt
        self.image = image

        self.max_array_length = max_array_length
        self.max_number_tokens = max_number_tokens
        self.max_string_token_length = max_string_token_length

    def _prepare_inputs(self, assistant_prefix: str = "") -> dict:
        """Prepare model inputs with task in user msg, partial JSON as assistant prefix."""
        user_text = (
            f"{self.prompt}\n"
            f"Output JSON matching this schema:\n{json.dumps(self.json_schema)}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Append partial JSON as the start of the assistant response
        text_input += assistant_prefix

        inputs = self.processor(
            text=[text_input],
            images=[self.image],
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def _get_progress(self, value: dict | list) -> str:
        """Build the assistant prefix from generation progress so far."""
        progress = json.dumps(value)
        marker_idx = progress.find(f'"{self.GENERATION_MARKER}"')
        if marker_idx == -1:
            marker_idx = progress.find(f"{self.GENERATION_MARKER}")
        if marker_idx != -1:
            progress = progress[:marker_idx]
        return progress

    def generate_number(self, value: dict | list) -> float:
        """Let the model generate a number value."""
        import torch

        prefix = self._get_progress(value)
        inputs = self._prepare_inputs(prefix)

        with torch.no_grad():
            response = self.model.generate(
                **inputs,
                max_new_tokens=self.max_number_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = response[0, input_len:]
        text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = text.strip().rstrip(".,}")

        # Extract first valid number
        num_str = ""
        for ch in text:
            if ch.isdigit() or ch == "." or (ch == "-" and not num_str):
                num_str += ch
            else:
                break

        try:
            val = float(num_str) if num_str else 0.0
            return int(val) if val == int(val) else val
        except ValueError:
            return 0

    def generate_boolean(self, value: dict | list) -> bool:
        """Let the model decide true/false."""
        import torch

        prefix = self._get_progress(value)
        inputs = self._prepare_inputs(prefix)

        with torch.no_grad():
            output = self.model.forward(**inputs)

        logits = output.logits[0, -1]

        true_id = self.processor.tokenizer.convert_tokens_to_ids("true")
        false_id = self.processor.tokenizer.convert_tokens_to_ids("false")

        if isinstance(true_id, int) and isinstance(false_id, int):
            return bool(logits[true_id] > logits[false_id])

        # Fallback: generate a token and check
        with torch.no_grad():
            response = self.model.generate(
                **inputs, max_new_tokens=3, do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        text = self.processor.tokenizer.decode(
            response[0, input_len:], skip_special_tokens=True
        ).strip().lower()
        return text.startswith("true")

    def generate_string(self, value: dict | list) -> str:
        """Let the model generate a string value."""
        import torch

        prefix = self._get_progress(value) + '"'
        inputs = self._prepare_inputs(prefix)

        with torch.no_grad():
            response = self.model.generate(
                **inputs,
                max_new_tokens=self.max_string_token_length,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = response[0, input_len:]
        text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Take text up to the closing quote
        if '"' in text:
            text = text.split('"')[0]
        return text.strip()

    def _should_continue_array(self, value: dict | list) -> bool:
        """Ask model if array should continue (comma) or end (bracket)."""
        import torch

        prefix = self._get_progress(value)
        inputs = self._prepare_inputs(prefix)

        with torch.no_grad():
            output = self.model.forward(**inputs)

        logits = output.logits[0, -1]
        top_ids = logits.topk(30).indices
        sorted_ids = top_ids[logits[top_ids].argsort(descending=True)]

        for token_id in sorted_ids:
            decoded = self.processor.tokenizer.decode(token_id)
            if "," in decoded:
                return True
            if "]" in decoded:
                return False

        return False

    def generate_value(
        self,
        schema: dict[str, Any],
        obj: dict | list,
        key: str | None = None,
    ) -> Any:
        schema_type = schema["type"]

        if schema_type == "number":
            if key is not None:
                obj[key] = self.GENERATION_MARKER
            elif isinstance(obj, list):
                obj.append(self.GENERATION_MARKER)
            return self.generate_number(self._root_value)

        elif schema_type == "boolean":
            if key is not None:
                obj[key] = self.GENERATION_MARKER
            elif isinstance(obj, list):
                obj.append(self.GENERATION_MARKER)
            return self.generate_boolean(self._root_value)

        elif schema_type == "string":
            if key is not None:
                obj[key] = self.GENERATION_MARKER
            elif isinstance(obj, list):
                obj.append(self.GENERATION_MARKER)
            return self.generate_string(self._root_value)

        elif schema_type == "array":
            new_array: list = []
            if key is not None:
                obj[key] = new_array
            elif isinstance(obj, list):
                obj.append(new_array)
            return self.generate_array(schema["items"], new_array)

        elif schema_type == "object":
            new_obj: dict = {}
            if key is not None:
                obj[key] = new_obj
            elif isinstance(obj, list):
                obj.append(new_obj)
            return self.generate_object(schema["properties"], new_obj)

        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

    def generate_object(
        self, properties: dict[str, Any], obj: dict
    ) -> dict:
        for key, schema in properties.items():
            obj[key] = self.generate_value(schema, obj, key)
        return obj

    def generate_array(
        self, item_schema: dict[str, Any], arr: list
    ) -> list:
        for i in range(self.max_array_length):
            element = self.generate_value(item_schema, arr)
            arr[-1] = element

            # Check if array should continue
            if i == 0:
                continue  # Force at least one element
            arr.append(self.GENERATION_MARKER)
            should_continue = self._should_continue_array(self._root_value)
            arr.pop()
            if not should_continue:
                break

        return arr

    def __call__(self) -> Any:
        """Generate a JSON value conforming to the schema."""
        schema_type = self.json_schema.get("type", "object")

        if schema_type == "object":
            self._root_value = {}
            return self.generate_object(
                self.json_schema["properties"], self._root_value
            )
        elif schema_type == "array":
            self._root_value = []
            return self.generate_array(
                self.json_schema["items"], self._root_value
            )
        else:
            raise ValueError(
                f"Root schema type must be 'object' or 'array', got '{schema_type}'"
            )
