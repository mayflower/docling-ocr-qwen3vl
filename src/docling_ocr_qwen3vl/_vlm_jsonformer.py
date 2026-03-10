"""Constrained JSON generation for vision-language models.

Adapted from jsonformer (https://github.com/1rgs/jsonformer) to support
Qwen3-VL and similar VLMs that require image inputs alongside text prompts.

Provides two strategies:
1. **Single-shot** (default): One model.generate() call with the partial
   JSON injected as an assistant prefix.  Fast (~1 forward pass) but the
   model may still produce broken JSON — a repair step patches common
   issues (trailing commas, unclosed brackets).
2. **Per-value (jsonformer)**: The model generates only values; all
   structural tokens are inserted programmatically.  Guarantees valid
   JSON but requires O(fields × elements) forward passes — very slow.

For instruction-tuned models, the partial JSON progress is placed as the
*assistant prefix* (after the generation prompt token), so the model
continues the JSON rather than starting a new response.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-shot generation with assistant prefix
# ---------------------------------------------------------------------------


def _fix_corrupted_keys(text: str) -> str:
    """Fix corrupted JSON keys like "y1:890 → "y1":890."""
    # Pattern: "key_name:value" where the closing quote is missing before colon
    return re.sub(r'"(\w+):(\d)', r'"\1":\2', text)


def _repair_json_array(text: str) -> str:
    """Try to fix common JSON issues in array output from small VLMs."""
    s = text.strip()
    s = _fix_corrupted_keys(s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    if s.startswith("[") and not s.endswith("]"):
        last_brace = s.rfind("}")
        if last_brace > 0:
            s = s[: last_brace + 1] + "]"
    return s


def _extract_valid_elements(text: str) -> list:
    """Extract individually valid JSON objects from a corrupted array.

    When the overall array JSON is broken, try to salvage valid objects
    by splitting on ``},{`` boundaries and parsing each one.
    """
    objects = re.findall(r"\{[^{}]*\}", text)
    results = []
    for obj_str in objects:
        fixed = _fix_corrupted_keys(obj_str)
        try:
            results.append(json.loads(fixed))
        except json.JSONDecodeError:
            continue
    return results


def _repair_json_object(text: str) -> str:
    """Try to fix common JSON issues in object output from small VLMs."""
    s = text.strip()
    s = _fix_corrupted_keys(s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    opens = s.count("{") + s.count("[")
    closes = s.count("}") + s.count("]")
    if opens > closes:
        arr_diff = s.count("[") - s.count("]")
        obj_diff = s.count("{") - s.count("}")
        s += "]" * max(arr_diff, 0) + "}" * max(obj_diff, 0)
    return s


def generate_json_single_shot(
    model,
    processor,
    prompt: str,
    image,
    *,
    max_new_tokens: int = 4096,
    root_type: str = "array",
) -> Any:
    """Generate JSON in one model.generate() call with assistant prefix.

    The prompt should contain the full task description, format spec,
    and examples.  The opening bracket (``[`` for arrays, ``{`` for
    objects) is placed as the assistant prefix so the model continues
    the JSON directly, avoiding markdown code fences.

    Returns the parsed Python object (list or dict), or an empty
    list/dict on failure.
    """
    import torch

    user_text = prompt

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Inject assistant prefix — the opening bracket
    prefix = "[" if root_type == "array" else "{"
    text_input += prefix

    inputs = processor(
        text=[text_input],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0, input_len:]

    # Strip thinking tokens if present
    from ._model_registry import extract_after_think_token

    new_tokens = extract_after_think_token(new_tokens.unsqueeze(0)).squeeze(0)

    raw_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    # Prepend the prefix we injected
    full_text = prefix + raw_text.strip()

    _log.debug("Single-shot raw output: %s", full_text[:500])

    # --- Parse with repair fallback ---
    if root_type == "array":
        return _parse_array(full_text)
    else:
        return _parse_object(full_text)


def _parse_array(text: str) -> list:
    """Parse a JSON array with repair fallback."""
    json_match = re.search(r"\[[\s\S]*\]", text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            repaired = _repair_json_array(json_match.group())
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    # Try to close a partial array
    partial = re.search(r"\[[\s\S]*", text)
    if partial:
        repaired = _repair_json_array(partial.group())
        try:
            result = json.loads(repaired)
            _log.info("Array JSON repaired (%d elements)", len(result))
            return result
        except json.JSONDecodeError:
            pass

    # Last resort: extract individual valid objects from corrupted text
    salvaged = _extract_valid_elements(text)
    if salvaged:
        _log.info("Salvaged %d elements from corrupted array JSON", len(salvaged))
        return salvaged

    _log.warning("Failed to parse array JSON.\nRaw: %s", text[:300])
    return []


def _parse_object(text: str) -> dict:
    """Parse a JSON object with repair fallback."""
    # Apply key repair upfront
    fixed = _fix_corrupted_keys(text)

    json_match = re.search(r"\{[\s\S]*\}", fixed)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            repaired = _repair_json_object(json_match.group())
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    partial = re.search(r"\{[\s\S]*", fixed)
    if partial:
        repaired = _repair_json_object(partial.group())
        try:
            result = json.loads(repaired)
            _log.info("Object JSON repaired (%d keys)", len(result))
            return result
        except json.JSONDecodeError as e:
            _log.warning("Failed to parse object JSON: %s\nRaw: %s", e, text[:300])

    return {}


# ---------------------------------------------------------------------------
# Per-value jsonformer (slow but guaranteed-valid fallback)
# ---------------------------------------------------------------------------


class VLMJsonformer:
    """Constrained JSON generator for vision-language models.

    Each value (string, number, boolean) is generated by a separate
    model call.  Structural tokens are inserted programmatically.
    This guarantees syntactically valid JSON but is very slow.
    """

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

            if i == 0:
                continue
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
