"""Explicit local Qwen3-VL backbone boundary for the action expert."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn


class QwenBackboneError(RuntimeError):
    """Raised when the local Qwen backbone cannot be prepared or loaded."""


@dataclass(frozen=True)
class QwenBackboneConfig:
    model_path: Path = field(default_factory=lambda: Path(
        os.environ.get("DUVLA_QWEN_PATH", str(Path.home() / "models/qwen/Qwen3-VL-2B-Instruct"))
    ).expanduser())
    freeze: bool = True
    local_files_only: bool = True
    use_bfloat16: bool = True
    hidden_layer: int = -1

    def __post_init__(self) -> None:
        if self.hidden_layer < -1:
            raise ValueError("hidden_layer must be -1 or a non-negative hidden-state index")


class QwenVLBackbone(nn.Module):
    """Lazy-loading Qwen3-VL wrapper with pooled and token-context outputs.

    The wrapper does not import Transformers or construct the 4.26GB model
    until :meth:`load` is called.  This keeps unit tests offline and prevents
    an accidental network download during development.
    """

    def __init__(self, config: QwenBackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config or QwenBackboneConfig()
        self.model: nn.Module | None = None
        self.processor: Any | None = None
        self.output_dim = 2048

    @staticmethod
    def pool_hidden_states(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Masked-mean pool Qwen hidden states to ``[batch, hidden_dim]``."""

        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, tokens, hidden_dim]")
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention_mask must have shape [batch, tokens]")
        mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denominator

    @staticmethod
    def reduce_hidden_states(
        hidden_states: Tensor,
        attention_mask: Tensor,
        num_tokens: int,
    ) -> Tensor:
        """Compress valid hidden states into a fixed number of ordered tokens.

        Qwen's multimodal sequence can be much longer than an action expert can
        afford to cache for every LIBERO frame.  This deterministic temporal
        binning keeps the original token order while producing ``[B,K,D]``.
        It is intentionally a transparent reduction, not a learned pooling
        layer, so the frozen-backbone cache remains reproducible.
        """

        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, tokens, hidden_dim]")
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention_mask must have shape [batch, tokens]")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        reduced: list[Tensor] = []
        for row, mask in zip(hidden_states, attention_mask):
            valid = row[mask.to(dtype=torch.bool)]
            if valid.shape[0] == 0:
                raise ValueError("each sample must contain at least one valid token")
            if valid.shape[0] < num_tokens:
                positions = torch.linspace(
                    0, valid.shape[0] - 1, num_tokens, device=valid.device
                ).round().to(dtype=torch.long)
                reduced.append(valid.index_select(0, positions))
                continue
            boundaries = torch.linspace(
                0, valid.shape[0], num_tokens + 1, device=valid.device
            ).round().to(dtype=torch.long)
            bins = [
                valid[boundaries[index] : boundaries[index + 1]].mean(dim=0)
                for index in range(num_tokens)
            ]
            reduced.append(torch.stack(bins, dim=0))
        return torch.stack(reduced, dim=0)

    def load(self, *, device: torch.device | str | None = None) -> QwenVLBackbone:
        """Load local Qwen weights and processor exactly once."""

        if self.model is not None:
            return self
        if not self.config.model_path.is_dir():
            raise QwenBackboneError(f"Qwen model directory does not exist: {self.config.model_path}")
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ModuleNotFoundError as exc:
            raise QwenBackboneError(
                "Transformers >= 4.57 is required; install it in your DuVLA environment"
            ) from exc
        dtype = torch.bfloat16 if self.config.use_bfloat16 else torch.float32
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                str(self.config.model_path),
                dtype=dtype,
                local_files_only=self.config.local_files_only,
                low_cpu_mem_usage=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                str(self.config.model_path), local_files_only=self.config.local_files_only
            )
        except Exception as exc:
            self.model = None
            self.processor = None
            raise QwenBackboneError(f"failed to load local Qwen backbone: {exc}") from exc
        if self.config.freeze:
            self.model.requires_grad_(False)
            self.model.eval()
        if device is not None:
            self.model.to(device)
        text_config = getattr(getattr(self.model, "config", None), "text_config", None)
        self.output_dim = int(getattr(text_config, "hidden_size", self.output_dim))
        return self

    def prepare_inputs(self, image: Any, wrist_image: Any, instruction: str) -> Any:
        """Build one Qwen multimodal input using two explicit camera views."""

        if self.processor is None:
            raise QwenBackboneError("call load() before prepare_inputs()")
        if not instruction.strip():
            raise ValueError("instruction cannot be empty")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "image", "image": wrist_image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    def prepare_view_inputs(self, image: Any, instruction: str) -> Any:
        """Build one single-camera Qwen input for camera-separated caching."""

        if self.processor is None:
            raise QwenBackboneError("call load() before prepare_view_inputs()")
        if not instruction.strip():
            raise ValueError("instruction cannot be empty")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    @staticmethod
    def collate_processor_inputs(
        inputs: Sequence[dict[str, Tensor]], *, pad_token_id: int
    ) -> dict[str, Tensor]:
        """Collate single-example Qwen inputs without changing image-token layout."""

        if not inputs:
            raise ValueError("inputs cannot be empty")
        keys = set(inputs[0])
        if any(set(item) != keys for item in inputs):
            raise ValueError("all processor inputs must have the same keys")
        result: dict[str, Tensor] = {}
        for key in keys:
            values = [item[key] for item in inputs]
            if not all(isinstance(value, Tensor) for value in values):
                raise ValueError(f"processor field {key} must be a tensor")
            tensors = [value for value in values if isinstance(value, Tensor)]
            if key in {"pixel_values", "image_grid_thw"}:
                result[key] = torch.cat(tensors, dim=0)
                continue
            if tensors[0].ndim == 2:
                width = max(tensor.shape[1] for tensor in tensors)
                padding = {
                    "input_ids": pad_token_id,
                    "attention_mask": 0,
                    "mm_token_type_ids": 0,
                }.get(key, 0)
                padded = tensors[0].new_full((len(tensors), width), padding)
                for row, tensor in enumerate(tensors):
                    padded[row, : tensor.shape[1]] = tensor[0]
                result[key] = padded
            elif tensors[0].ndim == 1:
                result[key] = torch.stack([tensor.squeeze(0) for tensor in tensors])
            else:
                raise ValueError(f"unsupported processor field shape for {key}: {tensors[0].shape}")
        return result

    def prepare_batch_inputs(self, examples: Sequence[tuple[Any, Any, str]]) -> Any:
        """Build a batch by collating the exact single-example processor path."""

        if self.processor is None:
            raise QwenBackboneError("call load() before prepare_batch_inputs()")
        if not examples:
            raise ValueError("examples cannot be empty")
        single = [self.prepare_inputs(image, wrist, instruction) for image, wrist, instruction in examples]
        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        return self.collate_processor_inputs([dict(item) for item in single], pad_token_id=pad_token_id)

    def prepare_view_batch_inputs(self, examples: Sequence[tuple[Any, str]]) -> Any:
        """Collate a batch for one named camera while preserving instruction context."""

        if self.processor is None:
            raise QwenBackboneError("call load() before prepare_view_batch_inputs()")
        if not examples:
            raise ValueError("examples cannot be empty")
        single = [self.prepare_view_inputs(image, instruction) for image, instruction in examples]
        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        return self.collate_processor_inputs([dict(item) for item in single], pad_token_id=pad_token_id)

    def _hidden_states(
        self,
        inputs: Any,
        *,
        hidden_layer: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run Qwen and return a selected hidden layer plus its attention mask."""

        hidden_states, attention_mask, _input_ids = self._hidden_state_stack(inputs)
        layer = self.config.hidden_layer if hidden_layer is None else hidden_layer
        if layer < -1:
            raise ValueError("hidden_layer must be -1 or a non-negative hidden-state index")
        if layer >= len(hidden_states):
            raise ValueError(
                f"hidden_layer {layer} is outside {len(hidden_states)} returned states"
            )
        return hidden_states[layer], attention_mask

    def _hidden_state_stack(
        self,
        inputs: Any,
    ) -> tuple[tuple[Tensor, ...], Tensor, Tensor]:
        """Run Qwen once and return every hidden state, mask, and token ids."""

        if self.model is None:
            raise QwenBackboneError("call load() before forward()")
        if "attention_mask" not in inputs or "input_ids" not in inputs:
            raise ValueError("processor inputs must contain attention_mask and input_ids")
        device = next(self.model.parameters()).device
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            inputs = {
                key: value.to(device) if isinstance(value, Tensor) else value
                for key, value in inputs.items()
            }
        context = torch.inference_mode() if self.config.freeze else torch.enable_grad()
        with context:
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        if outputs.hidden_states is None:
            raise QwenBackboneError("Qwen output did not contain hidden_states")
        return tuple(outputs.hidden_states), inputs["attention_mask"], inputs["input_ids"]

    @staticmethod
    def instruction_token_mask(
        input_ids: Tensor,
        instruction_token_candidates: Sequence[Sequence[int]],
    ) -> Tensor:
        """Locate the last exact instruction-token occurrence in each sequence.

        The last occurrence is intentional: chat templates can repeat text in a
        system prefix, while the user instruction is the final matching span.
        """

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        if len(instruction_token_candidates) != input_ids.shape[0]:
            raise ValueError("one instruction token sequence is required per batch row")
        result = torch.zeros_like(input_ids, dtype=torch.bool)
        for row_index, candidate in enumerate(instruction_token_candidates):
            tokens = [int(token) for token in candidate]
            if not tokens:
                raise ValueError("instruction token sequence cannot be empty")
            row = input_ids[row_index].tolist()
            starts = [
                start
                for start in range(len(row) - len(tokens) + 1)
                if row[start : start + len(tokens)] == tokens
            ]
            if not starts:
                raise ValueError(f"instruction tokens were not found in batch row {row_index}")
            start = starts[-1]
            result[row_index, start : start + len(tokens)] = True
        return result

    @staticmethod
    def tail_token_mask(attention_mask: Tensor, num_tokens: int) -> Tensor:
        """Select the final valid prompt tokens, which summarize prior multimodal context."""

        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, tokens]")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        result = torch.zeros_like(attention_mask, dtype=torch.bool)
        for row_index, row in enumerate(attention_mask.to(dtype=torch.bool)):
            valid = torch.nonzero(row, as_tuple=False).flatten()
            if valid.numel() < num_tokens:
                raise ValueError("prompt has fewer valid tokens than requested tail tokens")
            result[row_index, valid[-num_tokens:]] = True
        return result

    def _tokenize_instruction_candidates(self, instructions: Sequence[str]) -> list[list[int]]:
        if self.processor is None:
            raise QwenBackboneError("call load() before encoding instructions")
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise QwenBackboneError("Qwen processor has no tokenizer")
        candidates: list[list[int]] = []
        for instruction in instructions:
            if not instruction.strip():
                raise ValueError("instruction cannot be empty")
            encoded = tokenizer(instruction, add_special_tokens=False, return_tensors="pt")
            token_ids = encoded["input_ids"][0].tolist()
            candidates.append([int(token) for token in token_ids])
        return candidates

    def forward(self, inputs: Any) -> Tensor:
        """Run frozen Qwen and return one pooled feature per sample."""

        hidden_states, attention_mask = self._hidden_states(inputs)
        return self.pool_hidden_states(hidden_states, attention_mask)

    def forward_context(
        self,
        inputs: Any,
        *,
        num_tokens: int = 8,
        hidden_layer: int | None = None,
    ) -> Tensor:
        """Run frozen Qwen and return an ordered fixed-size token context."""

        hidden_states, attention_mask = self._hidden_states(inputs, hidden_layer=hidden_layer)
        return self.reduce_hidden_states(hidden_states, attention_mask, num_tokens)

    def forward_multiview_context(
        self,
        camera_inputs: Sequence[Any],
        *,
        num_tokens_per_camera: int,
        hidden_layer: int | None = None,
    ) -> Tensor:
        """Return explicit ``[B, cameras, tokens, D]`` frozen-Qwen context."""

        if len(camera_inputs) < 1:
            raise ValueError("camera_inputs cannot be empty")
        contexts = [
            self.forward_context(
                inputs,
                num_tokens=num_tokens_per_camera,
                hidden_layer=hidden_layer,
            )
            for inputs in camera_inputs
        ]
        batch_shapes = {tuple(context.shape) for context in contexts}
        if len(batch_shapes) != 1:
            raise ValueError("all camera contexts must have the same shape")
        return torch.stack(contexts, dim=1)

    def forward_multilayer_context(
        self,
        inputs: Any,
        *,
        num_tokens: int,
        hidden_layers: Sequence[int],
    ) -> Tensor:
        """Return aligned ordered contexts from several layers in one Qwen pass.

        The result is ``[B, layers, tokens, D]``.  Every layer uses the same
        attention mask and the same deterministic token bins, which is needed
        for a controlled layer-fusion comparison.
        """

        if not hidden_layers:
            raise ValueError("hidden_layers cannot be empty")
        if len(set(hidden_layers)) != len(hidden_layers):
            raise ValueError("hidden_layers must be unique")
        hidden_states, attention_mask, _input_ids = self._hidden_state_stack(inputs)
        contexts: list[Tensor] = []
        for layer in hidden_layers:
            if layer < -1 or layer >= len(hidden_states):
                raise ValueError(f"hidden layer {layer} is outside returned hidden states")
            contexts.append(
                self.reduce_hidden_states(hidden_states[layer], attention_mask, num_tokens)
            )
        return torch.stack(contexts, dim=1)

    def forward_multiview_multilayer_context(
        self,
        camera_inputs: Sequence[Any],
        *,
        num_tokens_per_camera: int,
        hidden_layers: Sequence[int],
    ) -> Tensor:
        """Return ``[B, layers, cameras, tokens, D]`` frozen-Qwen contexts."""

        if not camera_inputs:
            raise ValueError("camera_inputs cannot be empty")
        contexts = [
            self.forward_multilayer_context(
                inputs,
                num_tokens=num_tokens_per_camera,
                hidden_layers=hidden_layers,
            )
            for inputs in camera_inputs
        ]
        shapes = {tuple(context.shape) for context in contexts}
        if len(shapes) != 1:
            raise ValueError("all camera multilayer contexts must have the same shape")
        return torch.stack(contexts, dim=2)

    def forward_multilayer_spatial_semantic_context(
        self,
        inputs: Any,
        instructions: Sequence[str],
        *,
        hidden_layers: Sequence[int],
        expected_grid: tuple[int, int] = (8, 8),
        output_grid: tuple[int, int] = (4, 4),
        _hidden_stack: tuple | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return aligned spatial and instruction features from one Qwen pass.

        Spatial output is ``[B, layers, H_out*W_out, D]`` and semantic output
        is ``[B, layers, 1, D]``.  Image tokens are selected exactly with
        Qwen's image token id and then average-pooled on the raster grid.  The
        semantic token is the final valid prompt token at each layer, after
        the complete image and instruction span in the causal sequence.
        """

        if not hidden_layers or len(set(hidden_layers)) != len(hidden_layers):
            raise ValueError("hidden_layers must be non-empty and unique")
        expected_height, expected_width = expected_grid
        output_height, output_width = output_grid
        if (
            expected_height <= 0
            or expected_width <= 0
            or output_height <= 0
            or output_width <= 0
            or expected_height % output_height
            or expected_width % output_width
        ):
            raise ValueError("output_grid must divide expected_grid exactly")

        hidden_states, attention_mask, input_ids = self._hidden_state_stack(inputs) if _hidden_stack is None else _hidden_stack
        if len(instructions) != input_ids.shape[0] or any(
            not instruction.strip() for instruction in instructions
        ):
            raise ValueError("one non-empty instruction is required per batch row")
        for layer in hidden_layers:
            if layer < -1 or layer >= len(hidden_states):
                raise ValueError(f"hidden layer {layer} is outside returned hidden states")
        tokenizer = getattr(self.processor, "tokenizer", None)
        image_processor = getattr(self.processor, "image_processor", None)
        if tokenizer is None or image_processor is None:
            raise QwenBackboneError("Qwen processor lacks tokenizer or image processor")
        image_token_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        valid_mask = attention_mask.to(dtype=torch.bool)
        image_mask = input_ids.eq(image_token_id) & valid_mask
        semantic_mask = self.tail_token_mask(valid_mask, 1)

        grid_thw = inputs.get("image_grid_thw")
        if not isinstance(grid_thw, Tensor) or grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError("image_grid_thw must have shape [batch, 3]")
        if grid_thw.shape[0] != input_ids.shape[0]:
            raise ValueError("multilayer spatial context requires one image per sample")
        merge = int(getattr(image_processor, "merge_size", 0))
        if merge <= 0:
            raise ValueError("image processor merge_size must be positive")

        batch_spatial: list[Tensor] = []
        batch_semantic: list[Tensor] = []
        pool_height = expected_height // output_height
        pool_width = expected_width // output_width
        for row, grid in enumerate(grid_thw):
            temporal, height, width = (int(value) for value in grid.tolist())
            if temporal != 1 or height % merge or width % merge:
                raise ValueError("only one-frame image grids divisible by merge_size are supported")
            post_grid = (height // merge, width // merge)
            if post_grid != expected_grid:
                raise ValueError(
                    f"post-merge image grid {post_grid} does not match expected {expected_grid}"
                )
            layer_spatial: list[Tensor] = []
            layer_semantic: list[Tensor] = []
            for layer in hidden_layers:
                tokens = hidden_states[layer][row]
                image_tokens = tokens[image_mask[row]]
                if image_tokens.shape[0] != expected_height * expected_width:
                    raise ValueError("image token count does not match the declared post-merge grid")
                raster = image_tokens.reshape(expected_height, expected_width, -1)
                pooled = raster.reshape(
                    output_height,
                    pool_height,
                    output_width,
                    pool_width,
                    raster.shape[-1],
                ).mean(dim=(1, 3))
                layer_spatial.append(pooled.reshape(output_height * output_width, -1))
                semantic_tokens = tokens[semantic_mask[row]]
                if semantic_tokens.shape[0] == 0:
                    raise ValueError("instruction span cannot be empty")
                layer_semantic.append(semantic_tokens.mean(dim=0, keepdim=True))
            batch_spatial.append(torch.stack(layer_spatial, dim=0))
            batch_semantic.append(torch.stack(layer_semantic, dim=0))
        return torch.stack(batch_spatial, dim=0), torch.stack(batch_semantic, dim=0)

    def forward_v329_context(self, camera_inputs: Sequence[Any], instructions: Sequence[str]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """One forward per camera: old spatial/summary fields plus exact instruction tokens."""
        visuals, semantics, languages, masks = [], [], [], []
        for inputs in camera_inputs:
            stack = self._hidden_state_stack(inputs)
            hidden, valid, ids = stack
            spatial, semantic = self.forward_multilayer_spatial_semantic_context(
                inputs, instructions, hidden_layers=(12, 14, 18, -1),
                expected_grid=(8, 8), output_grid=(8, 8), _hidden_stack=stack,
            )
            candidates = [self.processor.tokenizer.encode(text, add_special_tokens=False) for text in instructions]
            span = self.instruction_token_mask(ids, candidates) & valid.bool()
            if any(not 0 < len(tokens) <= 32 for tokens in candidates):
                raise ValueError('instruction must contain 1..32 exact tokens; no truncation')
            language = hidden[-1].new_zeros((len(instructions), 32, self.output_dim))
            mask = torch.zeros((len(instructions), 32), dtype=torch.bool, device=language.device)
            for row, tokens in enumerate(candidates):
                if int(span[row].sum()) != len(tokens):
                    raise ValueError('instruction span intersects padding')
                language[row, :len(tokens)] = hidden[-1][row, span[row]]
                mask[row, :len(tokens)] = True
            visuals.append(spatial); semantics.append(semantic)
            languages.append(language); masks.append(mask)
        return (torch.stack(visuals, 2), torch.stack(semantics, 2).mean(2),
                torch.stack(languages, 1), torch.stack(masks, 1))

    def forward_v331_context(
        self,
        camera_inputs: Sequence[Any],
        instructions: Sequence[str],
        *,
        max_tokens: int = 256,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """V3.29-compatible features with exact, non-truncated variable language."""
        if max_tokens < 32:
            raise ValueError("V3.31 max_tokens must preserve the 32-token parent range")
        token_ids = [
            self.processor.tokenizer.encode(text, add_special_tokens=False)
            for text in instructions
        ]
        longest = max((len(tokens) for tokens in token_ids), default=0)
        if not 0 < longest <= max_tokens:
            raise ValueError(f"instruction must contain 1..{max_tokens} exact tokens")
        visuals, semantics, languages, masks = [], [], [], []
        for inputs in camera_inputs:
            stack = self._hidden_state_stack(inputs)
            hidden, valid, ids = stack
            spatial, semantic = self.forward_multilayer_spatial_semantic_context(
                inputs,
                instructions,
                hidden_layers=(12, 14, 18, -1),
                expected_grid=(8, 8),
                output_grid=(8, 8),
                _hidden_stack=stack,
            )
            span = self.instruction_token_mask(ids, token_ids) & valid.bool()
            language = hidden[-1].new_zeros(
                (len(instructions), longest, self.output_dim)
            )
            mask = torch.zeros(
                (len(instructions), longest), dtype=torch.bool, device=language.device
            )
            for row, tokens in enumerate(token_ids):
                if int(span[row].sum()) != len(tokens):
                    raise ValueError("instruction span intersects padding")
                language[row, : len(tokens)] = hidden[-1][row, span[row]]
                mask[row, : len(tokens)] = True
            visuals.append(spatial)
            semantics.append(semantic)
            languages.append(language)
            masks.append(mask)
        return (
            torch.stack(visuals, 2),
            torch.stack(semantics, 2).mean(2),
            torch.stack(languages, 1),
            torch.stack(masks, 1),
        )

    def forward_multiview_multilayer_spatial_semantic_context(
        self,
        camera_inputs: Sequence[Any],
        instructions: Sequence[str],
        *,
        hidden_layers: Sequence[int],
        expected_grid: tuple[int, int] = (8, 8),
        output_grid: tuple[int, int] = (4, 4),
    ) -> tuple[Tensor, Tensor]:
        """Return V2.1 visual ``[B,L,C,T,D]`` and semantic ``[B,L,1,D]``.

        Each camera is encoded independently to preserve the explicit camera
        axis.  Camera-conditioned instruction summaries are averaged because
        the V2.1 cache stores one task-semantic token per layer.
        """

        if not camera_inputs:
            raise ValueError("camera_inputs cannot be empty")
        outputs = [
            self.forward_multilayer_spatial_semantic_context(
                inputs,
                instructions,
                hidden_layers=hidden_layers,
                expected_grid=expected_grid,
                output_grid=output_grid,
            )
            for inputs in camera_inputs
        ]
        visual_shapes = {tuple(visual.shape) for visual, _semantic in outputs}
        semantic_shapes = {tuple(semantic.shape) for _visual, semantic in outputs}
        if len(visual_shapes) != 1 or len(semantic_shapes) != 1:
            raise ValueError("all camera multilayer contexts must have the same shape")
        visual = torch.stack([item[0] for item in outputs], dim=2)
        semantic = torch.stack([item[1] for item in outputs], dim=2).mean(dim=2)
        return visual, semantic

    def forward_spatial_grid_context(
        self,
        inputs: Any,
        *,
        hidden_layer: int = 14,
        expected_grid: tuple[int, int] = (8, 8),
    ) -> Tensor:
        """Return exact raster-ordered image tokens as ``[B, H*W, D]``.

        Unlike :meth:`forward_context`, this path does not bin the multimodal
        sequence.  Qwen's ``image_grid_thw`` and ``spatial_merge_size`` define
        the post-merge patch grid, while the ``<|image_pad|>`` positions select
        the corresponding language-model hidden states exactly.
        """

        hidden_states, attention_mask, input_ids = self._hidden_state_stack(inputs)
        if hidden_layer < -1 or hidden_layer >= len(hidden_states):
            raise ValueError(f"hidden layer {hidden_layer} is outside returned hidden states")
        tokenizer = getattr(self.processor, "tokenizer", None)
        image_processor = getattr(self.processor, "image_processor", None)
        if tokenizer is None or image_processor is None:
            raise QwenBackboneError("Qwen processor lacks tokenizer or image processor")
        image_token_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        image_mask = input_ids.eq(image_token_id) & attention_mask.to(dtype=torch.bool)
        grid_thw = inputs.get("image_grid_thw")
        if not isinstance(grid_thw, Tensor) or grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError("image_grid_thw must have shape [batch, 3]")
        if grid_thw.shape[0] != input_ids.shape[0]:
            raise ValueError("spatial-grid context requires exactly one image per sample")
        merge = int(getattr(image_processor, "merge_size", 0))
        if merge <= 0:
            raise ValueError("image processor merge_size must be positive")
        selected: list[Tensor] = []
        expected_height, expected_width = expected_grid
        for row, (tokens, grid) in enumerate(zip(hidden_states[hidden_layer], grid_thw)):
            temporal, height, width = (int(value) for value in grid.tolist())
            if temporal != 1 or height % merge or width % merge:
                raise ValueError("only one-frame image grids divisible by merge_size are supported")
            post_grid = (height // merge, width // merge)
            if post_grid != expected_grid:
                raise ValueError(
                    f"post-merge image grid {post_grid} does not match expected {expected_grid}"
                )
            image_tokens = tokens[image_mask[row]]
            if image_tokens.shape[0] != expected_height * expected_width:
                raise ValueError("image token count does not match the declared post-merge grid")
            selected.append(image_tokens)
        return torch.stack(selected, dim=0)

    def forward_multiview_spatial_grid_context(
        self,
        camera_inputs: Sequence[Any],
        *,
        hidden_layer: int = 14,
        expected_grid: tuple[int, int] = (8, 8),
    ) -> Tensor:
        """Return camera-separated exact Qwen image grids."""

        if len(camera_inputs) < 1:
            raise ValueError("camera_inputs cannot be empty")
        contexts = [
            self.forward_spatial_grid_context(
                inputs, hidden_layer=hidden_layer, expected_grid=expected_grid
            )
            for inputs in camera_inputs
        ]
        shapes = {tuple(context.shape) for context in contexts}
        if len(shapes) != 1:
            raise ValueError("all camera spatial grids must have the same shape")
        return torch.stack(contexts, dim=1)

    def forward_hybrid_context(
        self,
        inputs: Any,
        instructions: Sequence[str],
        *,
        num_tokens: int,
        semantic_tokens: int,
        spatial_hidden_layer: int = 14,
        semantic_hidden_layer: int = -1,
    ) -> Tensor:
        """Fuse spatial mid-layer tokens with explicit final-layer language tokens."""

        if semantic_tokens <= 0 or semantic_tokens >= num_tokens:
            raise ValueError("semantic_tokens must be between zero and num_tokens")
        hidden_states, attention_mask, input_ids = self._hidden_state_stack(inputs)
        for layer in (spatial_hidden_layer, semantic_hidden_layer):
            if layer < -1 or layer >= len(hidden_states):
                raise ValueError(f"hidden layer {layer} is outside returned hidden states")
        candidates = self._tokenize_instruction_candidates(instructions)
        instruction_mask = self.instruction_token_mask(input_ids, candidates)
        instruction_mask &= attention_mask.to(dtype=torch.bool)
        spatial_mask = attention_mask.to(dtype=torch.bool) & ~instruction_mask
        spatial = self.reduce_hidden_states(
            hidden_states[spatial_hidden_layer],
            spatial_mask,
            num_tokens - semantic_tokens,
        )
        semantic = self.reduce_hidden_states(
            hidden_states[semantic_hidden_layer],
            instruction_mask,
            semantic_tokens,
        )
        return torch.cat((spatial, semantic), dim=1)

    def forward_multiview_hybrid_context(
        self,
        camera_inputs: Sequence[Any],
        instructions: Sequence[str],
        *,
        num_tokens_per_camera: int,
        semantic_tokens_per_camera: int,
        spatial_hidden_layer: int = 14,
        semantic_hidden_layer: int = -1,
    ) -> Tensor:
        """Return camera-separated hybrid spatial/language Qwen contexts."""

        if len(camera_inputs) < 1:
            raise ValueError("camera_inputs cannot be empty")
        contexts = [
            self.forward_hybrid_context(
                inputs,
                instructions,
                num_tokens=num_tokens_per_camera,
                semantic_tokens=semantic_tokens_per_camera,
                spatial_hidden_layer=spatial_hidden_layer,
                semantic_hidden_layer=semantic_hidden_layer,
            )
            for inputs in camera_inputs
        ]
        batch_shapes = {tuple(context.shape) for context in contexts}
        if len(batch_shapes) != 1:
            raise ValueError("all camera hybrid contexts must have the same shape")
        return torch.stack(contexts, dim=1)

    def forward_grounded_context(
        self,
        inputs: Any,
        instructions: Sequence[str],
        *,
        num_tokens: int,
        semantic_tokens: int,
        grounding_tokens: int,
        spatial_hidden_layer: int = 14,
        semantic_hidden_layer: int = -1,
    ) -> Tensor:
        """Keep pure image tokens plus explicit instruction and global-goal tokens."""

        spatial_tokens = num_tokens - semantic_tokens - grounding_tokens
        if min(spatial_tokens, semantic_tokens, grounding_tokens) <= 0:
            raise ValueError("grounded context requires positive spatial, semantic, and goal tokens")
        hidden_states, attention_mask, input_ids = self._hidden_state_stack(inputs)
        for layer in (spatial_hidden_layer, semantic_hidden_layer):
            if layer < -1 or layer >= len(hidden_states):
                raise ValueError(f"hidden layer {layer} is outside returned hidden states")
        candidates = self._tokenize_instruction_candidates(instructions)
        instruction_mask = self.instruction_token_mask(input_ids, candidates)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise QwenBackboneError("Qwen processor has no tokenizer")
        image_token_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        image_mask = input_ids.eq(image_token_id) & attention_mask.to(dtype=torch.bool)
        if not image_mask.any(dim=1).all():
            raise ValueError("each grounded context sample must contain image tokens")
        goal_mask = self.tail_token_mask(attention_mask, grounding_tokens)
        spatial = self.reduce_hidden_states(
            hidden_states[spatial_hidden_layer], image_mask, spatial_tokens
        )
        semantic = self.reduce_hidden_states(
            hidden_states[semantic_hidden_layer], instruction_mask, semantic_tokens
        )
        goal = self.reduce_hidden_states(
            hidden_states[semantic_hidden_layer], goal_mask, grounding_tokens
        )
        return torch.cat((spatial, semantic, goal), dim=1)

    def forward_multiview_grounded_context(
        self,
        camera_inputs: Sequence[Any],
        instructions: Sequence[str],
        *,
        num_tokens_per_camera: int,
        semantic_tokens_per_camera: int,
        grounding_tokens_per_camera: int,
        spatial_hidden_layer: int = 14,
        semantic_hidden_layer: int = -1,
    ) -> Tensor:
        """Return camera-separated grounded spatial/language Qwen contexts."""

        if len(camera_inputs) < 1:
            raise ValueError("camera_inputs cannot be empty")
        contexts = [
            self.forward_grounded_context(
                inputs,
                instructions,
                num_tokens=num_tokens_per_camera,
                semantic_tokens=semantic_tokens_per_camera,
                grounding_tokens=grounding_tokens_per_camera,
                spatial_hidden_layer=spatial_hidden_layer,
                semantic_hidden_layer=semantic_hidden_layer,
            )
            for inputs in camera_inputs
        ]
        batch_shapes = {tuple(context.shape) for context in contexts}
        if len(batch_shapes) != 1:
            raise ValueError("all camera grounded contexts must have the same shape")
        return torch.stack(contexts, dim=1)
