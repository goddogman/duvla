from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from duvla.models.qwen_backbone import QwenBackboneConfig, QwenVLBackbone


def test_qwen_backbone_masked_pooling() -> None:
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

    pooled = QwenVLBackbone.pool_hidden_states(hidden, mask)

    assert pooled.shape == (2, 3)
    assert torch.equal(pooled[0], torch.tensor([1.5, 2.5, 3.5]))
    assert torch.equal(pooled[1], hidden[1, 0])


def test_qwen_backbone_reduces_ordered_tokens_to_fixed_context() -> None:
    hidden = torch.arange(1 * 6 * 2, dtype=torch.float32).reshape(1, 6, 2)
    mask = torch.tensor([[True, True, True, True, False, False]])

    reduced = QwenVLBackbone.reduce_hidden_states(hidden, mask, num_tokens=2)

    assert reduced.shape == (1, 2, 2)
    torch.testing.assert_close(reduced[0], torch.tensor([[1.0, 2.0], [5.0, 6.0]]))


def test_qwen_backbone_reduction_rejects_empty_sample() -> None:
    hidden = torch.zeros(1, 3, 2)
    mask = torch.zeros(1, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one valid token"):
        QwenVLBackbone.reduce_hidden_states(hidden, mask, num_tokens=2)


def test_qwen_backbone_rejects_missing_local_model(tmp_path: Path) -> None:
    backbone = QwenVLBackbone(QwenBackboneConfig(model_path=tmp_path / "missing"))

    with pytest.raises(RuntimeError, match="does not exist"):
        backbone.load()


def test_qwen_backbone_collates_padded_text_and_image_fields() -> None:
    first = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[0, 1, 1]]),
        "pixel_values": torch.ones((4, 2)),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    second = {
        "input_ids": torch.tensor([[4, 5]]),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[0, 1]]),
        "pixel_values": torch.zeros((4, 2)),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }

    result = QwenVLBackbone.collate_processor_inputs([first, second], pad_token_id=99)

    assert result["input_ids"].tolist() == [[1, 2, 3], [4, 5, 99]]
    assert result["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert result["pixel_values"].shape == (8, 2)
    assert result["image_grid_thw"].shape == (2, 3)


def test_qwen_backbone_selects_explicit_intermediate_hidden_state() -> None:
    class FakeQwen(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                hidden_states=(
                    torch.zeros(1, 3, 4),
                    torch.ones(1, 3, 4),
                    torch.full((1, 3, 4), 2.0),
                )
            )

    backbone = QwenVLBackbone()
    backbone.model = FakeQwen()
    inputs = {"input_ids": torch.ones(1, 3, dtype=torch.long), "attention_mask": torch.ones(1, 3)}

    selected, mask = backbone._hidden_states(inputs, hidden_layer=1)

    assert torch.equal(selected, torch.ones(1, 3, 4))
    assert torch.equal(mask, inputs["attention_mask"])


def test_qwen_backbone_stacks_camera_context_on_explicit_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    backbone = QwenVLBackbone()

    def fake_forward(inputs: object, *, num_tokens: int, hidden_layer: int | None) -> torch.Tensor:
        value = float(inputs)
        return torch.full((2, num_tokens, 4), value + float(hidden_layer or 0))

    monkeypatch.setattr(backbone, "forward_context", fake_forward)
    context = backbone.forward_multiview_context(
        (1, 2),
        num_tokens_per_camera=3,
        hidden_layer=4,
    )

    assert context.shape == (2, 2, 3, 4)
    assert torch.equal(context[:, 0], torch.full((2, 3, 4), 5.0))
    assert torch.equal(context[:, 1], torch.full((2, 3, 4), 6.0))


def test_multilayer_context_uses_one_qwen_pass_and_aligned_bins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone = QwenVLBackbone()
    attention = torch.tensor([[1, 1, 1, 1, 0, 0]])
    states = tuple(
        torch.full((1, 6, 2), float(index)) for index in range(4)
    )
    calls = 0

    def fake_stack(_inputs: object) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        return states, attention, torch.ones(1, 6, dtype=torch.long)

    monkeypatch.setattr(backbone, "_hidden_state_stack", fake_stack)
    result = backbone.forward_multilayer_context(
        {}, num_tokens=2, hidden_layers=(1, 2, -1)
    )

    assert calls == 1
    assert result.shape == (1, 3, 2, 2)
    assert torch.equal(result[:, 0], torch.ones(1, 2, 2))
    assert torch.equal(result[:, 1], torch.full((1, 2, 2), 2.0))
    assert torch.equal(result[:, 2], torch.full((1, 2, 2), 3.0))


def test_multiview_multilayer_context_keeps_explicit_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone = QwenVLBackbone()

    def fake_multilayer(
        inputs: object, *, num_tokens: int, hidden_layers: tuple[int, ...]
    ) -> torch.Tensor:
        assert hidden_layers == (12, 14, 18, -1)
        return torch.full((2, 4, num_tokens, 3), float(inputs))

    monkeypatch.setattr(backbone, "forward_multilayer_context", fake_multilayer)
    result = backbone.forward_multiview_multilayer_context(
        (1, 2), num_tokens_per_camera=5, hidden_layers=(12, 14, 18, -1)
    )

    assert result.shape == (2, 4, 2, 5, 3)
    assert torch.equal(result[:, :, 0], torch.ones(2, 4, 5, 3))
    assert torch.equal(result[:, :, 1], torch.full((2, 4, 5, 3), 2.0))


def test_instruction_token_mask_selects_last_exact_occurrence() -> None:
    input_ids = torch.tensor([[9, 4, 5, 8, 4, 5, 7], [1, 2, 3, 0, 0, 0, 0]])

    mask = QwenVLBackbone.instruction_token_mask(input_ids, ([4, 5], [2, 3]))

    assert mask.tolist() == [
        [False, False, False, False, True, True, False],
        [False, True, True, False, False, False, False],
    ]


def test_hybrid_context_keeps_shape_and_uses_distinct_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    backbone = QwenVLBackbone()
    attention = torch.ones(1, 8, dtype=torch.long)
    input_ids = torch.tensor([[1, 2, 3, 4, 10, 11, 6, 7]])
    spatial = torch.arange(8, dtype=torch.float32)[None, :, None].expand(1, 8, 2)
    semantic = (100 + torch.arange(8, dtype=torch.float32))[None, :, None].expand(1, 8, 2)
    monkeypatch.setattr(
        backbone,
        "_hidden_state_stack",
        lambda _inputs: ((torch.zeros_like(spatial), spatial, semantic), attention, input_ids),
    )
    monkeypatch.setattr(backbone, "_tokenize_instruction_candidates", lambda _items: [[10, 11]])

    result = backbone.forward_hybrid_context(
        {}, ["pick object"], num_tokens=4, semantic_tokens=2,
        spatial_hidden_layer=1, semantic_hidden_layer=2,
    )

    assert result.shape == (1, 4, 2)
    assert torch.equal(result[0, -2:], torch.tensor([[104.0, 104.0], [105.0, 105.0]]))


def test_tail_token_mask_selects_final_valid_prompt_tokens() -> None:
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])

    mask = QwenVLBackbone.tail_token_mask(attention, 2)

    assert mask.tolist() == [
        [False, False, True, True, False],
        [False, True, True, False, False],
    ]


def test_grounded_context_uses_only_image_instruction_and_tail_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        def convert_tokens_to_ids(self, token: str) -> int:
            assert token == "<|image_pad|>"
            return 99

    backbone = QwenVLBackbone()
    backbone.processor = SimpleNamespace(tokenizer=FakeTokenizer())
    input_ids = torch.tensor([[1, 99, 99, 99, 10, 11, 7, 8]])
    attention = torch.ones_like(input_ids)
    mid = torch.arange(8, dtype=torch.float32)[None, :, None].expand(1, 8, 2)
    final = (100 + torch.arange(8, dtype=torch.float32))[None, :, None].expand(1, 8, 2)
    monkeypatch.setattr(
        backbone,
        "_hidden_state_stack",
        lambda _inputs: ((torch.zeros_like(mid), mid, final), attention, input_ids),
    )
    monkeypatch.setattr(backbone, "_tokenize_instruction_candidates", lambda _items: [[10, 11]])

    result = backbone.forward_grounded_context(
        {}, ["pick object"], num_tokens=6, semantic_tokens=2, grounding_tokens=2,
        spatial_hidden_layer=1, semantic_hidden_layer=2,
    )

    assert result.shape == (1, 6, 2)
    assert torch.equal(result[0, :2], torch.tensor([[1.5, 1.5], [3.0, 3.0]]))
    assert torch.equal(result[0, 2:4], torch.tensor([[104.0, 104.0], [105.0, 105.0]]))
    assert torch.equal(result[0, 4:], torch.tensor([[106.0, 106.0], [107.0, 107.0]]))


def test_spatial_grid_context_preserves_exact_raster_image_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        def convert_tokens_to_ids(self, token: str) -> int:
            assert token == "<|image_pad|>"
            return 99

    backbone = QwenVLBackbone()
    backbone.processor = SimpleNamespace(
        tokenizer=FakeTokenizer(), image_processor=SimpleNamespace(merge_size=2)
    )
    input_ids = torch.tensor([[1, 99, 99, 99, 99, 7]])
    attention = torch.ones_like(input_ids)
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    monkeypatch.setattr(
        backbone,
        "_hidden_state_stack",
        lambda _inputs: ((torch.zeros_like(hidden), hidden), attention, input_ids),
    )
    result = backbone.forward_spatial_grid_context(
        {"image_grid_thw": torch.tensor([[1, 4, 4]])},
        hidden_layer=1,
        expected_grid=(2, 2),
    )
    assert result.shape == (1, 4, 2)
    assert torch.equal(result[0], hidden[0, 1:5])


def test_spatial_grid_context_rejects_mismatched_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone = QwenVLBackbone()
    backbone.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _token: 99),
        image_processor=SimpleNamespace(merge_size=2),
    )
    hidden = torch.zeros(1, 6, 2)
    monkeypatch.setattr(
        backbone,
        "_hidden_state_stack",
        lambda _inputs: (
            (hidden,),
            torch.ones(1, 6, dtype=torch.long),
            torch.tensor([[1, 99, 99, 99, 99, 7]]),
        ),
    )
    with pytest.raises(ValueError, match="does not match expected"):
        backbone.forward_spatial_grid_context(
            {"image_grid_thw": torch.tensor([[1, 4, 4]])},
            hidden_layer=0,
            expected_grid=(1, 4),
        )


def test_multilayer_spatial_semantic_context_preserves_axes_and_pools_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        def convert_tokens_to_ids(self, token: str) -> int:
            assert token == "<|image_pad|>"
            return 99

    backbone = QwenVLBackbone()
    backbone.processor = SimpleNamespace(
        tokenizer=FakeTokenizer(), image_processor=SimpleNamespace(merge_size=2)
    )
    input_ids = torch.tensor([[1, 99, 99, 99, 99, 10, 11]])
    attention = torch.ones_like(input_ids)
    layer_one = torch.arange(14, dtype=torch.float32).reshape(1, 7, 2)
    layer_two = layer_one + 100.0
    monkeypatch.setattr(
        backbone,
        "_hidden_state_stack",
        lambda _inputs: ((torch.zeros_like(layer_one), layer_one, layer_two), attention, input_ids),
    )
    visual, semantic = backbone.forward_multilayer_spatial_semantic_context(
        {"image_grid_thw": torch.tensor([[1, 4, 4]])},
        ["pick object"],
        hidden_layers=(1, -1),
        expected_grid=(2, 2),
        output_grid=(1, 1),
    )

    assert visual.shape == (1, 2, 1, 2)
    assert semantic.shape == (1, 2, 1, 2)
    torch.testing.assert_close(visual[0, 0, 0], layer_one[0, 1:5].mean(dim=0))
    torch.testing.assert_close(semantic[0, 0, 0], layer_one[0, -1])
    torch.testing.assert_close(semantic[0, 1, 0], layer_two[0, -1])


def test_multiview_multilayer_spatial_semantic_context_averages_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone = QwenVLBackbone()

    def fake_forward(
        inputs: object,
        _instructions: list[str],
        *,
        hidden_layers: tuple[int, ...],
        expected_grid: tuple[int, int],
        output_grid: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_layers == (12, 14, 18, -1)
        assert expected_grid == (8, 8)
        assert output_grid == (4, 4)
        value = float(inputs)
        return (
            torch.full((2, 4, 16, 3), value),
            torch.full((2, 4, 1, 3), value),
        )

    monkeypatch.setattr(backbone, "forward_multilayer_spatial_semantic_context", fake_forward)
    visual, semantic = backbone.forward_multiview_multilayer_spatial_semantic_context(
        (1, 3), ["a", "b"], hidden_layers=(12, 14, 18, -1)
    )

    assert visual.shape == (2, 4, 2, 16, 3)
    assert semantic.shape == (2, 4, 1, 3)
    assert torch.equal(visual[:, :, 0], torch.ones(2, 4, 16, 3))
    assert torch.equal(semantic, torch.full((2, 4, 1, 3), 2.0))
