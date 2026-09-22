"""CPU config regression: preserve native hybrid/MoE defaults under HF v5."""
import pytest
from sglang.srt.configs.qwen3_5 import Qwen3_5Config, Qwen3_5MoeConfig
from sglang.srt.utils.hf_transformers_utils import get_hf_text_config


@pytest.mark.parametrize('outer', [Qwen3_5Config, Qwen3_5MoeConfig])
def test_dict_subconfigs_restore_declared_types(outer):
    text_config = dict(hidden_size=3072, num_hidden_layers=48,
        num_attention_heads=32, num_key_value_heads=2,
        linear_num_key_heads=16, linear_num_value_heads=64, full_attention_interval=4,
        num_experts=256, num_experts_per_tok=8,
        layer_types=(['linear_attention'] * 3 + ['full_attention']) * 12,
        rope_parameters=dict(rope_type='default', rope_theta=10000000,
                             partial_rotary_factor=.25))
    cfg = outer(text_config=text_config,
                vision_config=dict(depth=27, hidden_size=1152, num_heads=16))
    text = get_hf_text_config(cfg)
    assert isinstance(text, outer.sub_configs['text_config'])
    assert isinstance(cfg.vision_config, outer.sub_configs['vision_config'])
    assert cfg.vision_config.depth == 27
    assert text.norm_topk_prob is True
    assert len(text.linear_layer_ids) == 36
    assert len(text.full_attention_layer_ids) == 12
    assert get_hf_text_config(cfg) is text


def test_unrelated_dict_config_keeps_original_generic_behavior():
    from transformers import PretrainedConfig
    cfg = PretrainedConfig()
    cfg.text_config = dict(hidden_size=32, num_attention_heads=4)
    text = get_hf_text_config(cfg)
    assert type(text) is PretrainedConfig
    assert text.hidden_size == 32
