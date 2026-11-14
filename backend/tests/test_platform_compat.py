import copy
import platform_compat as pc
import workflows


def _graph():
    return {
        '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'minimax_h3_ref2va_pruned_int8_convrot.safetensors', 'weight_dtype': 'default'}},
        '22': {'class_type': 'ModelAttentionBackend', 'inputs': {'model': ['1', 0], 'attention': 'comfy kitchen attention'}},
        '21': {'class_type': 'BFSMiniMaxH3NativeTemporal', 'inputs': {'model': ['22', 0], 'attention_backend': 'comfy kitchen (experimental)'}},
        '77': {'class_type': 'BlockSparseAttention', 'inputs': {'model': ['21', 0], 'selection': 'sol-attn'}},
        '10': {'class_type': 'CFGGuider', 'inputs': {'model': ['77', 0]}},
        '40': {'class_type': 'MinimaxH3LatentUpscaler3D', 'inputs': {'device': 'cuda'}},
    }


def test_cuda_graph_is_untouched():
    g = _graph()
    assert pc.adapt(copy.deepcopy(g), 'cuda') == g
    assert pc.adapt(copy.deepcopy(g), None) == g


def test_mps_swaps_cuda_only_pieces():
    g = pc.adapt(_graph(), 'mps')
    assert g['22']['inputs']['attention'] == 'pytorch attention'
    assert g['21']['inputs']['attention_backend'] == pc.TARA_FALLBACK
    assert '77' not in g and g['10']['inputs']['model'] == ['21', 0]
    assert g['40']['inputs']['device'] == 'mps'


def test_platform_override(monkeypatch):
    monkeypatch.setenv('HENGE_PLATFORM', 'mps')
    assert pc.device_type('http://127.0.0.1:1') == 'mps'


def test_quant_sibling_prefers_bf16_and_keeps_pruned_apart():
    names = {'minimax_h3_ref2va_pruned_bf16.safetensors', 'minimax_h3_ref2va_bf16.safetensors',
             'minimax_h3_fl2va_pruned_int8_convrot.safetensors', 'MiniMax-H3/minimax_h3_ref2va_pruned_int8_convrot.safetensors'}
    want = 'minimax_h3_ref2va_pruned_int8_convrot.safetensors'
    # The same file in a subfolder wins first (resolve_model's basename match), then the bf16 sibling.
    assert workflows.resolve_model('UNETLoader', want, names) == 'MiniMax-H3/minimax_h3_ref2va_pruned_int8_convrot.safetensors'
    names.discard('MiniMax-H3/minimax_h3_ref2va_pruned_int8_convrot.safetensors')
    assert workflows.resolve_model('UNETLoader', want, names) == 'minimax_h3_ref2va_pruned_bf16.safetensors'
    assert pc.quant_sibling(want, {'minimax_h3_fl2va_pruned_int8_convrot.safetensors'}) is None


def test_text_encoder_falls_back_to_installed_nvfp4():
    names = {'MiniMax-H3/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'}
    assert workflows.resolve_model('CLIPLoader', 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors', names) == next(iter(names))


def test_loras_never_swap_quantization():
    names = {'henge/minimax_h3_lms_v1.0_r64_fp16.safetensors'}
    assert workflows.resolve_model('LoraLoaderModelOnly', 'henge/minimax_h3_lms_v1.0_r64.safetensors', names) == 'henge/minimax_h3_lms_v1.0_r64.safetensors'


def test_gguf_only_install_switches_loader(monkeypatch):
    monkeypatch.setattr(workflows, 'INSTALLED', {'UNETLoader': set(), 'UnetLoaderGGUF': {'MiniMax-H3-Ref2VA-Pruned-Q8_0.gguf', 'minimax_h3_ref2va_pruned-Q4_K.gguf'}})
    g = workflows.localize_loras({'1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'minimax_h3_ref2va_pruned_int8_convrot.safetensors', 'weight_dtype': 'default'}}})
    assert g['1']['class_type'] == 'UnetLoaderGGUF'
    assert g['1']['inputs'] == {'unet_name': 'MiniMax-H3-Ref2VA-Pruned-Q8_0.gguf'}


def test_new_combo_format_is_listed():
    import requirements_check
    info={'LoadBackgroundRemovalModel':{'input':{'required':{'bg_removal_name':['COMBO',{'options':['BiRefNet/birefnet.safetensors']}]}}}}
    assert 'BiRefNet/birefnet.safetensors' in requirements_check.comfy_listed(info)


def test_background_removal_resolves_subfolder(monkeypatch):
    monkeypatch.setattr(workflows, 'INSTALLED', {'LoadBackgroundRemovalModel': {'BiRefNet/birefnet.safetensors'}})
    g = workflows.localize_loras({'2': {'class_type': 'LoadBackgroundRemovalModel', 'inputs': {'bg_removal_name': 'birefnet.safetensors'}}})
    assert g['2']['inputs']['bg_removal_name'] == 'BiRefNet/birefnet.safetensors'
