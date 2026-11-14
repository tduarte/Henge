"""Run Henge's graphs on a ComfyUI without NVIDIA kernels (Apple Silicon / MPS, or CPU).

The published H3 recipes use CUDA-only pieces: Comfy Kitchen INT8 attention, the TARA kernel's Kitchen backend and
Sol sparse attention (Triton). On a ComfyUI whose torch device is not CUDA, adapt() swaps them for the plain PyTorch
(SDPA) equivalents just before the graph is queued. Nothing changes on a CUDA backend.

HENGE_PLATFORM overrides the detection: 'cuda' keeps every graph as published, 'mps' or 'cpu' always adapts,
'auto' (default) asks the ComfyUI it talks to through /system_stats.
"""
import os,re,time
import requests

_CACHE={'url':None,'device':None,'at':0.}
CUDA_ONLY_ATTENTION={'comfy kitchen attention':'pytorch attention'}
TARA_FALLBACK='flash partitions (no compile)'

def device_type(url,timeout=5):
    """'cuda', 'mps', 'cpu' (or another torch device type) of the ComfyUI at url; None when it does not answer."""
    forced=os.environ.get('HENGE_PLATFORM','auto').strip().lower()
    if forced and forced!='auto':return forced
    if _CACHE['url']==url and _CACHE['device'] and time.time()-_CACHE['at']<300:return _CACHE['device']
    try:
        devices=requests.get(url.rstrip('/')+'/system_stats',timeout=timeout).json().get('devices') or []
        kind=str((devices[0] or {}).get('type') or '').lower() or None
    except (requests.RequestException,ValueError,IndexError,AttributeError):kind=None
    if kind:_CACHE.update(url=url,device=kind,at=time.time())
    return kind

def needs_adapting(kind):return bool(kind) and kind!='cuda'

def _bypass(graph,node_id):
    """Remove a MODEL->MODEL patch node, pointing its consumers at the node's own model input."""
    src=graph[node_id]['inputs'].get('model')
    if not isinstance(src,list):return False
    for other in graph.values():
        for key,value in (other.get('inputs') or {}).items():
            if value==[node_id,0]:other['inputs'][key]=src
    del graph[node_id]
    return True

def adapt(graph,kind):
    """The graph with its CUDA-only attention swapped for PyTorch equivalents when kind is not 'cuda'. In place."""
    if not needs_adapting(kind):return graph
    for node_id,n in list(graph.items()):
        cls=n.get('class_type');inputs=n.get('inputs') or {}
        if cls=='ModelAttentionBackend' and inputs.get('attention') in CUDA_ONLY_ATTENTION:
            inputs['attention']=CUDA_ONLY_ATTENTION[inputs['attention']]
        elif cls=='BFSMiniMaxH3NativeTemporal' and 'kitchen' in str(inputs.get('attention_backend','')):
            inputs['attention_backend']=TARA_FALLBACK
        elif cls=='BlockSparseAttention' and str(inputs.get('selection','')).startswith('sol'):
            _bypass(graph,node_id)   # Sol sparse attention needs Triton; dense SDPA gives the same result, slower
        elif cls=='MinimaxH3LatentUpscaler3D' and inputs.get('device')=='cuda':
            inputs['device']=kind
    return graph

# Quantization tokens in published file names. A file named for one format is matched with a sibling of another format
# (int8_convrot -> bf16, nvfp4_awq, GGUF ...) when only that sibling is installed, so a Mac can keep MPS-friendly weights.
QUANT=re.compile(r'[_.-](?:pruned[_-])?(?:int8[_-]convrot|int8|nvfp4(?:[_-]awq)?|fp8(?:[_-]e4m3fn(?:[_-]scaled)?)?|mxfp8|w4a8(?:[_-]gptq)?|awq|gptq|bf16|fp16|fp32|q[2-8](?:_[a-z0-9]+)*|f16|f32)(?=[_.-]|$)',re.I)
FORMAT_ORDER=os.environ.get('HENGE_QUANT_PREFERENCE','bf16,fp16,q8_0,int8_convrot,nvfp4_awq,fp32').lower().split(',')

def quant_stem(name):
    base=name.replace('\\','/').rsplit('/',1)[-1]
    base=re.sub(r'\.(safetensors|gguf|ckpt|pt|pth|sft)$','',base,flags=re.I)
    keep_pruned='pruned' in base.lower()
    stem=re.sub(r'[-_. ]+','_',QUANT.sub('',base).lower()).strip('_')   # MiniMax-H3-Ref2VA == minimax_h3_ref2va
    return stem+('|pruned' if keep_pruned else '')

def _rank(name):
    low=name.lower()
    for i,fmt in enumerate(FORMAT_ORDER):
        if fmt and fmt in low:return i
    return len(FORMAT_ORDER)

def quant_sibling(name,names):
    """An installed file that is the same model as name in another quantization, preferred per HENGE_QUANT_PREFERENCE."""
    stem=quant_stem(name)
    hits=[n for n in names if quant_stem(n)==stem]
    return sorted(hits,key=lambda n:(_rank(n),n))[0] if hits else None

# Loader classes for GGUF files (city96/ComfyUI-GGUF). A UNET/CLIP loader whose resolved file is .gguf is switched over.
GGUF_LOADERS={'UNETLoader':('UnetLoaderGGUF','unet_name'),'CLIPLoader':('CLIPLoaderGGUF','clip_name')}
