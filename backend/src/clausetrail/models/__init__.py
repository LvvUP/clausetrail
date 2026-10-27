def want_fp16(setting: str | bool) -> bool:
    """本地 bge 模型要不要用 FP16：true / false 按配置；auto 时只在有 CUDA GPU 时开启。"""
    if setting != "auto":
        return bool(setting)
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())
