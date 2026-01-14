"""HFE-Models: Model wrappers for hallucination detection."""

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "QwenWrapper":
        from hfe_models.qwen import QwenWrapper

        return QwenWrapper
    elif name == "VLLMWrapper":
        from hfe_models.vllm_wrapper import VLLMWrapper

        return VLLMWrapper
    elif name == "HFSampler":
        from hfe_models.hf_sampler import HFSampler

        return HFSampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "QwenWrapper",
    "VLLMWrapper",
    "HFSampler",
]
