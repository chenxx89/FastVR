"""FastVR inference implementation with lazily imported integration APIs."""

__all__ = [
    "FastVRInferenceConfig",
    "FastVRRuntime",
    "enhance_video_tensor",
    "load_fastvr_pipeline",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from . import api

    return getattr(api, name)
