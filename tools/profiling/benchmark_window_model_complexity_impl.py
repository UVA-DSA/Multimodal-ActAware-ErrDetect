#!/usr/bin/env python3
"""
Implementation module for the encoder-aware window complexity benchmark.
"""

import argparse
import atexit
import glob
import importlib
import importlib.util
import json
import math
import os
import pickle
import sys
import tempfile
import time
import types
from pathlib import Path
from statistics import mean, median, pstdev
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from dataclasses import asdict, dataclass
except ImportError:
    def dataclass(_cls=None, **kwargs):
        del kwargs

        def wrap(cls):
            field_names = list(getattr(cls, "__annotations__", {}).keys())

            def __init__(self, *args, **init_kwargs):
                if len(args) > len(field_names):
                    raise TypeError("too many positional arguments for dataclass fallback")
                remaining = dict(init_kwargs)
                for name, value in zip(field_names, args):
                    setattr(self, name, value)
                    remaining.pop(name, None)
                for name in field_names[len(args):]:
                    if name in remaining:
                        setattr(self, name, remaining.pop(name))
                    elif hasattr(cls, name):
                        setattr(self, name, getattr(cls, name))
                    else:
                        raise TypeError("missing required argument: {}".format(name))
                if remaining:
                    unexpected = ", ".join(sorted(remaining.keys()))
                    raise TypeError("unexpected arguments: {}".format(unexpected))

            cls.__init__ = __init__
            return cls

        if _cls is None:
            return wrap
        return wrap(_cls)

    def asdict(instance):
        return dict(instance.__dict__)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange, repeat
from torchvision import models, transforms

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHAIN_DIR = REPO_ROOT / "baselines" / "Chain-of-Gesture"
HF_CACHE_DIR = Path(os.environ.get("HF_HOME", REPO_ROOT / "outputs" / ".hf_cache"))

for _p in (REPO_ROOT, REPO_ROOT / "jigsaws", REPO_ROOT / "SAR_RARP"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def prepare_hf_hub_environment() -> None:
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Xet-backed downloads have been flaky on this cluster for large checkpoints.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))


prepare_hf_hub_environment()

from textprompts import (  # noqa: E402
    Context_Prompt,
    error_prompt,
    gesture_error_prompt,
    gesture_prompt,
    lowlevel_gesture_error_prompt,
)

PROMPT_SETS: Dict[str, Sequence[str]] = {
    "error": error_prompt,
    "gesture_error": gesture_error_prompt,
    "gesture": gesture_prompt,
    "lowlevel_gesture_error": lowlevel_gesture_error_prompt,
    "context": Context_Prompt,
}

WINDOW_LENGTH = 10
RESNET_FEATURE_DIM = 2048
CLIP_BASE_FEATURE_DIM = 768
DINOV2_LARGE_FEATURE_DIM = 1024
SED_DINOV2_FEATURE_DIM = 1000
KINEMATICS_DIM = 14
KIN_DOWNSAMPLE = 3
GVR_TEXT_DIM = 512

DEFAULT_DUMMY_BATCHES = 3
DEFAULT_WARMUP_ITERS = 1
DEFAULT_TIMING_ITERS = 3
DEFAULT_ENCODER_CHUNK_SIZE = 32
DEFAULT_FALLBACK_IMAGE_HEIGHT = 480
DEFAULT_FALLBACK_IMAGE_WIDTH = 640
DEFAULT_SEDMAMBA_LENGTHS = (120, 240, 360)

RESNET50_ENCODER_KEY = "resnet50"
CLIP_BASE_ENCODER_KEY = "clip-vit-base-patch32"
DINOV2_LARGE_ENCODER_KEY = "dinov2-large"
SED_DINOV2_GIANT_ENCODER_KEY = "dinov2-giant-reg-imagenet1k-1layer"

RESNET50_ENCODER_NAME = "ResNet50"
CLIP_BASE_ENCODER_NAME = "CLIP ViT-B/32"
DINOV2_LARGE_ENCODER_NAME = "DINOv2-Large"
SED_DINOV2_GIANT_ENCODER_NAME = "DINOv2-Giant-Reg-ImageNet1k-1L"

_TEMP_FILES: List[str] = []


@dataclass(frozen=True)
class RawImageProfile:
    dataset_name: str
    height: int
    width: int
    source: str


@dataclass
class EncoderRuntime:
    key: str
    display_name: str
    feature_dim: int
    model: torch.nn.Module
    preprocess_description: str
    preprocess_chunks: Callable[[Dict[str, Any]], List[Any]]
    encode_chunk: Callable[[Any], torch.Tensor]


@dataclass
class BenchmarkSpec:
    model_family: str
    model_name: str
    prompt_type: str
    encoder_key: str
    encoder_name: str
    feature_source: str
    benchmark_window_length: int
    sequence_profile: str
    input_description: str
    notes: str
    model: torch.nn.Module
    raw_batches: Sequence[Dict[str, Any]]
    prepare_raw_batch: Callable[[Dict[str, Any], EncoderRuntime], Dict[str, Any]]
    build_model_inputs: Callable[[Dict[str, Any], torch.Tensor, torch.device], Dict[str, Any]]
    forward_model: Callable[[torch.nn.Module, Dict[str, Any]], torch.Tensor]


@dataclass
class BenchmarkResult:
    model_family: str
    model_name: str
    prompt_type: str
    encoder_name: str
    feature_source: str
    benchmark_window_length: int
    sequence_profile: str
    input_description: str
    output_shape: str
    encoder_params: int
    model_params: int
    flops_per_window: int
    streaming_latency_ms: float
    streaming_latency_std_ms: float
    encoder_latency_ms: float
    encoder_latency_std_ms: float
    model_latency_ms: float
    model_latency_std_ms: float
    end_to_end_latency_ms: float
    end_to_end_latency_std_ms: float
    device: str
    notes: str


def register_temp_file(path: str) -> str:
    _TEMP_FILES.append(path)
    return path


@atexit.register
def _cleanup_temp_files() -> None:
    for path in _TEMP_FILES:
        try:
            os.remove(path)
        except OSError:
            pass


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def recursive_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: recursive_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(recursive_to_device(value, device) for value in obj)
    return obj


def freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def total_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def tensor_shape_to_str(tensor: torch.Tensor) -> str:
    return "x".join(str(dim) for dim in tensor.shape)


LATENCY_STAT = "median"


def summarize_samples(samples: Sequence[float]) -> Tuple[float, float]:
    """Central latency estimate + spread.

    Defaults to the median. On a shared GPU node a handful of iterations
    occasionally stall (contention, CPU-side preprocessing jitter, one-time
    allocation), and a single such sample drags the mean far off the typical
    cost -- observed as rows whose std exceeded their mean. The median ignores
    those spikes; for well-behaved rows it is indistinguishable from the mean.
    The reported spread stays the standard deviation, so a large std against a
    stable median still flags a noisy measurement.
    """
    if not samples:
        return 0.0, 0.0
    centre = median(samples) if LATENCY_STAT == "median" else mean(samples)
    return float(centre), float(pstdev(samples) if len(samples) > 1 else 0.0)


def deterministic_prompt_features(num_prompts: int, dim: int, offset: int = 0) -> torch.Tensor:
    base = torch.arange(num_prompts * dim, dtype=torch.float32).reshape(num_prompts, dim)
    return torch.sin((base + float(offset)) / 37.0)


def deterministic_feature_matrix(length: int, feature_dim: int, offset: int = 0) -> torch.Tensor:
    base = torch.arange(length * feature_dim, dtype=torch.float32).reshape(length, feature_dim)
    return torch.cos((base + float(offset)) / 53.0)


def get_resnet_transform(image_size: int = 224, normalize: bool = True):
    ops = [
        transforms.Resize((image_size + 16, image_size + 16)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    if normalize:
        ops.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    return transforms.Compose(ops)


def split_list_chunks(items: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def resolve_sar_rarp_root(data_root: Path) -> Path:
    candidate = data_root / "SAR_RARP50"
    if candidate.exists():
        return candidate
    return data_root


def discover_image_profile(
    *,
    dataset_name: str,
    patterns: Sequence[str],
    fallback_height: int = DEFAULT_FALLBACK_IMAGE_HEIGHT,
    fallback_width: int = DEFAULT_FALLBACK_IMAGE_WIDTH,
) -> RawImageProfile:
    for pattern in patterns:
        for match in glob.iglob(pattern):
            try:
                with Image.open(match) as img:
                    width, height = img.size
                return RawImageProfile(dataset_name, int(height), int(width), f"detected from {match}")
            except OSError:
                continue
    return RawImageProfile(dataset_name, fallback_height, fallback_width, f"fallback {fallback_height}x{fallback_width}")


def rounded_window_length(value: int) -> int:
    return max(WINDOW_LENGTH, int(round(float(value) / float(WINDOW_LENGTH)) * WINDOW_LENGTH))


def fallback_sedmamba_lengths(num_lengths: int) -> List[int]:
    base = list(DEFAULT_SEDMAMBA_LENGTHS)
    while len(base) < num_lengths:
        base.append(base[-1] + WINDOW_LENGTH * 6)
    return base[:num_lengths]


def discover_sar_rarp_sequence_lengths(data_root: Path, num_lengths: int) -> Tuple[List[int], str]:
    sar_root = resolve_sar_rarp_root(data_root)
    lengths: List[int] = []
    for pkl_root in (sar_root / "train_emb_DINOv2", sar_root / "test_emb_DINOv2"):
        if not pkl_root.exists():
            continue
        for pkl_path in sorted(pkl_root.glob("*.pkl")):
            try:
                with pkl_path.open("rb") as handle:
                    payload = pickle.load(handle)
                labels = payload.get("error_GT")
                if labels is not None:
                    lengths.append(int(len(labels)))
            except Exception:
                continue
    if lengths:
        quantiles = np.linspace(0.2, 0.8, num=max(num_lengths, 1))
        chosen: List[int] = []
        for quantile in quantiles:
            length = rounded_window_length(int(round(float(np.quantile(lengths, quantile)))))
            if length not in chosen:
                chosen.append(length)
        while len(chosen) < num_lengths:
            chosen.append(chosen[-1])
        return chosen[:num_lengths], f"derived from {len(lengths)} SAR-RARP50 label sequences under {sar_root}"
    fallback = fallback_sedmamba_lengths(num_lengths)
    return fallback, f"fallback representative SAR-RARP50-like lengths {fallback}"


def frame_windows_per_batch(raw_batch: Dict[str, Any]) -> int:
    return max(1, math.ceil(int(raw_batch["sequence_length"]) / WINDOW_LENGTH))


def format_sequence_profile(raw_batches: Sequence[Dict[str, Any]]) -> str:
    unique: List[str] = []
    for batch in raw_batches:
        length = str(int(batch["sequence_length"]))
        if length not in unique:
            unique.append(length)
    return "|".join(unique)


def make_frame_bank(
    *,
    sequence_length: int,
    image_height: int,
    image_width: int,
    seed: int,
    max_bank_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    bank_size = max(1, min(sequence_length, max_bank_size))
    rng = np.random.default_rng(seed)
    frame_bank = rng.integers(0, 256, size=(bank_size, image_height, image_width, 3), dtype=np.uint8)
    frame_indices = rng.integers(0, bank_size, size=(sequence_length,), dtype=np.int64)
    return frame_bank, frame_indices


def make_raw_image_batches(*, sequence_lengths: Sequence[int], image_profile: RawImageProfile, seed_offset: int) -> List[Dict[str, Any]]:
    batches: List[Dict[str, Any]] = []
    for index, sequence_length in enumerate(sequence_lengths):
        frame_bank, frame_indices = make_frame_bank(
            sequence_length=sequence_length,
            image_height=image_profile.height,
            image_width=image_profile.width,
            seed=1000 + seed_offset + index,
        )
        batches.append(
            {
                "sequence_length": int(sequence_length),
                "frame_bank_uint8": frame_bank,
                "frame_indices": frame_indices,
                "raw_image_height": image_profile.height,
                "raw_image_width": image_profile.width,
            }
        )
    return batches


def make_raw_kin_batches(*, sequence_lengths: Sequence[int], image_profile: RawImageProfile, seed_offset: int) -> List[Dict[str, Any]]:
    batches = make_raw_image_batches(sequence_lengths=sequence_lengths, image_profile=image_profile, seed_offset=seed_offset)
    for index, batch in enumerate(batches):
        seq_len = int(batch["sequence_length"])
        rng = np.random.default_rng(3000 + seed_offset + index)
        batch["kine_raw"] = rng.normal(0.0, 1.0, size=(seq_len * KIN_DOWNSAMPLE, KINEMATICS_DIM)).astype(np.float32)
    return batches


def make_raw_dual_batches(
    *,
    sequence_lengths: Sequence[int],
    image_profile: RawImageProfile,
    feature_dim: int,
    seed_offset: int,
) -> List[Dict[str, Any]]:
    batches = make_raw_image_batches(sequence_lengths=sequence_lengths, image_profile=image_profile, seed_offset=seed_offset)
    for index, batch in enumerate(batches):
        seq_len = int(batch["sequence_length"])
        batch["gesture_feature_stream"] = deterministic_feature_matrix(seq_len, feature_dim, offset=5000 + seed_offset + index)
    return batches


def materialize_frame_arrays(raw_batch: Dict[str, Any]) -> np.ndarray:
    return raw_batch["frame_bank_uint8"][raw_batch["frame_indices"]].copy()


def materialize_pil_images(raw_batch: Dict[str, Any]) -> List[Image.Image]:
    return [Image.fromarray(frame, mode="RGB") for frame in materialize_frame_arrays(raw_batch)]


def prepare_jigsaws_kinematics(raw_kinematics: np.ndarray, target_length: int) -> torch.Tensor:
    array = np.asarray(raw_kinematics, dtype=np.float32)
    mean_values = array.mean(axis=0, keepdims=True)
    std_values = array.std(axis=0, ddof=1, keepdims=True)
    std_values = np.where(np.isfinite(std_values) & (std_values > 1e-6), std_values, 1.0)
    standardized = (array - mean_values) / std_values
    downsampled = standardized[::KIN_DOWNSAMPLE]
    if downsampled.shape[0] > target_length:
        downsampled = downsampled[:target_length]
    elif downsampled.shape[0] < target_length:
        padding = np.zeros((target_length - downsampled.shape[0], KINEMATICS_DIM), dtype=np.float32)
        downsampled = np.concatenate([downsampled, padding], axis=0)
    return torch.from_numpy(downsampled.astype(np.float32, copy=False))


def load_module_from_file(module_name: str, file_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_transformers_text_shim():
    fake_module = types.ModuleType("transformers")

    class _UnusedFactory:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError(
                "The benchmark should not load CLIP text encoders from transformers. "
                "All GVR benchmark rows are expected to pass synthetic prompt_features."
            )

    fake_module.CLIPTextModel = _UnusedFactory
    fake_module.CLIPTokenizer = _UnusedFactory
    previous_module = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_module
    return previous_module


def _restore_transformers_module(previous_module) -> None:
    if previous_module is None:
        sys.modules.pop("transformers", None)
    else:
        sys.modules["transformers"] = previous_module


def import_gvr_module():
    if "benchmark_model_gvr_features" in sys.modules:
        return sys.modules["benchmark_model_gvr_features"]
    previous_module = _install_transformers_text_shim()
    try:
        return load_module_from_file("benchmark_model_gvr_features", REPO_ROOT / "jigsaws" / "model_gvr_features.py")
    finally:
        _restore_transformers_module(previous_module)


def _install_mamba_package_stubs() -> None:
    spec = importlib.util.find_spec("mamba_ssm")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("Could not locate the installed mamba_ssm package.")
    root = Path(next(iter(spec.submodule_search_locations)))
    package_paths = {
        "mamba_ssm": root,
        "mamba_ssm.modules": root / "modules",
        "mamba_ssm.ops": root / "ops",
        "mamba_ssm.ops.triton": root / "ops" / "triton",
        "mamba_ssm.utils": root / "utils",
    }
    for name, path in package_paths.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package


def _build_cpu_selective_scan_module() -> types.ModuleType:
    module = types.ModuleType("mamba_ssm.ops.selective_scan_interface")

    def selective_scan_ref(
        u,
        delta,
        A,
        B,
        C,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=False,
    ):
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        for index in range(u.shape[2]):
            x = deltaA[:, :, index] * x + deltaB_u[:, :, index]
            if not is_variable_C:
                y = torch.einsum("bdn,dn->bd", x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum("bdn,bn->bd", x, C[:, :, index])
                else:
                    y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, index])
            if index == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2)
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        return out if not return_last_state else (out, last_state)

    module.selective_scan_ref = selective_scan_ref
    module.selective_scan_fn = selective_scan_ref
    module.mamba_inner_fn = None
    return module


def import_sedmamba_module(device: torch.device):
    _install_mamba_package_stubs()
    sys.modules.pop("mamba_ssm.modules.mamba_simple", None)
    if device.type == "cpu":
        sys.modules["mamba_ssm.ops.selective_scan_interface"] = _build_cpu_selective_scan_module()
    else:
        sys.modules.pop("mamba_ssm.ops.selective_scan_interface", None)
    mamba_simple = importlib.import_module("mamba_ssm.modules.mamba_simple")
    if device.type == "cpu":
        mamba_simple.causal_conv1d_fn = None
    sys.modules["mamba_ssm"].Mamba = mamba_simple.Mamba
    return load_module_from_file("benchmark_sedmamba_baseline", REPO_ROOT / "baselines" / "SEDMamba" / "baseline" / "SEDMamba.py")


def install_fake_scipy() -> None:
    fake_scipy = types.ModuleType("scipy")
    fake_interpolate = types.ModuleType("scipy.interpolate")
    fake_scipy.interpolate = fake_interpolate
    sys.modules["scipy"] = fake_scipy
    sys.modules["scipy.interpolate"] = fake_interpolate


def _install_scipy_shim():
    previous_scipy = sys.modules.get("scipy")
    previous_interpolate = sys.modules.get("scipy.interpolate")
    install_fake_scipy()
    return previous_scipy, previous_interpolate


def _restore_scipy_modules(previous_scipy, previous_interpolate) -> None:
    if previous_scipy is None:
        sys.modules.pop("scipy", None)
    else:
        sys.modules["scipy"] = previous_scipy
    if previous_interpolate is None:
        sys.modules.pop("scipy.interpolate", None)
    else:
        sys.modules["scipy.interpolate"] = previous_interpolate


class FakeClipTextModel:
    def encode_text(self, tokenized: torch.Tensor) -> torch.Tensor:
        batch = tokenized.shape[0]
        base = torch.arange(batch * 512, dtype=torch.float32).reshape(batch, 512)
        return torch.cos(base / 53.0)


class FakeClipModule:
    def load(self, model_name: str, device: str = "cpu"):
        return FakeClipTextModel(), None

    def tokenize(self, texts: str | Sequence[str]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        return torch.zeros(len(texts), 77, dtype=torch.long)


def import_cog_module():
    if "benchmark_chain_of_gesture_models" in sys.modules:
        return sys.modules["benchmark_chain_of_gesture_models"]
    if str(CHAIN_DIR) not in sys.path:
        sys.path.insert(0, str(CHAIN_DIR))
    previous_scipy, previous_interpolate = _install_scipy_shim()
    try:
        module = load_module_from_file("benchmark_chain_of_gesture_models", CHAIN_DIR / "models.py")
    finally:
        _restore_scipy_modules(previous_scipy, previous_interpolate)
    module.clip = FakeClipModule()
    return module


def build_resnet50_backbone() -> torch.nn.Module:
    """A frozen ResNet50 trunk (fc removed), identical to the one build_resnet50_runtime uses."""
    weight_enum = getattr(models, "ResNet50_Weights", None)
    weights = weight_enum.DEFAULT if weight_enum is not None else None
    backbone = models.resnet50(weights=weights)
    backbone.fc = torch.nn.Identity()
    return freeze_module(backbone)


class ActivityAwareDualStream(torch.nn.Module):
    """Activity-aware dual-stream model that owns its activity-embedding vision encoder.

    The activity-aware setup runs the *same* video frames through two vision encoders:
    one producing generic spatial features, one producing activity-aware embeddings.
    Only the spatial-feature encoder is the benchmark's `encoder`; the activity-embedding
    encoder is owned here, so its parameters, FLOPs and latency are attributed to the
    model rather than to the encoder.
    """

    def __init__(self, inner: torch.nn.Module, activity_encoder: torch.nn.Module):
        super().__init__()
        self.inner = inner
        self.activity_encoder = activity_encoder

    def forward(self, base_features: torch.Tensor, ges_pixels: torch.Tensor, masks=None):
        feats = self.activity_encoder(ges_pixels)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.ndim > 2:
            feats = feats.flatten(1)
        ges_features = feats.to(base_features.dtype).unsqueeze(0)
        return self.inner(base_features, ges_features, masks=masks)


def build_resnet50_runtime(device: torch.device, chunk_size: int) -> EncoderRuntime:
    weight_enum = getattr(models, "ResNet50_Weights", None)
    weights = weight_enum.DEFAULT if weight_enum is not None else None
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model = freeze_module(model.to(device))
    transform = get_resnet_transform(224, True)

    def preprocess_chunks(raw_batch: Dict[str, Any]) -> List[torch.Tensor]:
        images = materialize_pil_images(raw_batch)
        return [
            torch.stack([transform(image) for image in image_chunk]).to(torch.float32)
            for image_chunk in split_list_chunks(images, chunk_size)
        ]

    def encode_chunk(chunk: torch.Tensor) -> torch.Tensor:
        outputs = model(chunk)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        if outputs.ndim > 2:
            outputs = outputs.flatten(1)
        return outputs.to(torch.float32)

    return EncoderRuntime(
        key=RESNET50_ENCODER_KEY,
        display_name=RESNET50_ENCODER_NAME,
        feature_dim=RESNET_FEATURE_DIM,
        model=model,
        preprocess_description="Resize(240,240) -> CenterCrop(224) -> ToTensor -> ImageNet normalize",
        preprocess_chunks=preprocess_chunks,
        encode_chunk=encode_chunk,
    )


def hf_from_pretrained_kwargs() -> Dict[str, Any]:
    prepare_hf_hub_environment()
    return {
        "cache_dir": str(HF_CACHE_DIR),
    }


def build_clip_base_runtime(device: torch.device, chunk_size: int) -> EncoderRuntime:
    from transformers import CLIPProcessor, CLIPVisionModel

    pretrained_kwargs = hf_from_pretrained_kwargs()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", **pretrained_kwargs)
    model = freeze_module(
        CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", **pretrained_kwargs).to(device)
    )

    def preprocess_chunks(raw_batch: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        images = materialize_pil_images(raw_batch)
        return [
            {"pixel_values": processor(images=list(image_chunk), return_tensors="pt")["pixel_values"].to(torch.float32)}
            for image_chunk in split_list_chunks(images, chunk_size)
        ]

    def encode_chunk(chunk: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = model(**chunk)
        features = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None else outputs.last_hidden_state[:, 0, :]
        return features.to(torch.float32)

    return EncoderRuntime(
        key=CLIP_BASE_ENCODER_KEY,
        display_name=CLIP_BASE_ENCODER_NAME,
        feature_dim=CLIP_BASE_FEATURE_DIM,
        model=model,
        preprocess_description="CLIPProcessor image preprocessing for ViT-B/32",
        preprocess_chunks=preprocess_chunks,
        encode_chunk=encode_chunk,
    )


def build_dinov2_large_runtime(device: torch.device, chunk_size: int) -> EncoderRuntime:
    from transformers import AutoImageProcessor, AutoModel

    pretrained_kwargs = hf_from_pretrained_kwargs()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large", **pretrained_kwargs)
    model = freeze_module(AutoModel.from_pretrained("facebook/dinov2-large", **pretrained_kwargs).to(device))

    def preprocess_chunks(raw_batch: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        images = materialize_pil_images(raw_batch)
        return [
            {"pixel_values": processor(images=list(image_chunk), return_tensors="pt")["pixel_values"].to(torch.float32)}
            for image_chunk in split_list_chunks(images, chunk_size)
        ]

    def encode_chunk(chunk: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = model(**chunk)
        features = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None else outputs.last_hidden_state[:, 0, :]
        return features.to(torch.float32)

    return EncoderRuntime(
        key=DINOV2_LARGE_ENCODER_KEY,
        display_name=DINOV2_LARGE_ENCODER_NAME,
        feature_dim=DINOV2_LARGE_FEATURE_DIM,
        model=model,
        preprocess_description="AutoImageProcessor preprocessing for DINOv2-Large",
        preprocess_chunks=preprocess_chunks,
        encode_chunk=encode_chunk,
    )


def build_sed_dinov2_giant_runtime(device: torch.device, chunk_size: int) -> EncoderRuntime:
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    model_id = "facebook/dinov2-with-registers-giant-imagenet1k-1-layer"
    pretrained_kwargs = hf_from_pretrained_kwargs()
    processor = AutoImageProcessor.from_pretrained(model_id, **pretrained_kwargs)
    try:
        model = freeze_module(
            AutoModelForImageClassification.from_pretrained(model_id, **pretrained_kwargs).to(device)
        )
    except OSError as exc:
        raise RuntimeError(
            "Failed to load the SEDMamba DINOv2 giant-with-registers classifier checkpoint. "
            "The benchmark now disables Xet and uses a repo-local Hugging Face cache at "
            f"`{HF_CACHE_DIR}`, so please retry the run. If it still fails, remove any partial "
            f"download for `{model_id}` under that cache directory and rerun."
        ) from exc

    def preprocess_chunks(raw_batch: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        images = materialize_pil_images(raw_batch)
        return [
            {"pixel_values": processor(images=list(image_chunk), return_tensors="pt")["pixel_values"].to(torch.float32)}
            for image_chunk in split_list_chunks(images, chunk_size)
        ]

    def encode_chunk(chunk: Dict[str, torch.Tensor]) -> torch.Tensor:
        return model(**chunk).logits.to(torch.float32)

    return EncoderRuntime(
        key=SED_DINOV2_GIANT_ENCODER_KEY,
        display_name=SED_DINOV2_GIANT_ENCODER_NAME,
        feature_dim=SED_DINOV2_FEATURE_DIM,
        model=model,
        preprocess_description="AutoImageProcessor preprocessing for DINOv2-Giant-Reg ImageNet1k classifier",
        preprocess_chunks=preprocess_chunks,
        encode_chunk=encode_chunk,
    )


def build_encoder_runtime(encoder_key: str, device: torch.device, chunk_size: int) -> EncoderRuntime:
    if encoder_key == RESNET50_ENCODER_KEY:
        return build_resnet50_runtime(device, chunk_size)
    if encoder_key == CLIP_BASE_ENCODER_KEY:
        return build_clip_base_runtime(device, chunk_size)
    if encoder_key == DINOV2_LARGE_ENCODER_KEY:
        return build_dinov2_large_runtime(device, chunk_size)
    if encoder_key == SED_DINOV2_GIANT_ENCODER_KEY:
        return build_sed_dinov2_giant_runtime(device, chunk_size)
    raise ValueError(f"Unknown encoder key: {encoder_key}")


def ensure_transformers_available() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        message = str(exc)
        if "h11" in message:
            raise RuntimeError(
                "The current Python environment can see `transformers`, but its HTTP stack is incomplete: "
                f"{message}. Install `h11` into the same environment or user-site packages that provide "
                "`transformers` and `huggingface_hub`, then rerun the benchmark."
            ) from exc
        raise RuntimeError(
            "The benchmark requires `transformers` for the CLIP and DINOv2 encoder rows. "
            "Please run it in an environment that has `transformers` installed, or install "
            "`transformers` and its dependencies into the current environment."
        ) from exc


def prepare_image_only_batch(raw_batch: Dict[str, Any], encoder_runtime: EncoderRuntime) -> Dict[str, Any]:
    seq_len = int(raw_batch["sequence_length"])
    return {"sequence_length": seq_len, "encoder_chunks": encoder_runtime.preprocess_chunks(raw_batch), "masks": torch.ones(seq_len, dtype=torch.bool)}


def prepare_kin_batch(raw_batch: Dict[str, Any], encoder_runtime: EncoderRuntime) -> Dict[str, Any]:
    prepared = prepare_image_only_batch(raw_batch, encoder_runtime)
    prepared["kine"] = prepare_jigsaws_kinematics(raw_batch["kine_raw"], int(raw_batch["sequence_length"]))
    return prepared


def prepare_dual_batch(raw_batch: Dict[str, Any], encoder_runtime: EncoderRuntime) -> Dict[str, Any]:
    prepared = prepare_image_only_batch(raw_batch, encoder_runtime)
    prepared["ges_features"] = raw_batch["gesture_feature_stream"].clone().to(torch.float32)
    return prepared


def build_sedmamba_model_inputs(prepared_batch: Dict[str, Any], encoded_features: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    del prepared_batch, device
    return {"features": encoded_features.transpose(0, 1).unsqueeze(0)}


def build_cog_model_inputs(prepared_batch: Dict[str, Any], encoded_features: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    del prepared_batch, device
    return {"features": encoded_features.unsqueeze(0)}


def build_gvr_pred_model_inputs(prepared_batch: Dict[str, Any], encoded_features: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    return {"features": encoded_features.unsqueeze(0), "masks": prepared_batch["masks"].unsqueeze(0).to(device)}


def build_gvr_kin_model_inputs(prepared_batch: Dict[str, Any], encoded_features: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    return {
        "features": encoded_features.unsqueeze(0),
        "kine": prepared_batch["kine"].unsqueeze(0).to(device),
        "masks": prepared_batch["masks"].unsqueeze(0).to(device),
    }


def build_gvr_context_model_inputs(prepared_batch: Dict[str, Any], encoded_features: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    del prepared_batch, device
    return {"features": encoded_features.unsqueeze(0)}


def build_gvr_dual_model_inputs(prepared_batch: Dict[str, Any], encoded_features: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    # The activity-embedding encoder lives inside the model, so it receives pixels, not
    # features -- the same preprocessed frames the spatial encoder consumed.
    pixels = torch.cat([chunk.to(device) for chunk in prepared_batch["encoder_chunks"]], dim=0)
    return {
        "base_features": encoded_features.unsqueeze(0),
        "ges_pixels": pixels,
        "masks": prepared_batch["masks"].unsqueeze(0).to(device),
    }


def encode_chunks(encoder_runtime: EncoderRuntime, encoder_chunks: Sequence[Any], device: torch.device) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    with torch.inference_mode():
        for chunk in encoder_chunks:
            outputs.append(encoder_runtime.encode_chunk(recursive_to_device(chunk, device)))
    if outputs:
        return torch.cat(outputs, dim=0)
    return torch.empty(0, encoder_runtime.feature_dim, device=device)


def encode_chunks_forward_only_timed(
    encoder_runtime: EncoderRuntime,
    encoder_chunks: Sequence[Any],
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    outputs: List[torch.Tensor] = []
    total_ms = 0.0
    with torch.inference_mode():
        for chunk in encoder_chunks:
            device_chunk = recursive_to_device(chunk, device)
            maybe_sync(device)
            start = time.perf_counter()
            outputs.append(encoder_runtime.encode_chunk(device_chunk))
            maybe_sync(device)
            total_ms += (time.perf_counter() - start) * 1000.0
    if outputs:
        return torch.cat(outputs, dim=0), total_ms
    return torch.empty(0, encoder_runtime.feature_dim, device=device), 0.0


def profiler_activities(device: torch.device) -> List[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda" and hasattr(torch.profiler.ProfilerActivity, "CUDA"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def profile_flops(callable_fn: Callable[[], Any], device: torch.device) -> int:
    try:
        with torch.profiler.profile(activities=profiler_activities(device), with_flops=True, acc_events=True) as profiler:
            with torch.inference_mode():
                callable_fn()
            maybe_sync(device)
        return int(sum(event.flops or 0 for event in profiler.key_averages()))
    except Exception:
        return 0


def measure_streaming_latency(spec: BenchmarkSpec, encoder_runtime: EncoderRuntime, device: torch.device, warmup_iters: int, timing_iters: int) -> Tuple[float, float]:
    for index in range(warmup_iters):
        spec.prepare_raw_batch(spec.raw_batches[index % len(spec.raw_batches)], encoder_runtime)
    samples: List[float] = []
    for index in range(timing_iters):
        raw_batch = spec.raw_batches[index % len(spec.raw_batches)]
        maybe_sync(device)
        start = time.perf_counter()
        spec.prepare_raw_batch(raw_batch, encoder_runtime)
        maybe_sync(device)
        samples.append(((time.perf_counter() - start) * 1000.0) / float(frame_windows_per_batch(raw_batch)))
    return summarize_samples(samples)


def measure_encoder_latency(
    spec: BenchmarkSpec,
    encoder_runtime: EncoderRuntime,
    prepared_batches: Sequence[Dict[str, Any]],
    device: torch.device,
    warmup_iters: int,
    timing_iters: int,
) -> Tuple[float, float]:
    for index in range(warmup_iters):
        encode_chunks(encoder_runtime, prepared_batches[index % len(prepared_batches)]["encoder_chunks"], device)
    samples: List[float] = []
    for index in range(timing_iters):
        raw_batch = spec.raw_batches[index % len(spec.raw_batches)]
        _, total_ms = encode_chunks_forward_only_timed(
            encoder_runtime,
            prepared_batches[index % len(prepared_batches)]["encoder_chunks"],
            device,
        )
        samples.append(total_ms / float(frame_windows_per_batch(raw_batch)))
    return summarize_samples(samples)


def measure_model_latency(
    spec: BenchmarkSpec,
    model_inputs: Sequence[Dict[str, Any]],
    device: torch.device,
    warmup_iters: int,
    timing_iters: int,
) -> Tuple[str, float, float]:
    model = spec.model.to(device).eval()
    with torch.inference_mode():
        sample_output = spec.forward_model(model, model_inputs[0])
    output_shape = tensor_shape_to_str(sample_output.detach().cpu())
    for index in range(warmup_iters):
        with torch.inference_mode():
            spec.forward_model(model, model_inputs[index % len(model_inputs)])
        maybe_sync(device)
    samples: List[float] = []
    for index in range(timing_iters):
        raw_batch = spec.raw_batches[index % len(spec.raw_batches)]
        maybe_sync(device)
        start = time.perf_counter()
        with torch.inference_mode():
            spec.forward_model(model, model_inputs[index % len(model_inputs)])
        maybe_sync(device)
        samples.append(((time.perf_counter() - start) * 1000.0) / float(frame_windows_per_batch(raw_batch)))
    mean_ms, std_ms = summarize_samples(samples)
    return output_shape, mean_ms, std_ms


def measure_end_to_end_latency(
    spec: BenchmarkSpec,
    encoder_runtime: EncoderRuntime,
    device: torch.device,
    warmup_iters: int,
    timing_iters: int,
) -> Tuple[float, float]:
    def run_once(raw_batch: Dict[str, Any]) -> torch.Tensor:
        prepared = spec.prepare_raw_batch(raw_batch, encoder_runtime)
        encoded = encode_chunks(encoder_runtime, prepared["encoder_chunks"], device)
        model_inputs = spec.build_model_inputs(prepared, encoded, device)
        with torch.inference_mode():
            return spec.forward_model(spec.model, model_inputs)

    for index in range(warmup_iters):
        run_once(spec.raw_batches[index % len(spec.raw_batches)])
        maybe_sync(device)
    samples: List[float] = []
    for index in range(timing_iters):
        raw_batch = spec.raw_batches[index % len(spec.raw_batches)]
        maybe_sync(device)
        start = time.perf_counter()
        run_once(raw_batch)
        maybe_sync(device)
        samples.append(((time.perf_counter() - start) * 1000.0) / float(frame_windows_per_batch(raw_batch)))
    return summarize_samples(samples)


def compute_flops_per_window(spec: BenchmarkSpec, encoder_runtime: EncoderRuntime, prepared_batches: Sequence[Dict[str, Any]], device: torch.device) -> int:
    flops_samples: List[float] = []
    for raw_batch, prepared in zip(spec.raw_batches, prepared_batches):
        encoder_flops = 0
        for chunk in prepared["encoder_chunks"]:
            device_chunk = recursive_to_device(chunk, device)
            encoder_flops += profile_flops(lambda _chunk=device_chunk: encoder_runtime.encode_chunk(_chunk), device)
        encoded = encode_chunks(encoder_runtime, prepared["encoder_chunks"], device)
        model_inputs = spec.build_model_inputs(prepared, encoded, device)
        model_flops = profile_flops(lambda _inputs=model_inputs: spec.forward_model(spec.model, _inputs), device)
        flops_samples.append(float(encoder_flops + model_flops) / float(frame_windows_per_batch(raw_batch)))
    return int(round(mean(flops_samples))) if flops_samples else 0


def benchmark_spec(
    spec: BenchmarkSpec,
    encoder_runtime: EncoderRuntime,
    device: torch.device,
    warmup_iters: int,
    timing_iters: int,
) -> BenchmarkResult:
    spec.model = spec.model.to(device).eval()
    encoder_runtime.model = encoder_runtime.model.to(device).eval()

    prepared_batches = [spec.prepare_raw_batch(raw_batch, encoder_runtime) for raw_batch in spec.raw_batches]
    encoded_batches = [encode_chunks(encoder_runtime, prepared["encoder_chunks"], device) for prepared in prepared_batches]
    model_inputs = [spec.build_model_inputs(prepared, encoded, device) for prepared, encoded in zip(prepared_batches, encoded_batches)]

    streaming_latency_ms, streaming_latency_std_ms = measure_streaming_latency(spec, encoder_runtime, device, warmup_iters, timing_iters)
    encoder_latency_ms, encoder_latency_std_ms = measure_encoder_latency(spec, encoder_runtime, prepared_batches, device, warmup_iters, timing_iters)
    output_shape, model_latency_ms, model_latency_std_ms = measure_model_latency(spec, model_inputs, device, warmup_iters, timing_iters)
    end_to_end_latency_ms, end_to_end_latency_std_ms = measure_end_to_end_latency(spec, encoder_runtime, device, warmup_iters, timing_iters)
    flops_per_window = compute_flops_per_window(spec, encoder_runtime, prepared_batches, device)

    return BenchmarkResult(
        model_family=spec.model_family,
        model_name=spec.model_name,
        prompt_type=spec.prompt_type,
        encoder_name=spec.encoder_name,
        feature_source=spec.feature_source,
        benchmark_window_length=spec.benchmark_window_length,
        sequence_profile=spec.sequence_profile,
        input_description=spec.input_description,
        output_shape=output_shape,
        encoder_params=total_params(encoder_runtime.model),
        model_params=total_params(spec.model),
        flops_per_window=flops_per_window,
        streaming_latency_ms=streaming_latency_ms,
        streaming_latency_std_ms=streaming_latency_std_ms,
        encoder_latency_ms=encoder_latency_ms,
        encoder_latency_std_ms=encoder_latency_std_ms,
        model_latency_ms=model_latency_ms,
        model_latency_std_ms=model_latency_std_ms,
        end_to_end_latency_ms=end_to_end_latency_ms,
        end_to_end_latency_std_ms=end_to_end_latency_std_ms,
        device=str(device),
        notes=spec.notes,
    )


def build_sedmamba_spec(*, device: torch.device, sar_rarp_profile: RawImageProfile, sequence_lengths: Sequence[int], sequence_note: str) -> BenchmarkSpec:
    sed_module = import_sedmamba_module(device)
    model = sed_module.MultiStageModel(num_block=3, com_factor=64, dim=SED_DINOV2_FEATURE_DIM, num_classes=1)
    raw_batches = make_raw_image_batches(sequence_lengths=sequence_lengths, image_profile=sar_rarp_profile, seed_offset=10)
    return BenchmarkSpec(
        model_family="SEDMamba",
        model_name="MultiStageModel",
        prompt_type="n/a",
        encoder_key=SED_DINOV2_GIANT_ENCODER_KEY,
        encoder_name=SED_DINOV2_GIANT_ENCODER_NAME,
        feature_source="dinov2_giant_reg_imagenet1k_1layer",
        benchmark_window_length=WINDOW_LENGTH,
        sequence_profile=format_sequence_profile(raw_batches),
        input_description=(
            f"raw_frames=Lx{sar_rarp_profile.height}x{sar_rarp_profile.width}x3 uint8 -> "
            f"DINOv2 logits=Lx{SED_DINOV2_FEATURE_DIM} -> model=1x{SED_DINOV2_FEATURE_DIM}xL"
        ),
        notes=(
            f"SAR-RARP-like synthetic sequence lengths {format_sequence_profile(raw_batches)}; "
            f"{sequence_note}; raw image profile {sar_rarp_profile.source}; "
            "reported latency and FLOPs are normalized by ceil(sequence_length / 10)."
        ),
        model=model,
        raw_batches=raw_batches,
        prepare_raw_batch=prepare_image_only_batch,
        build_model_inputs=build_sedmamba_model_inputs,
        forward_model=lambda _model, batch: _model(batch["features"]),
    )


def build_cog_spec(*, device: torch.device, jigsaws_profile: RawImageProfile, num_batches: int) -> BenchmarkSpec:
    cog_module = import_cog_module()
    gest_prompt_file = register_temp_file(tempfile.NamedTemporaryFile(prefix="cog_prompt_", suffix=".pt", delete=False).name)
    args = SimpleNamespace(train=1, k=WINDOW_LENGTH, layers=10, stages=8, lambda_=0.15, dmodel=64, len_q=WINDOW_LENGTH)
    model = cog_module.COG(
        args,
        num_layers_Basic=11,
        num_layers_R=args.layers,
        num_R=3,
        num_f_maps=64,
        num_f_dim=RESNET_FEATURE_DIM,
        num_classes=2,
        causal_conv=True,
        d_model=args.dmodel,
        d_q=args.dmodel // 8,
        len_q=args.len_q,
        device=device,
        gest_prompt=gest_prompt_file,
    )
    raw_batches = make_raw_image_batches(sequence_lengths=[WINDOW_LENGTH] * num_batches, image_profile=jigsaws_profile, seed_offset=100)
    return BenchmarkSpec(
        model_family="CoG",
        model_name="COG",
        prompt_type="default_gesture_list",
        encoder_key=RESNET50_ENCODER_KEY,
        encoder_name=RESNET50_ENCODER_NAME,
        feature_source="resnet50",
        benchmark_window_length=WINDOW_LENGTH,
        sequence_profile=format_sequence_profile(raw_batches),
        input_description=f"raw_frames=10x{jigsaws_profile.height}x{jigsaws_profile.width}x3 uint8 -> ResNet50 features=1x10x{RESNET_FEATURE_DIM}",
        notes=(
            f"CoG latency is reported for a 10-frame segment; raw image profile {jigsaws_profile.source}; "
            "CLIP text prompt encoder remains excluded from encoder stats via synthetic prompt embeddings."
        ),
        model=model,
        raw_batches=raw_batches,
        prepare_raw_batch=prepare_image_only_batch,
        build_model_inputs=build_cog_model_inputs,
        forward_model=lambda _model, batch: _model(batch["features"])[0][0],
    )


def build_cog_gvr_only_spec(*, device: torch.device, jigsaws_profile: RawImageProfile, num_batches: int) -> BenchmarkSpec:
    """CoG with ONLY its Gestural-Visual Reasoning module (no multi-scale temporal reasoning).

    Built with exactly the same args, encoder, prompts, window length and raw image
    profile as build_cog_spec(), so the two rows differ only by the MSTR stack.
    """
    cog_module = import_cog_module()
    gest_prompt_file = register_temp_file(tempfile.NamedTemporaryFile(prefix="cog_prompt_", suffix=".pt", delete=False).name)
    args = SimpleNamespace(train=1, k=WINDOW_LENGTH, layers=10, stages=8, lambda_=0.15, dmodel=64, len_q=WINDOW_LENGTH)
    model = cog_module.COG_GVROnly(
        args,
        num_f_dim=RESNET_FEATURE_DIM,
        num_classes=2,
        d_model=args.dmodel,
        d_q=args.dmodel // 8,
        len_q=args.len_q,
        device=device,
        gest_prompt=gest_prompt_file,
    )
    raw_batches = make_raw_image_batches(sequence_lengths=[WINDOW_LENGTH] * num_batches, image_profile=jigsaws_profile, seed_offset=100)
    return BenchmarkSpec(
        model_family="CoG-GVR",
        model_name="COG_GVROnly",
        prompt_type="default_gesture_list",
        encoder_key=RESNET50_ENCODER_KEY,
        encoder_name=RESNET50_ENCODER_NAME,
        feature_source="resnet50",
        benchmark_window_length=WINDOW_LENGTH,
        sequence_profile=format_sequence_profile(raw_batches),
        input_description=f"raw_frames=10x{jigsaws_profile.height}x{jigsaws_profile.width}x3 uint8 -> ResNet50 features=1x10x{RESNET_FEATURE_DIM}",
        notes=(
            f"CoG GVR module only (multi-scale temporal reasoning removed); 10-frame segment; "
            f"raw image profile {jigsaws_profile.source}; CLIP text prompt encoder excluded from "
            "encoder stats via synthetic prompt embeddings."
        ),
        model=model,
        raw_batches=raw_batches,
        prepare_raw_batch=prepare_image_only_batch,
        build_model_inputs=build_cog_model_inputs,
        forward_model=lambda _model, batch: _model(batch["features"])[0][0],
    )


def build_gvr_pred_specs(*, encoder_key: str, encoder_name: str, feature_dim: int, jigsaws_profile: RawImageProfile, num_batches: int) -> List[BenchmarkSpec]:
    gvr_module = import_gvr_module()
    specs: List[BenchmarkSpec] = []
    raw_batches = make_raw_image_batches(sequence_lengths=[WINDOW_LENGTH] * num_batches, image_profile=jigsaws_profile, seed_offset=200)
    for prompt_index, (prompt_type, prompts) in enumerate(PROMPT_SETS.items()):
        model = gvr_module.GVRModulePredFeatures(
            gesture_prompts=prompts,
            d_text=GVR_TEXT_DIM,
            d_model=feature_dim,
            num_heads=1,
            segment_length=WINDOW_LENGTH,
            position=True,
            dropout=0.5,
            prompt_features=deterministic_prompt_features(len(prompts), GVR_TEXT_DIM, offset=100 * prompt_index),
            use_pooled_branch=True,
        )
        specs.append(
            BenchmarkSpec(
                model_family="GVR",
                model_name="GVRModulePredFeatures",
                prompt_type=prompt_type,
                encoder_key=encoder_key,
                encoder_name=encoder_name,
                feature_source=encoder_key,
                benchmark_window_length=WINDOW_LENGTH,
                sequence_profile=format_sequence_profile(raw_batches),
                input_description=f"raw_frames=10x{jigsaws_profile.height}x{jigsaws_profile.width}x3 uint8 -> features=1x10x{feature_dim}, masks=1x10 bool",
                notes=(
                    f"Prompt-conditioned prediction head with encoder {encoder_name}; "
                    f"raw image profile {jigsaws_profile.source}; synthetic CLIP text prompt features kept outside encoder stats."
                ),
                model=model,
                raw_batches=raw_batches,
                prepare_raw_batch=prepare_image_only_batch,
                build_model_inputs=build_gvr_pred_model_inputs,
                forward_model=lambda _model, batch: _model(batch["features"], masks=batch["masks"]),
            )
        )
    return specs


def build_gvr_kin_specs(*, encoder_key: str, encoder_name: str, feature_dim: int, jigsaws_profile: RawImageProfile, num_batches: int) -> List[BenchmarkSpec]:
    gvr_module = import_gvr_module()
    specs: List[BenchmarkSpec] = []
    raw_batches = make_raw_kin_batches(sequence_lengths=[WINDOW_LENGTH] * num_batches, image_profile=jigsaws_profile, seed_offset=300)
    for prompt_index, (prompt_type, prompts) in enumerate(PROMPT_SETS.items()):
        model = gvr_module.GVRModulePredKinFeatures(
            gesture_prompts=prompts,
            d_text=GVR_TEXT_DIM,
            d_model=feature_dim,
            num_heads=1,
            segment_length=WINDOW_LENGTH,
            position=True,
            dropout=0.3,
            prompt_features=deterministic_prompt_features(len(prompts), GVR_TEXT_DIM, offset=900 + 100 * prompt_index),
            use_pooled_branch=False,
        )
        specs.append(
            BenchmarkSpec(
                model_family="GVR",
                model_name="GVRModulePredKinFeatures",
                prompt_type=prompt_type,
                encoder_key=encoder_key,
                encoder_name=encoder_name,
                feature_source=f"{encoder_key}+kinematics",
                benchmark_window_length=WINDOW_LENGTH,
                sequence_profile=format_sequence_profile(raw_batches),
                input_description=(
                    f"raw_frames=10x{jigsaws_profile.height}x{jigsaws_profile.width}x3 uint8 + "
                    f"raw_kin=30x{KINEMATICS_DIM} float32 -> features=1x10x{feature_dim}, kine=1x10x{KINEMATICS_DIM}, masks=1x10 bool"
                ),
                notes=(
                    f"Visual+kinematics prompt-conditioned head with encoder {encoder_name}; "
                    "streaming latency includes JIGSAWS-style kinematics normalization and 3x downsampling."
                ),
                model=model,
                raw_batches=raw_batches,
                prepare_raw_batch=prepare_kin_batch,
                build_model_inputs=build_gvr_kin_model_inputs,
                forward_model=lambda _model, batch: _model(batch["features"], batch["kine"], masks=batch["masks"]),
            )
        )
    return specs


def build_gvr_context_specs(*, encoder_key: str, encoder_name: str, feature_dim: int, jigsaws_profile: RawImageProfile, num_batches: int) -> List[BenchmarkSpec]:
    gvr_module = import_gvr_module()
    model = gvr_module.GVRModuleContexPredFeatures(
        d_model_ges=GVR_TEXT_DIM,
        d_model=feature_dim,
        num_heads=1,
        segment_length=WINDOW_LENGTH,
        layer_norm1=True,
        layer_norm2=True,
        layer_norm3=True,
        position=False,
        dropout=0.1,
    )
    raw_batches = make_raw_image_batches(sequence_lengths=[WINDOW_LENGTH] * num_batches, image_profile=jigsaws_profile, seed_offset=400)
    return [
        BenchmarkSpec(
            model_family="GVR",
            model_name="GVRModuleContexPredFeatures",
            prompt_type="n/a",
            encoder_key=encoder_key,
            encoder_name=encoder_name,
            feature_source=encoder_key,
            benchmark_window_length=WINDOW_LENGTH,
            sequence_profile=format_sequence_profile(raw_batches),
            input_description=f"raw_frames=10x{jigsaws_profile.height}x{jigsaws_profile.width}x3 uint8 -> features=1x10x{feature_dim}",
            notes=f"Single-stream context model with encoder {encoder_name}; raw image profile {jigsaws_profile.source}.",
            model=model,
            raw_batches=raw_batches,
            prepare_raw_batch=prepare_image_only_batch,
            build_model_inputs=build_gvr_context_model_inputs,
            forward_model=lambda _model, batch: _model(batch["features"]),
        )
    ]


def build_gvr_dual_spec(*, jigsaws_profile: RawImageProfile, num_batches: int) -> BenchmarkSpec:
    gvr_module = import_gvr_module()
    inner = gvr_module.GVRModuleContexPredDualFeatures(
        base_dim=RESNET_FEATURE_DIM,
        ges_dim=RESNET_FEATURE_DIM,
        d_embed=RESNET_FEATURE_DIM,
        num_heads=1,
        segment_length=WINDOW_LENGTH,
        layer_norm1=False,
        layer_norm2=True,
        layer_norm3=False,
        position=True,
        dropout=0.3,
    )
    model = ActivityAwareDualStream(inner, build_resnet50_backbone())
    raw_batches = make_raw_dual_batches(
        sequence_lengths=[WINDOW_LENGTH] * num_batches,
        image_profile=jigsaws_profile,
        feature_dim=RESNET_FEATURE_DIM,
        seed_offset=500,
    )
    return BenchmarkSpec(
        model_family="GVR",
        model_name="GVRModuleContexPredDualFeatures",
        prompt_type="n/a",
        encoder_key=RESNET50_ENCODER_KEY,
        encoder_name=RESNET50_ENCODER_NAME,
        feature_source="resnet50 (spatial, encoder) + resnet50 (activity embeddings, in-model)",
        benchmark_window_length=WINDOW_LENGTH,
        sequence_profile=format_sequence_profile(raw_batches),
        input_description=f"raw_frames=10x{jigsaws_profile.height}x{jigsaws_profile.width}x3 uint8 -> base=1x10x{RESNET_FEATURE_DIM} (encoder) + same frames -> in-model activity ResNet50 -> ges=1x10x{RESNET_FEATURE_DIM}, masks=1x10 bool",
        notes=(
            "Activity-aware dual-stream row. The spatial-feature ResNet50 is the encoder; the "
            "second (activity-embedding) ResNet50 is owned by the model, so its params, FLOPs and "
            "latency count towards the model, not the encoder. Both encoders consume the same frames."
        ),
        model=model,
        raw_batches=raw_batches,
        prepare_raw_batch=prepare_dual_batch,
        build_model_inputs=build_gvr_dual_model_inputs,
        forward_model=lambda _model, batch: _model(batch["base_features"], batch["ges_pixels"], masks=batch["masks"]),
    )


def collect_model_specs(
    *,
    device: torch.device,
    num_batches: int,
    jigsaws_profile: RawImageProfile,
    sar_rarp_profile: RawImageProfile,
    sed_sequence_lengths: Sequence[int],
    sed_sequence_note: str,
    families: Optional[set] = None,
) -> List[BenchmarkSpec]:
    # `families` filters BEFORE construction: building a spec imports that model's
    # dependencies (SEDMamba needs mamba_ssm, which lives in a separate environment),
    # so filtering afterwards would still fail on unrelated models.
    def want(family: str) -> bool:
        return families is None or family.lower() in families

    specs: List[BenchmarkSpec] = []
    if want("CoG"):
        specs.append(build_cog_spec(device=device, jigsaws_profile=jigsaws_profile, num_batches=num_batches))
    if want("CoG-GVR"):
        specs.append(build_cog_gvr_only_spec(device=device, jigsaws_profile=jigsaws_profile, num_batches=num_batches))
    if want("GVR"):
        for enc_key, enc_name, feat_dim in (
            (RESNET50_ENCODER_KEY, RESNET50_ENCODER_NAME, RESNET_FEATURE_DIM),
            (DINOV2_LARGE_ENCODER_KEY, DINOV2_LARGE_ENCODER_NAME, DINOV2_LARGE_FEATURE_DIM),
            (CLIP_BASE_ENCODER_KEY, CLIP_BASE_ENCODER_NAME, CLIP_BASE_FEATURE_DIM),
        ):
            specs.extend(build_gvr_pred_specs(encoder_key=enc_key, encoder_name=enc_name, feature_dim=feat_dim, jigsaws_profile=jigsaws_profile, num_batches=num_batches))
            specs.extend(build_gvr_kin_specs(encoder_key=enc_key, encoder_name=enc_name, feature_dim=feat_dim, jigsaws_profile=jigsaws_profile, num_batches=num_batches))
            specs.extend(build_gvr_context_specs(encoder_key=enc_key, encoder_name=enc_name, feature_dim=feat_dim, jigsaws_profile=jigsaws_profile, num_batches=num_batches))
            if enc_key == RESNET50_ENCODER_KEY:
                specs.append(build_gvr_dual_spec(jigsaws_profile=jigsaws_profile, num_batches=num_batches))
    if want("SEDMamba"):
        specs.append(build_sedmamba_spec(device=device, sar_rarp_profile=sar_rarp_profile, sequence_lengths=sed_sequence_lengths, sequence_note=sed_sequence_note))
    return specs


def pretty_print_results(results: Sequence[BenchmarkResult]) -> None:
    headers = ("family", "model", "encoder", "prompt_type", "enc_params", "mdl_params", "flops", "stream_ms", "enc_ms", "mdl_ms", "e2e_ms")
    widths = [10, 28, 22, 24, 12, 12, 14, 10, 10, 10, 10]
    print(
        f"{headers[0]:<{widths[0]}} {headers[1]:<{widths[1]}} {headers[2]:<{widths[2]}} "
        f"{headers[3]:<{widths[3]}} {headers[4]:>{widths[4]}} {headers[5]:>{widths[5]}} "
        f"{headers[6]:>{widths[6]}} {headers[7]:>{widths[7]}} {headers[8]:>{widths[8]}} "
        f"{headers[9]:>{widths[9]}} {headers[10]:>{widths[10]}}"
    )
    print("-" * (sum(widths) + 10))
    for row in results:
        print(
            f"{row.model_family:<{widths[0]}} {row.model_name:<{widths[1]}} "
            f"{row.encoder_name:<{widths[2]}} {row.prompt_type:<{widths[3]}} "
            f"{row.encoder_params:>{widths[4]}} {row.model_params:>{widths[5]}} "
            f"{row.flops_per_window:>{widths[6]}} {row.streaming_latency_ms:>{widths[7]}.3f} "
            f"{row.encoder_latency_ms:>{widths[8]}.3f} {row.model_latency_ms:>{widths[9]}.3f} "
            f"{row.end_to_end_latency_ms:>{widths[10]}.3f}"
        )


def release_encoder_runtime(runtime: Optional[EncoderRuntime], device: torch.device) -> None:
    if runtime is not None:
        runtime.model = runtime.model.to("cpu")
        empty_cuda_cache(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark encoder params, downstream params, FLOPs, and per-window latency for SEDMamba, CoG, and GVR feature models."
    )
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated model_family values to profile (e.g. 'CoG'). "
                             "Default: every spec.")
    parser.add_argument("--latency_stat", choices=["median", "mean"], default="median",
                        help="Central estimate for latency samples. median (default) is robust to "
                             "transient stalls on a shared node; mean reproduces the older behaviour.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dummy_batches", type=int, default=DEFAULT_DUMMY_BATCHES)
    parser.add_argument("--warmup_iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--timing_iters", type=int, default=DEFAULT_TIMING_ITERS)
    parser.add_argument("--encoder_chunk_size", type=int, default=DEFAULT_ENCODER_CHUNK_SIZE)
    parser.add_argument("--data_root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output_csv", type=Path, default=REPO_ROOT / "profile_reports" / "window_model_complexity_len10.csv")
    parser.add_argument("--output_json", type=Path, default=REPO_ROOT / "profile_reports" / "window_model_complexity_len10.json")
    args = parser.parse_args()

    set_seed(42)
    globals()["LATENCY_STAT"] = args.latency_stat
    device = torch.device(args.device)
    ensure_parent_dir(args.output_csv)
    ensure_parent_dir(args.output_json)
    ensure_transformers_available()

    sar_rarp_root = resolve_sar_rarp_root(args.data_root)
    jigsaws_profile = discover_image_profile(
        dataset_name="JIGSAWS",
        patterns=[str(args.data_root / "vid_frames" / "*" / "*" / "frame_*.png")],
    )
    sar_rarp_profile = discover_image_profile(
        dataset_name="SAR-RARP50",
        patterns=[
            str(sar_rarp_root / "training_set" / "*" / "images" / "*.png"),
            str(sar_rarp_root / "testing_set" / "*" / "images" / "*.png"),
        ],
    )
    sed_sequence_lengths, sed_sequence_note = discover_sar_rarp_sequence_lengths(args.data_root, max(1, args.dummy_batches))

    specs = collect_model_specs(
        device=device,
        num_batches=max(1, args.dummy_batches),
        jigsaws_profile=jigsaws_profile,
        sar_rarp_profile=sar_rarp_profile,
        sed_sequence_lengths=sed_sequence_lengths,
        sed_sequence_note=sed_sequence_note,
        families=({s.strip().lower() for s in args.only.split(',') if s.strip()} if args.only else None),
    )
    if args.only:
        wanted = {s.strip().lower() for s in args.only.split(',') if s.strip()}
        specs = [s for s in specs if s.model_family.lower() in wanted]
        print(f"[filter] profiling {len(specs)} spec(s): "
              + ', '.join(f'{s.model_family}/{s.prompt_type}' for s in specs))


    print(f"[INFO] Benchmark device: {device}")
    print(f"[INFO] Benchmark window length: {WINDOW_LENGTH}")
    print(f"[INFO] Dummy batches per fixed-window model: {args.dummy_batches}")
    print(f"[INFO] Warmup iterations: {args.warmup_iters}")
    print(f"[INFO] Timing iterations: {args.timing_iters}")
    print(f"[INFO] Encoder chunk size: {args.encoder_chunk_size}")
    print(f"[INFO] JIGSAWS raw image profile: {jigsaws_profile.height}x{jigsaws_profile.width} ({jigsaws_profile.source})")
    print(f"[INFO] SAR-RARP50 raw image profile: {sar_rarp_profile.height}x{sar_rarp_profile.width} ({sar_rarp_profile.source})")
    print(f"[INFO] SEDMamba representative sequence lengths: {sed_sequence_lengths} ({sed_sequence_note})")

    results: List[BenchmarkResult] = []
    current_encoder_key = None
    current_encoder_runtime = None  # type: Optional[EncoderRuntime]

    for spec in specs:
        if spec.encoder_key != current_encoder_key:
            release_encoder_runtime(current_encoder_runtime, device)
            current_encoder_runtime = build_encoder_runtime(spec.encoder_key, device, args.encoder_chunk_size)
            current_encoder_key = spec.encoder_key
            print(f"[INFO] Loaded encoder runtime: {current_encoder_runtime.display_name}")

        assert current_encoder_runtime is not None
        print(f"[INFO] Benchmarking {spec.model_family} / {spec.model_name} / {spec.encoder_name} / {spec.prompt_type}")
        results.append(
            benchmark_spec(
                spec,
                current_encoder_runtime,
                device,
                warmup_iters=args.warmup_iters,
                timing_iters=args.timing_iters,
            )
        )
        spec.model = spec.model.to("cpu")
        empty_cuda_cache(device)

    release_encoder_runtime(current_encoder_runtime, device)

    csv_rows = [asdict(row) for row in results]
    with args.output_csv.open("w", encoding="utf-8") as handle:
        headers = list(csv_rows[0].keys())
        handle.write(",".join(headers) + "\n")
        for row in csv_rows:
            values: List[str] = []
            for header in headers:
                value = row[header]
                if isinstance(value, str):
                    value = value.replace('"', "'")
                    values.append(f"\"{value}\"")
                else:
                    values.append(str(value))
            handle.write(",".join(values) + "\n")

    payload = {
        "device": str(device),
        "benchmark_window_length": WINDOW_LENGTH,
        "dummy_batches": args.dummy_batches,
        "warmup_iters": args.warmup_iters,
        "timing_iters": args.timing_iters,
        "encoder_chunk_size": args.encoder_chunk_size,
        "jigsaws_raw_image_profile": asdict(jigsaws_profile),
        "sar_rarp_raw_image_profile": asdict(sar_rarp_profile),
        "sedmamba_sequence_lengths": list(sed_sequence_lengths),
        "sedmamba_sequence_note": sed_sequence_note,
        "results": csv_rows,
    }
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    pretty_print_results(results)
    print(f"\n[INFO] Wrote CSV:  {args.output_csv}")
    print(f"[INFO] Wrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
