#!/usr/bin/env python3
"""Compatibility wrapper for the encoder-aware benchmark implementation."""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from benchmark_window_model_complexity_impl import main


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Compatibility wrapper for the encoder-aware benchmark implementation."""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from benchmark_window_model_complexity_impl import main


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Compatibility wrapper for the encoder-aware benchmark implementation."""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from benchmark_window_model_complexity_impl import main


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Compatibility wrapper for the encoder-aware benchmark implementation."""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from benchmark_window_model_complexity_impl import main as _benchmark_main

if __name__ == "__main__":
    raise SystemExit(_benchmark_main())

import tempfile
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
import tempfile
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAIN_DIR = REPO_ROOT / "Chain-of-Gesture"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def summarize_samples(samples: Sequence[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    mean_value = mean(samples)
    std_value = pstdev(samples) if len(samples) > 1 else 0.0
    return float(mean_value), float(std_value)


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
                return RawImageProfile(
                    dataset_name=dataset_name,
                    height=int(height),
                    width=int(width),
                    source=f"detected from {match}",
                )
            except OSError:
                continue
    return RawImageProfile(
        dataset_name=dataset_name,
        height=fallback_height,
        width=fallback_width,
        source=f"fallback {fallback_height}x{fallback_width}",
    )


def rounded_window_length(value: int) -> int:
    rounded = int(round(float(value) / float(WINDOW_LENGTH)) * WINDOW_LENGTH)
    return max(WINDOW_LENGTH, rounded)


def fallback_sedmamba_lengths(num_lengths: int) -> List[int]:
    base = list(DEFAULT_SEDMAMBA_LENGTHS)
    if num_lengths <= len(base):
        return base[:num_lengths]
    while len(base) < num_lengths:
        base.append(base[-1] + WINDOW_LENGTH * 6)
    return base[:num_lengths]


def discover_sar_rarp_sequence_lengths(data_root: Path, num_lengths: int) -> tuple[List[int], str]:
    sar_root = resolve_sar_rarp_root(data_root)
    pkl_roots = [sar_root / "train_emb_DINOv2", sar_root / "test_emb_DINOv2"]
    lengths: List[int] = []
    for pkl_root in pkl_roots:
        if not pkl_root.exists():
            continue
        for pkl_path in sorted(pkl_root.glob("*.pkl")):
            try:
                with pkl_path.open("rb") as handle:
                    payload = pickle.load(handle)
                labels = payload.get("error_GT")
                if labels is None:
                    continue
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
        return chosen[:num_lengths], (
            f"derived from {len(lengths)} SAR-RARP50 label sequences under {sar_root}"
        )
    fallback = fallback_sedmamba_lengths(num_lengths)
    return fallback, f"fallback representative SAR-RARP50-like lengths {fallback}"


def frame_windows_per_batch(raw_batch: Dict[str, Any]) -> int:
    seq_len = int(raw_batch["sequence_length"])
    return max(1, math.ceil(seq_len / WINDOW_LENGTH))


def format_sequence_profile(raw_batches: Sequence[Dict[str, Any]]) -> str:
    lengths = [str(int(batch["sequence_length"])) for batch in raw_batches]
    unique_lengths: List[str] = []
    for length in lengths:
        if length not in unique_lengths:
            unique_lengths.append(length)
    return "|".join(unique_lengths)


def make_frame_bank(
    *,
    sequence_length: int,
    image_height: int,
    image_width: int,
    seed: int,
    max_bank_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    bank_size = max(1, min(sequence_length, max_bank_size))
    rng = np.random.default_rng(seed)
    frame_bank = rng.integers(
        0,
        256,
        size=(bank_size, image_height, image_width, 3),
        dtype=np.uint8,
    )
    frame_indices = rng.integers(0, bank_size, size=(sequence_length,), dtype=np.int64)
    return frame_bank, frame_indices


def make_raw_image_batches(
    *,
    sequence_lengths: Sequence[int],
    image_profile: RawImageProfile,
    seed_offset: int,
) -> List[Dict[str, Any]]:
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


def make_raw_kin_batches(
    *,
    sequence_lengths: Sequence[int],
    image_profile: RawImageProfile,
    seed_offset: int,
) -> List[Dict[str, Any]]:
    batches = make_raw_image_batches(
        sequence_lengths=sequence_lengths,
        image_profile=image_profile,
        seed_offset=seed_offset,
    )
    for index, batch in enumerate(batches):
        seq_len = int(batch["sequence_length"])
        rng = np.random.default_rng(3000 + seed_offset + index)
        batch["kine_raw"] = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(seq_len * KIN_DOWNSAMPLE, KINEMATICS_DIM),
        ).astype(np.float32)
    return batches


def make_raw_dual_batches(
    *,
    sequence_lengths: Sequence[int],
    image_profile: RawImageProfile,
    feature_dim: int,
    seed_offset: int,
) -> List[Dict[str, Any]]:
    batches = make_raw_image_batches(
        sequence_lengths=sequence_lengths,
        image_profile=image_profile,
        seed_offset=seed_offset,
    )
    for index, batch in enumerate(batches):
        seq_len = int(batch["sequence_length"])
        batch["gesture_feature_stream"] = deterministic_feature_matrix(
            seq_len,
            feature_dim,
            offset=5000 + seed_offset + index,
        )
    return batches


def materialize_frame_arrays(raw_batch: Dict[str, Any]) -> np.ndarray:
    frame_bank = raw_batch["frame_bank_uint8"]
    frame_indices = raw_batch["frame_indices"]
    return frame_bank[frame_indices].copy()


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


def import_gvr_module():
    return load_module_from_file(
        "benchmark_model_gvr_features",
        REPO_ROOT / "model_gvr_features.py",
    )


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


def import_sedmamba_module(device: torch.device):
    _install_mamba_package_stubs()
    selective_scan = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
    mamba_simple = importlib.import_module("mamba_ssm.modules.mamba_simple")
    if device.type == "cpu":
        mamba_simple.causal_conv1d_fn = None
        mamba_simple.selective_scan_fn = selective_scan.selective_scan_ref
    sys.modules["mamba_ssm"].Mamba = mamba_simple.Mamba
    return load_module_from_file(
        "benchmark_sedmamba_baseline",
        REPO_ROOT / "SEDMamba" / "baseline" / "SEDMamba.py",
    )


def install_fake_scipy() -> None:
    fake_scipy = types.ModuleType("scipy")
    fake_interpolate = types.ModuleType("scipy.interpolate")
    fake_scipy.interpolate = fake_interpolate
    sys.modules["scipy"] = fake_scipy
    sys.modules["scipy.interpolate"] = fake_interpolate


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
    try:
        import scipy  # noqa: F401
    except ImportError:
        install_fake_scipy()
    if str(CHAIN_DIR) not in sys.path:
        sys.path.insert(0, str(CHAIN_DIR))
    module = load_module_from_file(
        "benchmark_chain_of_gesture_models",
        CHAIN_DIR / "models.py",
    )
    module.clip = FakeClipModule()
    return module


def build_resnet50_runtime(device: torch.device, chunk_size: int) -> EncoderRuntime:
    weight_enum = getattr(models, "ResNet50_Weights", None)
    weights = weight_enum.DEFAULT if weight_enum is not None else None
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model = freeze_module(model.to(device))
    transform = get_resnet_transform(image_size=224, normalize=True)

    def preprocess_chunks(raw_batch: Dict[str, Any]) -> List[torch.Tensor]:
        images = materialize_pil_images(raw_batch)
        chunks: List[torch.Tensor] = []
        for image_chunk in split_list_chunks(images, chunk_size):
            chunk_tensor = torch.stack([transform(image) for image in image_chunk]).to(torch.float32)
            chunks.append(chunk_tensor)
        return chunks

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


def build_clip_base_runtime(device: torch.device, chunk_size: int) -> EncoderRuntime:
    from transformers import CLIPProcessor, CLIPVisionModel

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = freeze_module(
        CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    )

    def preprocess_chunks(raw_batch: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        images = materialize_pil_images(raw_batch)
        chunks: List[Dict[str, torch.Tensor]] = []
        for image_chunk in split_list_chunks(images, chunk_size):
            inputs = processor(images=list(image_chunk), return_tensors="pt")
            chunks.append({"pixel_values": inputs["pixel_values"].to(torch.float32)})
        return chunks

    def encode_chunk(chunk: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = model(**chunk)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0, :]
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

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model = freeze_module(AutoModel.from_pretrained("facebook/dinov2-large").to(device))

    def preprocess_chunks(raw_batch: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        images = materialize_pil_images(raw_batch)
        chunks: List[Dict[str, torch.Tensor]] = []
        for image_chunk in split_list_chunks(images, chunk_size):
            inputs = processor(images=list(image_chunk), return_tensors="pt")
            chunks.append({"pixel_values": inputs["pixel_values"].to(torch.float32)})
        return chunks

    def encode_chunk(chunk: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = model(**chunk)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0, :]
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
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = freeze_module(AutoModelForImageClassification.from_pretrained(model_id).to(device))

    def preprocess_chunks(raw_batch: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        images = materialize_pil_images(raw_batch)
        chunks: List[Dict[str, torch.Tensor]] = []
        for image_chunk in split_list_chunks(images, chunk_size):
            inputs = processor(images=list(image_chunk), return_tensors="pt")
            chunks.append({"pixel_values": inputs["pixel_values"].to(torch.float32)})
        return chunks

    def encode_chunk(chunk: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = model(**chunk)
        return outputs.logits.to(torch.float32)

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


def prepare_image_only_batch(raw_batch: Dict[str, Any], encoder_runtime: EncoderRuntime) -> Dict[str, Any]:
    seq_len = int(raw_batch["sequence_length"])
    return {
        "sequence_length": seq_len,
        "encoder_chunks": encoder_runtime.preprocess_chunks(raw_batch),
        "masks": torch.ones(seq_len, dtype=torch.bool),
    }


def prepare_kin_batch(raw_batch: Dict[str, Any], encoder_runtime: EncoderRuntime) -> Dict[str, Any]:
    prepared = prepare_image_only_batch(raw_batch, encoder_runtime)
    seq_len = int(raw_batch["sequence_length"])
    prepared["kine"] = prepare_jigsaws_kinematics(raw_batch["kine_raw"], target_length=seq_len)
    return prepared


def prepare_dual_batch(raw_batch: Dict[str, Any], encoder_runtime: EncoderRuntime) -> Dict[str, Any]:
    prepared = prepare_image_only_batch(raw_batch, encoder_runtime)
    prepared["ges_features"] = raw_batch["gesture_feature_stream"].clone().to(torch.float32)
    return prepared


def build_sedmamba_model_inputs(
    prepared_batch: Dict[str, Any],
    encoded_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    del prepared_batch, device
    return {"features": encoded_features.transpose(0, 1).unsqueeze(0)}


def build_cog_model_inputs(
    prepared_batch: Dict[str, Any],
    encoded_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    del prepared_batch, device
    return {"features": encoded_features.unsqueeze(0)}


def build_gvr_pred_model_inputs(
    prepared_batch: Dict[str, Any],
    encoded_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    return {
        "features": encoded_features.unsqueeze(0),
        "masks": prepared_batch["masks"].unsqueeze(0).to(device),
    }


def build_gvr_kin_model_inputs(
    prepared_batch: Dict[str, Any],
    encoded_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    return {
        "features": encoded_features.unsqueeze(0),
        "kine": prepared_batch["kine"].unsqueeze(0).to(device),
        "masks": prepared_batch["masks"].unsqueeze(0).to(device),
    }


def build_gvr_context_model_inputs(
    prepared_batch: Dict[str, Any],
    encoded_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    del prepared_batch, device
    return {"features": encoded_features.unsqueeze(0)}


def build_gvr_dual_model_inputs(
    prepared_batch: Dict[str, Any],
    encoded_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    return {
        "base_features": encoded_features.unsqueeze(0),
        "ges_features": prepared_batch["ges_features"].unsqueeze(0).to(device),
        "masks": prepared_batch["masks"].unsqueeze(0).to(device),
    }


def encode_chunks(
    encoder_runtime: EncoderRuntime,
    encoder_chunks: Sequence[Any],
    device: torch.device,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    with torch.inference_mode():
        for chunk in encoder_chunks:
            device_chunk = recursive_to_device(chunk, device)
            outputs.append(encoder_runtime.encode_chunk(device_chunk))
    if outputs:
        return torch.cat(outputs, dim=0)
    return torch.empty(0, encoder_runtime.feature_dim, device=device)


def encode_chunks_forward_only_timed(
    encoder_runtime: EncoderRuntime,
    encoder_chunks: Sequence[Any],
    device: torch.device,
) -> tuple[torch.Tensor, float]:
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
        with torch.profiler.profile(
            activities=profiler_activities(device),
            with_flops=True,
            acc_events=True,
        ) as profiler:
            with torch.inference_mode():
                callable_fn()
            maybe_sync(device)
        return int(sum(event.flops or 0 for event in profiler.key_averages()))
    except Exception:
        return 0


def measure_streaming_latency(
    spec: BenchmarkSpec,
    encoder_runtime: EncoderRuntime,
    device: torch.device,
    warmup_iters: int,
    timing_iters: int,
) -> tuple[float, float]:
    for index in range(warmup_iters):
        raw_batch = spec.raw_batches[index % len(spec.raw_batches)]
        spec.prepare_raw_batch(raw_batch, encoder_runtime)
    samples_ms: List[float] = []
    for index in range(timing_iters):
        raw_batch = spec.raw_batches[index % len(spec.raw_batches)]
        maybe_sync(device)
        start = time.perf_counter()
        spec.prepare_raw_batch(raw_batch, encoder_runtime)
        maybe_sync(device)
        samples_ms.append(
            ((time.perf_counter() - start) * 1000.0) / float(frame_windows_per_batch(raw_batch))
        )
    return summarize_samples(samples_ms)


def measure_encoder_latency(
    spec: BenchmarkSpec,
    encoder_runtime: EncoderRuntime,
    prepared_batches: Sequence[Dict[str, Any]],
    device: torch.device,
    warmup_iters: int,
    timing_iters: int,
) -> tuple[float, float]:
    for index in range(warmup_iters):
        prepared = prepared_batches[index % len(prepared_batches)]
        encode_chunks(encoder_runtime, prepared["encoder_chunks"], device)
    samples_ms: List[float] = []
    for index in range(timing_iters):
        raw_batch = spec.raw_batches[index % len(spec.raw_batches)]
        prepared = prepared_batches[index % len(prepared_batches)]
        _, total_ms = encode_chunks_forward_only_timed(
            encoder_runtime,
            prepared["encoder_chunks"],
            device,
        )
        samples_ms.append(total_ms / float(frame_windows_per_batch(raw_batch)))
    return summarize_samples(samples_ms)


def measure_model_latency(
    spec: BenchmarkSpec,
    model_inputs: Sequence[Dict[str, Any]],
    device: torch.device,
    warmup_iters: int,
    timing_iters: int,
) -> tuple[str, float, float]:
    model = spec.model.to(device).eval()
    with torch.inference_mode():
        sample_output = spec.forward_model(model, model_inputs[0])
    output_shape = tensor_shape_to_str(sample_output.detach().cpu())
    for index in range(warmup_iters):
        with torch.inference_mode():
            spec.forward_model(model, model_inputs[index % len(model_inputs)])
        maybe_sync(device)
    samples_ms: List[float] = []
    for index in range(timing_iters):
        raw_batch = spec.raw_batches[index % len(spec.raw_batches)]
        maybe_sync(device)
        start = time.perf_counter()
        with torch.inference_mode():
            spec.forward_model(model, model_inputs[index % len(model_inputs)])
        maybe_sync(device)
        samples_ms.append(
            ((time.perf_counter() - start) * 1000.0) / float(frame_windows_per_batch(raw_batch))
        )
    latency_mean, latency_std = summarize_samples(samples_ms)
    return output_shape, latency_mean, latency_std


def measure_end_to_end_latency(
    spec: BenchmarkSpec,
    encoder_runtime: EncoderRuntime,
    device: torch.device,
    warmup_iters: int,
    timing_iters: int,
) -> tuple[float, float]:
    def run_once(raw_batch: Dict[str, Any]) -> torch.Tensor:
        prepared = spec.prepare_raw_batch(raw_batch, encoder_runtime)
        encoded = encode_chunks(encoder_runtime, prepared["encoder_chunks"], device)
        model_inputs = spec.build_model_inputs(prepared, encoded, device)
        with torch.inference_mode():
            return spec.forward_model(spec.model, model_inputs)

    for index in range(warmup_iters):
        run_once(spec.raw_batches[index % len(spec.raw_batches)])
        maybe_sync(device)
    samples_ms: List[float] = []
    for index in range(timing_iters):
        raw_batch = spec.raw_batches[index % len(spec.raw_batches)]
        maybe_sync(device)
        start = time.perf_counter()
        run_once(raw_batch)
        maybe_sync(device)
        samples_ms.append(
            ((time.perf_counter() - start) * 1000.0) / float(frame_windows_per_batch(raw_batch))
        )
    return summarize_samples(samples_ms)


def compute_flops_per_window(
    spec: BenchmarkSpec,
    encoder_runtime: EncoderRuntime,
    prepared_batches: Sequence[Dict[str, Any]],
    device: torch.device,
) -> int:
    flops_samples: List[float] = []
    for raw_batch, prepared in zip(spec.raw_batches, prepared_batches):
        encoder_flops = 0
        for chunk in prepared["encoder_chunks"]:
            device_chunk = recursive_to_device(chunk, device)
            encoder_flops += profile_flops(
                lambda _chunk=device_chunk: encoder_runtime.encode_chunk(_chunk),
                device,
            )
        encoded = encode_chunks(encoder_runtime, prepared["encoder_chunks"], device)
        model_inputs = spec.build_model_inputs(prepared, encoded, device)
        model_flops = profile_flops(
            lambda _inputs=model_inputs: spec.forward_model(spec.model, _inputs),
            device,
        )
        flops_samples.append(
            float(encoder_flops + model_flops) / float(frame_windows_per_batch(raw_batch))
        )
    if flops_samples:
        return int(round(mean(flops_samples)))
    return 0


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
    model_inputs = [
        spec.build_model_inputs(prepared, encoded, device)
        for prepared, encoded in zip(prepared_batches, encoded_batches)
    ]

    streaming_latency_ms, streaming_latency_std_ms = measure_streaming_latency(
        spec,
        encoder_runtime,
        device,
        warmup_iters,
        timing_iters,
    )
    encoder_latency_ms, encoder_latency_std_ms = measure_encoder_latency(
        spec,
        encoder_runtime,
        prepared_batches,
        device,
        warmup_iters,
        timing_iters,
    )
    output_shape, model_latency_ms, model_latency_std_ms = measure_model_latency(
        spec,
        model_inputs,
        device,
        warmup_iters,
        timing_iters,
    )
    end_to_end_latency_ms, end_to_end_latency_std_ms = measure_end_to_end_latency(
        spec,
        encoder_runtime,
        device,
        warmup_iters,
        timing_iters,
    )
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


def build_sedmamba_spec(
    *,
    device: torch.device,
    sar_rarp_profile: RawImageProfile,
    sequence_lengths: Sequence[int],
    sequence_note: str,
) -> BenchmarkSpec:
    sed_module = import_sedmamba_module(device)
    model = sed_module.MultiStageModel(
        num_block=3,
        com_factor=64,
        dim=SED_DINOV2_FEATURE_DIM,
        num_classes=1,
    )
    raw_batches = make_raw_image_batches(
        sequence_lengths=sequence_lengths,
        image_profile=sar_rarp_profile,
        seed_offset=10,
    )
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


def build_cog_spec(
    *,
    device: torch.device,
    jigsaws_profile: RawImageProfile,
    num_batches: int,
) -> BenchmarkSpec:
    cog_module = import_cog_module()
    gest_prompt_file = register_temp_file(
        tempfile.NamedTemporaryFile(prefix="cog_prompt_", suffix=".pt", delete=False).name
    )
    args = SimpleNamespace(
        train=1,
        k=WINDOW_LENGTH,
        layers=10,
        stages=8,
        lambda_=0.15,
        dmodel=64,
        len_q=WINDOW_LENGTH,
    )
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
    raw_batches = make_raw_image_batches(
        sequence_lengths=[WINDOW_LENGTH] * num_batches,
        image_profile=jigsaws_profile,
        seed_offset=100,
    )
    return BenchmarkSpec(
        model_family="CoG",
        model_name="COG",
        prompt_type="default_gesture_list",
        encoder_key=RESNET50_ENCODER_KEY,
        encoder_name=RESNET50_ENCODER_NAME,
        feature_source="resnet50",
        benchmark_window_length=WINDOW_LENGTH,
        sequence_profile=format_sequence_profile(raw_batches),
        input_description=(
            f"raw_frames=10x{jigsaws_profile.height}x{jigsaws_profile.width}x3 uint8 -> "
            f"ResNet50 features=1x10x{RESNET_FEATURE_DIM}"
        ),
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


def build_gvr_pred_specs(
    *,
    encoder_key: str,
    encoder_name: str,
    feature_dim: int,
    jigsaws_profile: RawImageProfile,
    num_batches: int,
) -> List[BenchmarkSpec]:
    gvr_module = import_gvr_module()
    specs: List[BenchmarkSpec] = []
    raw_batches = make_raw_image_batches(
        sequence_lengths=[WINDOW_LENGTH] * num_batches,
        image_profile=jigsaws_profile,
        seed_offset=200,
    )
    for prompt_index, (prompt_type, prompts) in enumerate(PROMPT_SETS.items()):
        prompt_features = deterministic_prompt_features(
            len(prompts),
            GVR_TEXT_DIM,
            offset=100 * prompt_index,
        )
        model = gvr_module.GVRModulePredFeatures(
            gesture_prompts=prompts,
            d_text=GVR_TEXT_DIM,
            d_model=feature_dim,
            num_heads=1,
            segment_length=WINDOW_LENGTH,
            position=True,
            dropout=0.5,
            prompt_features=prompt_features,
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
                input_description=(
                    f"raw_frames=10x{jigsaws_profile.height}x{jigsaws_profile.width}x3 uint8 -> "
                    f"features=1x10x{feature_dim}, masks=1x10 bool"
                ),
                notes=(
                    f"Prompt-conditioned prediction head with encoder {encoder_name}; "
                    f"raw image profile {jigsaws_profile.source}; synthetic CLIP text prompt features kept outside encoder stats."
                ),
                model=model,
                raw_batches=raw_batches,
                prepare_raw_batch=prepare_image_only_batch,
                build_model_inputs=build_gvr_pred_model_inputs,
                forward_model=lambda _model, batch: _model(
                    batch["features"],
                    masks=batch["masks"],
                ),
            )
        )
    return specs


def build_gvr_kin_specs(
    *,
    encoder_key: str,
    encoder_name: str,
    feature_dim: int,
    jigsaws_profile: RawImageProfile,
    num_batches: int,
) -> List[BenchmarkSpec]:
    gvr_module = import_gvr_module()
    specs: List[BenchmarkSpec] = []
    raw_batches = make_raw_kin_batches(
        sequence_lengths=[WINDOW_LENGTH] * num_batches,
        image_profile=jigsaws_profile,
        seed_offset=300,
    )
    for prompt_index, (prompt_type, prompts) in enumerate(PROMPT_SETS.items()):
        prompt_features = deterministic_prompt_features(
            len(prompts),
            GVR_TEXT_DIM,
            offset=900 + 100 * prompt_index,
        )
        model = gvr_module.GVRModulePredKinFeatures(
            gesture_prompts=prompts,
            d_text=GVR_TEXT_DIM,
            d_model=feature_dim,
            num_heads=1,
            segment_length=WINDOW_LENGTH,
            position=True,
            dropout=0.3,
            prompt_features=prompt_features,
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
                    f"raw_kin=30x{KINEMATICS_DIM} float32 -> features=1x10x{feature_dim}, "
                    f"kine=1x10x{KINEMATICS_DIM}, masks=1x10 bool"
                ),
                notes=(
                    f"Visual+kinematics prompt-conditioned head with encoder {encoder_name}; "
                    "streaming latency includes JIGSAWS-style kinematics normalization and 3x downsampling."
                ),
                model=model,
                raw_batches=raw_batches,
                prepare_raw_batch=prepare_kin_batch,
                build_model_inputs=build_gvr_kin_model_inputs,
                forward_model=lambda _model, batch: _model(
                    batch["features"],
                    batch["kine"],
                    masks=batch["masks"],
                ),
            )
        )
    return specs


def build_gvr_context_specs(
    *,
    encoder_key: str,
    encoder_name: str,
    feature_dim: int,
    jigsaws_profile: RawImageProfile,
    num_batches: int,
) -> List[BenchmarkSpec]:
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
    raw_batches = make_raw_image_batches(
        sequence_lengths=[WINDOW_LENGTH] * num_batches,
        image_profile=jigsaws_profile,
        seed_offset=400,
    )
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
            input_description=(
                f"raw_frames=10x{jigsaws_profile.height}x{jigsaws_profile.width}x3 uint8 -> "
                f"features=1x10x{feature_dim}"
            ),
            notes=(
                f"Single-stream context model with encoder {encoder_name}; "
                f"raw image profile {jigsaws_profile.source}."
            ),
            model=model,
            raw_batches=raw_batches,
            prepare_raw_batch=prepare_image_only_batch,
            build_model_inputs=build_gvr_context_model_inputs,
            forward_model=lambda _model, batch: _model(batch["features"]),
        )
    ]


def build_gvr_dual_spec(
    *,
    jigsaws_profile: RawImageProfile,
    num_batches: int,
) -> BenchmarkSpec:
    gvr_module = import_gvr_module()
    model = gvr_module.GVRModuleContexPredDualFeatures(
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
        feature_source="resnet50+synthetic_gesture_stream",
        benchmark_window_length=WINDOW_LENGTH,
        sequence_profile=format_sequence_profile(raw_batches),
        input_description=(
            f"raw_frames=10x{jigsaws_profile.height}x{jigsaws_profile.width}x3 uint8 -> "
            f"base=1x10x{RESNET_FEATURE_DIM}, ges=1x10x{RESNET_FEATURE_DIM}, masks=1x10 bool"
        ),
        notes=(
            "Dual-stream context row keeps the original paired-stream structure. "
            "Encoder stats/latency cover the base ResNet50 stream; the gesture stream remains a synthetic paired feature input."
        ),
        model=model,
        raw_batches=raw_batches,
        prepare_raw_batch=prepare_dual_batch,
        build_model_inputs=build_gvr_dual_model_inputs,
        forward_model=lambda _model, batch: _model(
            batch["base_features"],
            batch["ges_features"],
            masks=batch["masks"],
        ),
    )


def collect_model_specs(
    *,
    device: torch.device,
    num_batches: int,
    jigsaws_profile: RawImageProfile,
    sar_rarp_profile: RawImageProfile,
    sed_sequence_lengths: Sequence[int],
    sed_sequence_note: str,
) -> List[BenchmarkSpec]:
    specs: List[BenchmarkSpec] = []

    specs.append(
        build_cog_spec(
            device=device,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )
    specs.extend(
        build_gvr_pred_specs(
            encoder_key=RESNET50_ENCODER_KEY,
            encoder_name=RESNET50_ENCODER_NAME,
            feature_dim=RESNET_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )
    specs.extend(
        build_gvr_kin_specs(
            encoder_key=RESNET50_ENCODER_KEY,
            encoder_name=RESNET50_ENCODER_NAME,
            feature_dim=RESNET_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )
    specs.extend(
        build_gvr_context_specs(
            encoder_key=RESNET50_ENCODER_KEY,
            encoder_name=RESNET50_ENCODER_NAME,
            feature_dim=RESNET_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )
    specs.append(build_gvr_dual_spec(jigsaws_profile=jigsaws_profile, num_batches=num_batches))

    specs.extend(
        build_gvr_pred_specs(
            encoder_key=DINOV2_LARGE_ENCODER_KEY,
            encoder_name=DINOV2_LARGE_ENCODER_NAME,
            feature_dim=DINOV2_LARGE_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )
    specs.extend(
        build_gvr_kin_specs(
            encoder_key=DINOV2_LARGE_ENCODER_KEY,
            encoder_name=DINOV2_LARGE_ENCODER_NAME,
            feature_dim=DINOV2_LARGE_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )
    specs.extend(
        build_gvr_context_specs(
            encoder_key=DINOV2_LARGE_ENCODER_KEY,
            encoder_name=DINOV2_LARGE_ENCODER_NAME,
            feature_dim=DINOV2_LARGE_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )

    specs.extend(
        build_gvr_pred_specs(
            encoder_key=CLIP_BASE_ENCODER_KEY,
            encoder_name=CLIP_BASE_ENCODER_NAME,
            feature_dim=CLIP_BASE_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )
    specs.extend(
        build_gvr_kin_specs(
            encoder_key=CLIP_BASE_ENCODER_KEY,
            encoder_name=CLIP_BASE_ENCODER_NAME,
            feature_dim=CLIP_BASE_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )
    specs.extend(
        build_gvr_context_specs(
            encoder_key=CLIP_BASE_ENCODER_KEY,
            encoder_name=CLIP_BASE_ENCODER_NAME,
            feature_dim=CLIP_BASE_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )

    specs.append(
        build_sedmamba_spec(
            device=device,
            sar_rarp_profile=sar_rarp_profile,
            sequence_lengths=sed_sequence_lengths,
            sequence_note=sed_sequence_note,
        )
    )
    return specs


def pretty_print_results(results: Sequence[BenchmarkResult]) -> None:
    headers = (
        "family",
        "model",
        "encoder",
        "prompt_type",
        "enc_params",
        "mdl_params",
        "flops",
        "stream_ms",
        "enc_ms",
        "mdl_ms",
        "e2e_ms",
    )
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


def release_encoder_runtime(runtime: EncoderRuntime | None, device: torch.device) -> None:
    if runtime is None:
        return
    runtime.model = runtime.model.to("cpu")
    empty_cuda_cache(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark encoder params, downstream params, FLOPs, and per-window latency "
            "for SEDMamba, CoG, and GVR feature models."
        )
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Benchmark device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--dummy_batches",
        type=int,
        default=DEFAULT_DUMMY_BATCHES,
        help="Number of distinct synthetic windows per fixed-window model.",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=DEFAULT_WARMUP_ITERS,
        help="Number of warmup iterations per benchmark stage.",
    )
    parser.add_argument(
        "--timing_iters",
        type=int,
        default=DEFAULT_TIMING_ITERS,
        help="Number of timed iterations per benchmark stage.",
    )
    parser.add_argument(
        "--encoder_chunk_size",
        type=int,
        default=DEFAULT_ENCODER_CHUNK_SIZE,
        help="Chunk size used when encoding frame sequences.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=REPO_ROOT / "data",
        help=(
            "Optional local data root used only to infer JIGSAWS/SAR-RARP image sizes and "
            "representative SAR-RARP50 sequence lengths. Benchmark inputs remain synthetic."
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=REPO_ROOT / "profile_reports" / "window_model_complexity_len10.csv",
        help="CSV path for benchmark results.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=REPO_ROOT / "profile_reports" / "window_model_complexity_len10.json",
        help="JSON path for benchmark results.",
    )
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)
    ensure_parent_dir(args.output_csv)
    ensure_parent_dir(args.output_json)

    sar_rarp_root = resolve_sar_rarp_root(args.data_root)
    jigsaws_profile = discover_image_profile(
        dataset_name="JIGSAWS",
        patterns=[
            str(args.data_root / "vid_frames" / "*" / "*" / "frame_*.png"),
        ],
    )
    sar_rarp_profile = discover_image_profile(
        dataset_name="SAR-RARP50",
        patterns=[
            str(sar_rarp_root / "training_set" / "*" / "images" / "*.png"),
            str(sar_rarp_root / "testing_set" / "*" / "images" / "*.png"),
        ],
    )
    sed_sequence_lengths, sed_sequence_note = discover_sar_rarp_sequence_lengths(
        args.data_root,
        max(1, args.dummy_batches),
    )

    specs = collect_model_specs(
        device=device,
        num_batches=max(1, args.dummy_batches),
        jigsaws_profile=jigsaws_profile,
        sar_rarp_profile=sar_rarp_profile,
        sed_sequence_lengths=sed_sequence_lengths,
        sed_sequence_note=sed_sequence_note,
    )

    print(f"[INFO] Benchmark device: {device}")
    print(f"[INFO] Benchmark window length: {WINDOW_LENGTH}")
    print(f"[INFO] Dummy batches per fixed-window model: {args.dummy_batches}")
    print(f"[INFO] Warmup iterations: {args.warmup_iters}")
    print(f"[INFO] Timing iterations: {args.timing_iters}")
    print(f"[INFO] Encoder chunk size: {args.encoder_chunk_size}")
    print(
        f"[INFO] JIGSAWS raw image profile: {jigsaws_profile.height}x{jigsaws_profile.width} "
        f"({jigsaws_profile.source})"
    )
    print(
        f"[INFO] SAR-RARP50 raw image profile: {sar_rarp_profile.height}x{sar_rarp_profile.width} "
        f"({sar_rarp_profile.source})"
    )
    print(f"[INFO] SEDMamba representative sequence lengths: {sed_sequence_lengths} ({sed_sequence_note})")

    results: List[BenchmarkResult] = []
    current_encoder_key = None
    current_encoder_runtime: EncoderRuntime | None = None

    for spec in specs:
        if spec.encoder_key != current_encoder_key:
            release_encoder_runtime(current_encoder_runtime, device)
            current_encoder_runtime = build_encoder_runtime(
                spec.encoder_key,
                device,
                args.encoder_chunk_size,
            )
            current_encoder_key = spec.encoder_key
            print(f"[INFO] Loaded encoder runtime: {current_encoder_runtime.display_name}")

        assert current_encoder_runtime is not None
        print(
            f"[INFO] Benchmarking {spec.model_family} / {spec.model_name} / "
            f"{spec.encoder_name} / {spec.prompt_type}"
        )
        result = benchmark_spec(
            spec,
            current_encoder_runtime,
            device,
            warmup_iters=args.warmup_iters,
            timing_iters=args.timing_iters,
        )
        results.append(result)
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
#!/usr/bin/env python3
"""
Benchmark encoder-aware per-window complexity for the main feature-based models.

For each benchmark row, the script reports:
- encoder parameter count (frozen encoder)
- downstream model parameter count
- approximate encoder + downstream FLOPs per 10-sample window
- streaming latency: synthetic window materialization + encoder preprocessing
- encoder latency: encoder forward only on already-preprocessed tensors
- model latency: downstream model forward on encoder features
- end-to-end latency per 10-sample window

Notes
- The benchmark remains synthetic and does not read dataset windows at runtime.
- Synthetic inputs mirror the repo's real tensor conventions:
  - RGB frames are uint8 before preprocessing.
  - Encoder preprocessors produce float32 tensors.
  - GVR masks are bool.
  - GVR kinematics are prepared as float32 tensors with JIGSAWS-style
    standardization and 3x downsampling before the model sees (T, 14).
- GVR rows use segment_length=10.
- CoG latency is reported for one 10-frame segment.
- SEDMamba is timed on longer synthetic sequences whose lengths are derived from
  local SAR-RARP50 label sequences when available, then normalized back to one
  10-sample window.
- FLOPs remain approximate because they rely on torch.profiler event estimates.
"""

# from __future__ import annotations

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
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAIN_DIR = REPO_ROOT / "Chain-of-Gesture"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def summarize_samples(samples: Sequence[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    mean_value = mean(samples)
    std_value = pstdev(samples) if len(samples) > 1 else 0.0
    return float(mean_value), float(std_value)


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
                return RawImageProfile(
                    dataset_name=dataset_name,
                    height=int(height),
                    width=int(width),
                    source=f"detected from {match}",
                )
            except OSError:
                continue
    return RawImageProfile(
        dataset_name=dataset_name,
        height=fallback_height,
        width=fallback_width,
        source=f"fallback {fallback_height}x{fallback_width}",
    )


def rounded_window_length(value: int) -> int:
    rounded = int(round(float(value) / float(WINDOW_LENGTH)) * WINDOW_LENGTH)
    return max(WINDOW_LENGTH, rounded)


def fallback_sedmamba_lengths(num_lengths: int) -> List[int]:
    base = list(DEFAULT_SEDMAMBA_LENGTHS)
    if num_lengths <= len(base):
        return base[:num_lengths]
    while len(base) < num_lengths:
        base.append(base[-1] + WINDOW_LENGTH * 6)
    return base[:num_lengths]


def discover_sar_rarp_sequence_lengths(data_root: Path, num_lengths: int) -> tuple[List[int], str]:
    sar_root = resolve_sar_rarp_root(data_root)
    pkl_roots = [sar_root / "train_emb_DINOv2", sar_root / "test_emb_DINOv2"]
    lengths: List[int] = []
    for pkl_root in pkl_roots:
        if not pkl_root.exists():
            continue
        for pkl_path in sorted(pkl_root.glob("*.pkl")):
            try:
                with pkl_path.open("rb") as handle:
                    payload = pickle.load(handle)
                labels = payload.get("error_GT")
                if labels is None:
                    continue
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
        return chosen[:num_lengths], (
            f"derived from {len(lengths)} SAR-RARP50 label sequences under {sar_root}"
        )
    fallback = fallback_sedmamba_lengths(num_lengths)
    return fallback, f"fallback representative SAR-RARP50-like lengths {fallback}"


def frame_windows_per_batch(raw_batch: Dict[str, Any]) -> int:
    seq_len = int(raw_batch["sequence_length"])
    return max(1, math.ceil(seq_len / WINDOW_LENGTH))


def format_sequence_profile(raw_batches: Sequence[Dict[str, Any]]) -> str:
    lengths = [str(int(batch["sequence_length"])) for batch in raw_batches]
    unique_lengths = []
    for length in lengths:
        if length not in unique_lengths:
            unique_lengths.append(length)
    return "|".join(unique_lengths)


def make_frame_bank(
    *,
    sequence_length: int,
    image_height: int,
    image_width: int,
    seed: int,
    max_bank_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    bank_size = max(1, min(sequence_length, max_bank_size))
    rng = np.random.default_rng(seed)
    frame_bank = rng.integers(
        0,
        256,
        size=(bank_size, image_height, image_width, 3),
        dtype=np.uint8,
    )
    frame_indices = rng.integers(0, bank_size, size=(sequence_length,), dtype=np.int64)
    return frame_bank, frame_indices


def make_raw_image_batches(
    *,
    sequence_lengths: Sequence[int],
    image_profile: RawImageProfile,
    seed_offset: int,
) -> List[Dict[str, Any]]:
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


def make_raw_kin_batches(
    *,
    sequence_lengths: Sequence[int],
    image_profile: RawImageProfile,
    seed_offset: int,
) -> List[Dict[str, Any]]:
    batches = make_raw_image_batches(
        sequence_lengths=sequence_lengths,
        image_profile=image_profile,
        seed_offset=seed_offset,
    )
    for index, batch in enumerate(batches):
        seq_len = int(batch["sequence_length"])
        rng = np.random.default_rng(3000 + seed_offset + index)
        batch["kine_raw"] = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(seq_len * KIN_DOWNSAMPLE, KINEMATICS_DIM),
        ).astype(np.float32)
    return batches


def make_raw_dual_batches(
    *,
    sequence_lengths: Sequence[int],
    image_profile: RawImageProfile,
    feature_dim: int,
    seed_offset: int,
) -> List[Dict[str, Any]]:
    batches = make_raw_image_batches(
        sequence_lengths=sequence_lengths,
        image_profile=image_profile,
        seed_offset=seed_offset,
    )
    for index, batch in enumerate(batches):
        seq_len = int(batch["sequence_length"])
        batch["gesture_feature_stream"] = deterministic_feature_matrix(
            seq_len,
            feature_dim,
            offset=5000 + seed_offset + index,
        )
    return batches


def materialize_frame_arrays(raw_batch: Dict[str, Any]) -> np.ndarray:
    frame_bank = raw_batch["frame_bank_uint8"]
    frame_indices = raw_batch["frame_indices"]
    return frame_bank[frame_indices].copy()


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


def import_gvr_module():
    return load_module_from_file(
        "benchmark_model_gvr_features",
        REPO_ROOT / "model_gvr_features.py",
    )


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


def import_sedmamba_module(device: torch.device):
    _install_mamba_package_stubs()
    selective_scan = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
    mamba_simple = importlib.import_module("mamba_ssm.modules.mamba_simple")
    if device.type == "cpu":
        mamba_simple.causal_conv1d_fn = None
        mamba_simple.selective_scan_fn = selective_scan.selective_scan_ref
    sys.modules["mamba_ssm"].Mamba = mamba_simple.Mamba
    return load_module_from_file(
        "benchmark_sedmamba_baseline",
        REPO_ROOT / "SEDMamba" / "baseline" / "SEDMamba.py",
    )


def install_fake_scipy() -> None:
    fake_scipy = types.ModuleType("scipy")
    fake_interpolate = types.ModuleType("scipy.interpolate")
    fake_scipy.interpolate = fake_interpolate
    sys.modules["scipy"] = fake_scipy
    sys.modules["scipy.interpolate"] = fake_interpolate


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
    try:
        import scipy  # noqa: F401
    except ImportError:
        install_fake_scipy()
    if str(CHAIN_DIR) not in sys.path:
        sys.path.insert(0, str(CHAIN_DIR))
    module = load_module_from_file(
        "benchmark_chain_of_gesture_models",
        CHAIN_DIR / "models.py",
    )
    module.clip = FakeClipModule()
    return module


def build_resnet50_runtime(device: torch.device, chunk_size: int) -> EncoderRuntime:
    weight_enum = getattr(models, "ResNet50_Weights", None)
    weights = weight_enum.DEFAULT if weight_enum is not None else None
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model = freeze_module(model.to(device))
    transform = get_resnet_transform(image_size=224, normalize=True)

    def preprocess_chunks(raw_batch: Dict[str, Any]) -> List[torch.Tensor]:
        images = materialize_pil_images(raw_batch)
        chunks: List[torch.Tensor] = []
        for image_chunk in split_list_chunks(images, chunk_size):
            chunk_tensor = torch.stack([transform(image) for image in image_chunk]).to(torch.float32)
            chunks.append(chunk_tensor)
        return chunks

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


def build_clip_base_runtime(device: torch.device, chunk_size: int) -> EncoderRuntime:
    from transformers import CLIPProcessor, CLIPVisionModel

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = freeze_module(
        CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    )

    def preprocess_chunks(raw_batch: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        images = materialize_pil_images(raw_batch)
        chunks: List[Dict[str, torch.Tensor]] = []
        for image_chunk in split_list_chunks(images, chunk_size):
            inputs = processor(images=list(image_chunk), return_tensors="pt")
            chunks.append({"pixel_values": inputs["pixel_values"].to(torch.float32)})
        return chunks

    def encode_chunk(chunk: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = model(**chunk)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0, :]
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

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model = freeze_module(AutoModel.from_pretrained("facebook/dinov2-large").to(device))

    def preprocess_chunks(raw_batch: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        images = materialize_pil_images(raw_batch)
        chunks: List[Dict[str, torch.Tensor]] = []
        for image_chunk in split_list_chunks(images, chunk_size):
            inputs = processor(images=list(image_chunk), return_tensors="pt")
            chunks.append({"pixel_values": inputs["pixel_values"].to(torch.float32)})
        return chunks

    def encode_chunk(chunk: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = model(**chunk)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0, :]
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
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = freeze_module(AutoModelForImageClassification.from_pretrained(model_id).to(device))

    def preprocess_chunks(raw_batch: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
        images = materialize_pil_images(raw_batch)
        chunks: List[Dict[str, torch.Tensor]] = []
        for image_chunk in split_list_chunks(images, chunk_size):
            inputs = processor(images=list(image_chunk), return_tensors="pt")
            chunks.append({"pixel_values": inputs["pixel_values"].to(torch.float32)})
        return chunks

    def encode_chunk(chunk: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = model(**chunk)
        return outputs.logits.to(torch.float32)

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


def prepare_image_only_batch(raw_batch: Dict[str, Any], encoder_runtime: EncoderRuntime) -> Dict[str, Any]:
    seq_len = int(raw_batch["sequence_length"])
    return {
        "sequence_length": seq_len,
        "encoder_chunks": encoder_runtime.preprocess_chunks(raw_batch),
        "masks": torch.ones(seq_len, dtype=torch.bool),
    }


def prepare_kin_batch(raw_batch: Dict[str, Any], encoder_runtime: EncoderRuntime) -> Dict[str, Any]:
    prepared = prepare_image_only_batch(raw_batch, encoder_runtime)
    seq_len = int(raw_batch["sequence_length"])
    prepared["kine"] = prepare_jigsaws_kinematics(raw_batch["kine_raw"], target_length=seq_len)
    return prepared


def prepare_dual_batch(raw_batch: Dict[str, Any], encoder_runtime: EncoderRuntime) -> Dict[str, Any]:
    prepared = prepare_image_only_batch(raw_batch, encoder_runtime)
    prepared["ges_features"] = raw_batch["gesture_feature_stream"].clone().to(torch.float32)
    return prepared


def build_sedmamba_model_inputs(
    prepared_batch: Dict[str, Any],
    encoded_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    del prepared_batch, device
    return {"features": encoded_features.transpose(0, 1).unsqueeze(0)}


def build_cog_model_inputs(
    prepared_batch: Dict[str, Any],
    encoded_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    del prepared_batch, device
    return {"features": encoded_features.unsqueeze(0)}


def build_gvr_pred_model_inputs(
    prepared_batch: Dict[str, Any],
    encoded_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    return {
        "features": encoded_features.unsqueeze(0),
        "masks": prepared_batch["masks"].unsqueeze(0).to(device),
    }


def build_gvr_kin_model_inputs(
    prepared_batch: Dict[str, Any],
    encoded_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    return {
        "features": encoded_features.unsqueeze(0),
        "kine": prepared_batch["kine"].unsqueeze(0).to(device),
        "masks": prepared_batch["masks"].unsqueeze(0).to(device),
    }


def build_gvr_context_model_inputs(
    prepared_batch: Dict[str, Any],
    encoded_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    del prepared_batch, device
    return {"features": encoded_features.unsqueeze(0)}


def build_gvr_dual_model_inputs(
    prepared_batch: Dict[str, Any],
    encoded_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    return {
        "base_features": encoded_features.unsqueeze(0),
        "ges_features": prepared_batch["ges_features"].unsqueeze(0).to(device),
        "masks": prepared_batch["masks"].unsqueeze(0).to(device),
    }


def encode_chunks(
    encoder_runtime: EncoderRuntime,
    encoder_chunks: Sequence[Any],
    device: torch.device,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    with torch.inference_mode():
        for chunk in encoder_chunks:
            device_chunk = recursive_to_device(chunk, device)
            outputs.append(encoder_runtime.encode_chunk(device_chunk))
    return torch.cat(outputs, dim=0) if outputs else torch.empty(0, encoder_runtime.feature_dim, device=device)


def encode_chunks_forward_only_timed(
    encoder_runtime: EncoderRuntime,
    encoder_chunks: Sequence[Any],
    device: torch.device,
) -> tuple[torch.Tensor, float]:
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
    if not outputs:
        return torch.empty(0, encoder_runtime.feature_dim, device=device), 0.0
    return torch.cat(outputs, dim=0), total_ms


def profiler_activities(device: torch.device) -> List[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda" and hasattr(torch.profiler.ProfilerActivity, "CUDA"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def profile_flops(callable_fn: Callable[[], Any], device: torch.device) -> int:
    try:
        with torch.profiler.profile(
            activities=profiler_activities(device),
            with_flops=True,
            acc_events=True,
        ) as profiler:
            with torch.inference_mode():
                callable_fn()
            maybe_sync(device)
        return int(sum(event.flops or 0 for event in profiler.key_averages()))
    except Exception:
        return 0


def measure_streaming_latency(
    spec: BenchmarkSpec,
    encoder_runtime: EncoderRuntime,
    device: torch.device,
    warmup_iters: int,
    timing_iters: int,
) -> tuple[float, float]:
    for index in range(warmup_iters):
        raw_batch = spec.raw_batches[index % len(spec.raw_batches)]
        spec.prepare_raw_batch(raw_batch, encoder_runtime)
    samples_ms: List[float] = []
    for index in range(timing_iters):
        raw_batch = spec.raw_batches[index % len(spec.raw_batches)]
        maybe_sync(device)
        start = time.perf_counter()
        spec.prepare_raw_batch(raw_batch, encoder_runtime)
        maybe_sync(device)
        samples_ms.append(
            ((time.perf_counter() - start) * 1000.0) / float(frame_windows_per_batch(raw_batch))
        )
    return summarize_samples(samples_ms)


def measure_encoder_latency(
    spec: BenchmarkSpec,
    encoder_runtime: EncoderRuntime,
    prepared_batches: Sequence[Dict[str, Any]],
    device: torch.device,
    warmup_iters: int,
    timing_iters: int,
) -> tuple[float, float]:
    for index in range(warmup_iters):
        prepared = prepared_batches[index % len(prepared_batches)]
        encode_chunks(encoder_runtime, prepared["encoder_chunks"], device)
    samples_ms: List[float] = []
    for index in range(timing_iters):
        raw_batch = spec.raw_batches[index % len(spec.raw_batches)]
        prepared = prepared_batches[index % len(prepared_batches)]
        _, total_ms = encode_chunks_forward_only_timed(
            encoder_runtime,
            prepared["encoder_chunks"],
            device,
        )
        samples_ms.append(total_ms / float(frame_windows_per_batch(raw_batch)))
    return summarize_samples(samples_ms)


def measure_model_latency(
    spec: BenchmarkSpec,
    model_inputs: Sequence[Dict[str, Any]],
    device: torch.device,
    warmup_iters: int,
    timing_iters: int,
) -> tuple[str, float, float]:
    model = spec.model.to(device).eval()
    sample_output = None
    with torch.inference_mode():
        sample_output = spec.forward_model(model, model_inputs[0])
    output_shape = tensor_shape_to_str(sample_output.detach().cpu())
    for index in range(warmup_iters):
        with torch.inference_mode():
            spec.forward_model(model, model_inputs[index % len(model_inputs)])
        maybe_sync(device)
    samples_ms: List[float] = []
    for index in range(timing_iters):
        raw_batch = spec.raw_batches[index % len(spec.raw_batches)]
        maybe_sync(device)
        start = time.perf_counter()
        with torch.inference_mode():
            spec.forward_model(model, model_inputs[index % len(model_inputs)])
        maybe_sync(device)
        samples_ms.append(
            ((time.perf_counter() - start) * 1000.0) / float(frame_windows_per_batch(raw_batch))
        )
    latency_mean, latency_std = summarize_samples(samples_ms)
    return output_shape, latency_mean, latency_std


def measure_end_to_end_latency(
    spec: BenchmarkSpec,
    encoder_runtime: EncoderRuntime,
    device: torch.device,
    warmup_iters: int,
    timing_iters: int,
) -> tuple[float, float]:
    def run_once(raw_batch: Dict[str, Any]) -> torch.Tensor:
        prepared = spec.prepare_raw_batch(raw_batch, encoder_runtime)
        encoded = encode_chunks(encoder_runtime, prepared["encoder_chunks"], device)
        model_inputs = spec.build_model_inputs(prepared, encoded, device)
        with torch.inference_mode():
            return spec.forward_model(spec.model, model_inputs)

    for index in range(warmup_iters):
        run_once(spec.raw_batches[index % len(spec.raw_batches)])
        maybe_sync(device)
    samples_ms: List[float] = []
    for index in range(timing_iters):
        raw_batch = spec.raw_batches[index % len(spec.raw_batches)]
        maybe_sync(device)
        start = time.perf_counter()
        run_once(raw_batch)
        maybe_sync(device)
        samples_ms.append(
            ((time.perf_counter() - start) * 1000.0) / float(frame_windows_per_batch(raw_batch))
        )
    return summarize_samples(samples_ms)


def compute_flops_per_window(
    spec: BenchmarkSpec,
    encoder_runtime: EncoderRuntime,
    prepared_batches: Sequence[Dict[str, Any]],
    device: torch.device,
) -> int:
    flops_samples: List[float] = []
    for raw_batch, prepared in zip(spec.raw_batches, prepared_batches):
        encoder_flops = 0
        for chunk in prepared["encoder_chunks"]:
            device_chunk = recursive_to_device(chunk, device)
            encoder_flops += profile_flops(
                lambda _chunk=device_chunk: encoder_runtime.encode_chunk(_chunk),
                device,
            )
        encoded = encode_chunks(encoder_runtime, prepared["encoder_chunks"], device)
        model_inputs = spec.build_model_inputs(prepared, encoded, device)
        model_flops = profile_flops(
            lambda _inputs=model_inputs: spec.forward_model(spec.model, _inputs),
            device,
        )
        flops_samples.append(
            float(encoder_flops + model_flops) / float(frame_windows_per_batch(raw_batch))
        )
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
    model_inputs = [
        spec.build_model_inputs(prepared, encoded, device)
        for prepared, encoded in zip(prepared_batches, encoded_batches)
    ]

    streaming_latency_ms, streaming_latency_std_ms = measure_streaming_latency(
        spec,
        encoder_runtime,
        device,
        warmup_iters,
        timing_iters,
    )
    encoder_latency_ms, encoder_latency_std_ms = measure_encoder_latency(
        spec,
        encoder_runtime,
        prepared_batches,
        device,
        warmup_iters,
        timing_iters,
    )
    output_shape, model_latency_ms, model_latency_std_ms = measure_model_latency(
        spec,
        model_inputs,
        device,
        warmup_iters,
        timing_iters,
    )
    end_to_end_latency_ms, end_to_end_latency_std_ms = measure_end_to_end_latency(
        spec,
        encoder_runtime,
        device,
        warmup_iters,
        timing_iters,
    )
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


def build_sedmamba_spec(
    *,
    device: torch.device,
    sar_rarp_profile: RawImageProfile,
    sequence_lengths: Sequence[int],
    sequence_note: str,
):
    sed_module = import_sedmamba_module(device)
    model = sed_module.MultiStageModel(
        num_block=3,
        com_factor=64,
        dim=SED_DINOV2_FEATURE_DIM,
        num_classes=1,
    )
    raw_batches = make_raw_image_batches(
        sequence_lengths=sequence_lengths,
        image_profile=sar_rarp_profile,
        seed_offset=10,
    )
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


def build_cog_spec(
    *,
    device: torch.device,
    jigsaws_profile: RawImageProfile,
):
    cog_module = import_cog_module()
    gest_prompt_file = register_temp_file(
        tempfile.NamedTemporaryFile(prefix="cog_prompt_", suffix=".pt", delete=False).name
    )
    args = SimpleNamespace(
        train=1,
        k=WINDOW_LENGTH,
        layers=10,
        stages=8,
        lambda_=0.15,
        dmodel=64,
        len_q=WINDOW_LENGTH,
    )
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
    raw_batches = make_raw_image_batches(
        sequence_lengths=[WINDOW_LENGTH] * DEFAULT_DUMMY_BATCHES,
        image_profile=jigsaws_profile,
        seed_offset=100,
    )
    return BenchmarkSpec(
        model_family="CoG",
        model_name="COG",
        prompt_type="default_gesture_list",
        encoder_key=RESNET50_ENCODER_KEY,
        encoder_name=RESNET50_ENCODER_NAME,
        feature_source="resnet50",
        benchmark_window_length=WINDOW_LENGTH,
        sequence_profile=format_sequence_profile(raw_batches),
        input_description=(
            f"raw_frames=10x{jigsaws_profile.height}x{jigsaws_profile.width}x3 uint8 -> "
            f"ResNet50 features=1x10x{RESNET_FEATURE_DIM}"
        ),
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


def build_gvr_pred_specs(
    *,
    encoder_key: str,
    encoder_name: str,
    feature_dim: int,
    jigsaws_profile: RawImageProfile,
    num_batches: int,
):
    gvr_module = import_gvr_module()
    specs: List[BenchmarkSpec] = []
    raw_batches = make_raw_image_batches(
        sequence_lengths=[WINDOW_LENGTH] * num_batches,
        image_profile=jigsaws_profile,
        seed_offset=200,
    )
    for prompt_index, (prompt_type, prompts) in enumerate(PROMPT_SETS.items()):
        prompt_features = deterministic_prompt_features(
            len(prompts),
            GVR_TEXT_DIM,
            offset=100 * prompt_index,
        )
        model = gvr_module.GVRModulePredFeatures(
            gesture_prompts=prompts,
            d_text=GVR_TEXT_DIM,
            d_model=feature_dim,
            num_heads=1,
            segment_length=WINDOW_LENGTH,
            position=True,
            dropout=0.5,
            prompt_features=prompt_features,
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
                input_description=(
                    f"raw_frames=10x{jigsaws_profile.height}x{jigsaws_profile.width}x3 uint8 -> "
                    f"features=1x10x{feature_dim}, masks=1x10 bool"
                ),
                notes=(
                    f"Prompt-conditioned prediction head with encoder {encoder_name}; "
                    f"raw image profile {jigsaws_profile.source}; synthetic CLIP text prompt features kept outside encoder stats."
                ),
                model=model,
                raw_batches=raw_batches,
                prepare_raw_batch=prepare_image_only_batch,
                build_model_inputs=build_gvr_pred_model_inputs,
                forward_model=lambda _model, batch: _model(
                    batch["features"],
                    masks=batch["masks"],
                ),
            )
        )
    return specs


def build_gvr_kin_specs(
    *,
    encoder_key: str,
    encoder_name: str,
    feature_dim: int,
    jigsaws_profile: RawImageProfile,
    num_batches: int,
):
    gvr_module = import_gvr_module()
    specs: List[BenchmarkSpec] = []
    raw_batches = make_raw_kin_batches(
        sequence_lengths=[WINDOW_LENGTH] * num_batches,
        image_profile=jigsaws_profile,
        seed_offset=300,
    )
    for prompt_index, (prompt_type, prompts) in enumerate(PROMPT_SETS.items()):
        prompt_features = deterministic_prompt_features(
            len(prompts),
            GVR_TEXT_DIM,
            offset=900 + 100 * prompt_index,
        )
        model = gvr_module.GVRModulePredKinFeatures(
            gesture_prompts=prompts,
            d_text=GVR_TEXT_DIM,
            d_model=feature_dim,
            num_heads=1,
            segment_length=WINDOW_LENGTH,
            position=True,
            dropout=0.3,
            prompt_features=prompt_features,
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
                    f"raw_kin=30x{KINEMATICS_DIM} float32 -> features=1x10x{feature_dim}, "
                    f"kine=1x10x{KINEMATICS_DIM}, masks=1x10 bool"
                ),
                notes=(
                    f"Visual+kinematics prompt-conditioned head with encoder {encoder_name}; "
                    "streaming latency includes JIGSAWS-style kinematics normalization and 3x downsampling."
                ),
                model=model,
                raw_batches=raw_batches,
                prepare_raw_batch=prepare_kin_batch,
                build_model_inputs=build_gvr_kin_model_inputs,
                forward_model=lambda _model, batch: _model(
                    batch["features"],
                    batch["kine"],
                    masks=batch["masks"],
                ),
            )
        )
    return specs


def build_gvr_context_specs(
    *,
    encoder_key: str,
    encoder_name: str,
    feature_dim: int,
    jigsaws_profile: RawImageProfile,
    num_batches: int,
):
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
    raw_batches = make_raw_image_batches(
        sequence_lengths=[WINDOW_LENGTH] * num_batches,
        image_profile=jigsaws_profile,
        seed_offset=400,
    )
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
            input_description=(
                f"raw_frames=10x{jigsaws_profile.height}x{jigsaws_profile.width}x3 uint8 -> "
                f"features=1x10x{feature_dim}"
            ),
            notes=(
                f"Single-stream context model with encoder {encoder_name}; "
                f"raw image profile {jigsaws_profile.source}."
            ),
            model=model,
            raw_batches=raw_batches,
            prepare_raw_batch=prepare_image_only_batch,
            build_model_inputs=build_gvr_context_model_inputs,
            forward_model=lambda _model, batch: _model(batch["features"]),
        )
    ]


def build_gvr_dual_spec(
    *,
    jigsaws_profile: RawImageProfile,
    num_batches: int,
):
    gvr_module = import_gvr_module()
    model = gvr_module.GVRModuleContexPredDualFeatures(
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
        feature_source="resnet50+synthetic_gesture_stream",
        benchmark_window_length=WINDOW_LENGTH,
        sequence_profile=format_sequence_profile(raw_batches),
        input_description=(
            f"raw_frames=10x{jigsaws_profile.height}x{jigsaws_profile.width}x3 uint8 -> "
            f"base=1x10x{RESNET_FEATURE_DIM}, ges=1x10x{RESNET_FEATURE_DIM}, masks=1x10 bool"
        ),
        notes=(
            "Dual-stream context row keeps the original paired-stream structure. "
            "Encoder stats/latency cover the base ResNet50 stream; the gesture stream remains a synthetic paired feature input."
        ),
        model=model,
        raw_batches=raw_batches,
        prepare_raw_batch=prepare_dual_batch,
        build_model_inputs=build_gvr_dual_model_inputs,
        forward_model=lambda _model, batch: _model(
            batch["base_features"],
            batch["ges_features"],
            masks=batch["masks"],
        ),
    )


def collect_model_specs(
    *,
    device: torch.device,
    num_batches: int,
    jigsaws_profile: RawImageProfile,
    sar_rarp_profile: RawImageProfile,
    sed_sequence_lengths: Sequence[int],
    sed_sequence_note: str,
) -> List[BenchmarkSpec]:
    specs: List[BenchmarkSpec] = []

    specs.append(
        build_cog_spec(
            device=device,
            jigsaws_profile=jigsaws_profile,
        )
    )
    specs.extend(
        build_gvr_pred_specs(
            encoder_key=RESNET50_ENCODER_KEY,
            encoder_name=RESNET50_ENCODER_NAME,
            feature_dim=RESNET_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )
    specs.extend(
        build_gvr_kin_specs(
            encoder_key=RESNET50_ENCODER_KEY,
            encoder_name=RESNET50_ENCODER_NAME,
            feature_dim=RESNET_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )
    specs.extend(
        build_gvr_context_specs(
            encoder_key=RESNET50_ENCODER_KEY,
            encoder_name=RESNET50_ENCODER_NAME,
            feature_dim=RESNET_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )
    specs.append(build_gvr_dual_spec(jigsaws_profile=jigsaws_profile, num_batches=num_batches))

    specs.extend(
        build_gvr_pred_specs(
            encoder_key=DINOV2_LARGE_ENCODER_KEY,
            encoder_name=DINOV2_LARGE_ENCODER_NAME,
            feature_dim=DINOV2_LARGE_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )
    specs.extend(
        build_gvr_kin_specs(
            encoder_key=DINOV2_LARGE_ENCODER_KEY,
            encoder_name=DINOV2_LARGE_ENCODER_NAME,
            feature_dim=DINOV2_LARGE_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )
    specs.extend(
        build_gvr_context_specs(
            encoder_key=DINOV2_LARGE_ENCODER_KEY,
            encoder_name=DINOV2_LARGE_ENCODER_NAME,
            feature_dim=DINOV2_LARGE_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )

    specs.extend(
        build_gvr_pred_specs(
            encoder_key=CLIP_BASE_ENCODER_KEY,
            encoder_name=CLIP_BASE_ENCODER_NAME,
            feature_dim=CLIP_BASE_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )
    specs.extend(
        build_gvr_kin_specs(
            encoder_key=CLIP_BASE_ENCODER_KEY,
            encoder_name=CLIP_BASE_ENCODER_NAME,
            feature_dim=CLIP_BASE_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )
    specs.extend(
        build_gvr_context_specs(
            encoder_key=CLIP_BASE_ENCODER_KEY,
            encoder_name=CLIP_BASE_ENCODER_NAME,
            feature_dim=CLIP_BASE_FEATURE_DIM,
            jigsaws_profile=jigsaws_profile,
            num_batches=num_batches,
        )
    )

    specs.append(
        build_sedmamba_spec(
            device=device,
            sar_rarp_profile=sar_rarp_profile,
            sequence_lengths=sed_sequence_lengths,
            sequence_note=sed_sequence_note,
        )
    )
    return specs


def pretty_print_results(results: Sequence[BenchmarkResult]) -> None:
    headers = (
        "family",
        "model",
        "encoder",
        "prompt_type",
        "enc_params",
        "mdl_params",
        "flops",
        "stream_ms",
        "enc_ms",
        "mdl_ms",
        "e2e_ms",
    )
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


def release_encoder_runtime(runtime: EncoderRuntime | None, device: torch.device) -> None:
    if runtime is None:
        return
    runtime.model = runtime.model.to("cpu")
    empty_cuda_cache(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark encoder params, downstream params, FLOPs, and per-window latency "
            "for SEDMamba, CoG, and GVR feature models."
        )
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Benchmark device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--dummy_batches",
        type=int,
        default=DEFAULT_DUMMY_BATCHES,
        help="Number of distinct synthetic windows per fixed-window model.",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=DEFAULT_WARMUP_ITERS,
        help="Number of warmup iterations per benchmark stage.",
    )
    parser.add_argument(
        "--timing_iters",
        type=int,
        default=DEFAULT_TIMING_ITERS,
        help="Number of timed iterations per benchmark stage.",
    )
    parser.add_argument(
        "--encoder_chunk_size",
        type=int,
        default=DEFAULT_ENCODER_CHUNK_SIZE,
        help="Chunk size used when encoding frame sequences.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=REPO_ROOT / "data",
        help=(
            "Optional local data root used only to infer JIGSAWS/SAR-RARP image sizes and "
            "representative SAR-RARP50 sequence lengths. Benchmark inputs remain synthetic."
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=REPO_ROOT / "profile_reports" / "window_model_complexity_len10.csv",
        help="CSV path for benchmark results.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=REPO_ROOT / "profile_reports" / "window_model_complexity_len10.json",
        help="JSON path for benchmark results.",
    )
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)
    ensure_parent_dir(args.output_csv)
    ensure_parent_dir(args.output_json)

    sar_rarp_root = resolve_sar_rarp_root(args.data_root)
    jigsaws_profile = discover_image_profile(
        dataset_name="JIGSAWS",
        patterns=[
            str(args.data_root / "vid_frames" / "*" / "*" / "frame_*.png"),
        ],
    )
    sar_rarp_profile = discover_image_profile(
        dataset_name="SAR-RARP50",
        patterns=[
            str(sar_rarp_root / "training_set" / "*" / "images" / "*.png"),
            str(sar_rarp_root / "testing_set" / "*" / "images" / "*.png"),
        ],
    )
    sed_sequence_lengths, sed_sequence_note = discover_sar_rarp_sequence_lengths(
        args.data_root,
        max(1, args.dummy_batches),
    )

    specs = collect_model_specs(
        device=device,
        num_batches=max(1, args.dummy_batches),
        jigsaws_profile=jigsaws_profile,
        sar_rarp_profile=sar_rarp_profile,
        sed_sequence_lengths=sed_sequence_lengths,
        sed_sequence_note=sed_sequence_note,
    )

    print(f"[INFO] Benchmark device: {device}")
    print(f"[INFO] Benchmark window length: {WINDOW_LENGTH}")
    print(f"[INFO] Dummy batches per fixed-window model: {args.dummy_batches}")
    print(f"[INFO] Warmup iterations: {args.warmup_iters}")
    print(f"[INFO] Timing iterations: {args.timing_iters}")
    print(f"[INFO] Encoder chunk size: {args.encoder_chunk_size}")
    print(
        f"[INFO] JIGSAWS raw image profile: {jigsaws_profile.height}x{jigsaws_profile.width} "
        f"({jigsaws_profile.source})"
    )
    print(
        f"[INFO] SAR-RARP50 raw image profile: {sar_rarp_profile.height}x{sar_rarp_profile.width} "
        f"({sar_rarp_profile.source})"
    )
    print(f"[INFO] SEDMamba representative sequence lengths: {sed_sequence_lengths} ({sed_sequence_note})")

    results: List[BenchmarkResult] = []
    current_encoder_key = None
    current_encoder_runtime: EncoderRuntime | None = None

    for spec in specs:
        if spec.encoder_key != current_encoder_key:
            release_encoder_runtime(current_encoder_runtime, device)
            current_encoder_runtime = build_encoder_runtime(
                spec.encoder_key,
                device,
                args.encoder_chunk_size,
            )
            current_encoder_key = spec.encoder_key
            print(f"[INFO] Loaded encoder runtime: {current_encoder_runtime.display_name}")

        assert current_encoder_runtime is not None
        print(
            f"[INFO] Benchmarking {spec.model_family} / {spec.model_name} / "
            f"{spec.encoder_name} / {spec.prompt_type}"
        )
        result = benchmark_spec(
            spec,
            current_encoder_runtime,
            device,
            warmup_iters=args.warmup_iters,
            timing_iters=args.timing_iters,
        )
        results.append(result)
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
#!/usr/bin/env python3
"""
Benchmark non-encoder model complexity for the main feature-based models.

What this script reports for each requested model:
- non-encoder parameter count
- trainable parameter count
- approximate FLOPs for one dummy forward pass
- mean / std response time per window

Notes
- External encoders are intentionally excluded from the instantiated models:
  DINOv2, ResNet, CLIP text/vision encoders are not loaded as model parameters.
- GVR prompt-conditioned models use synthetic precomputed prompt features so the
  CLIP text encoder is excluded while preserving the same downstream tensor
  shapes used by the error detection heads.
- Chain-of-Gesture uses a lightweight fake CLIP text interface for the same
  reason. The resulting prompt embedding table remains part of the model state.
- On CPU, SEDMamba is forced onto its reference scan path so it can run without
  CUDA-only fused kernels. This changes the execution path for latency/FLOPs on
  CPU, but keeps the same learned parameter shapes.
"""

# from __future__ import annotations

import argparse
import atexit
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAIN_DIR = REPO_ROOT / "Chain-of-Gesture"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

RESNET_FEATURE_DIM = 2048
DINO_FEATURE_DIM = 1000
KINEMATICS_DIM = 14
GVR_TEXT_DIM = 512
SEGMENT_LENGTH = 10
DEFAULT_DUMMY_BATCHES = 3
DEFAULT_WARMUP_ITERS = 1
DEFAULT_TIMING_ITERS = 3

_TEMP_FILES: List[str] = []


@dataclass
class BenchmarkResult:
    model_family: str
    model_name: str
    prompt_type: str
    feature_source: str
    sequence_length: int
    input_description: str
    output_shape: str
    non_encoder_params: int
    trainable_params: int
    flops_per_window: int
    latency_mean_ms: float
    latency_std_ms: float
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def recursive_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: recursive_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(recursive_to_device(value, device) for value in obj)
    return obj


def total_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def trainable_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def tensor_shape_to_str(tensor: torch.Tensor) -> str:
    return "x".join(str(dim) for dim in tensor.shape)


def deterministic_prompt_features(num_prompts: int, dim: int, offset: int = 0) -> torch.Tensor:
    base = torch.arange(num_prompts * dim, dtype=torch.float32).reshape(num_prompts, dim)
    return torch.sin((base + float(offset)) / 37.0)


def install_transformers_shim() -> None:
    fake = types.ModuleType("transformers")

    class _UnusedFactory:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError(
                "The benchmark uses precomputed prompt features and should never "
                "load CLIP text encoders."
            )

    fake.CLIPTextModel = _UnusedFactory
    fake.CLIPTokenizer = _UnusedFactory
    sys.modules["transformers"] = fake


def install_fake_scipy() -> None:
    fake_scipy = types.ModuleType("scipy")
    fake_interpolate = types.ModuleType("scipy.interpolate")
    fake_scipy.interpolate = fake_interpolate
    sys.modules["scipy"] = fake_scipy
    sys.modules["scipy.interpolate"] = fake_interpolate


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


def import_gvr_module():
    install_transformers_shim()
    return load_module_from_file(
        "benchmark_model_gvr_features",
        REPO_ROOT / "model_gvr_features.py",
    )


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
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg


def import_sedmamba_module(device: torch.device):
    _install_mamba_package_stubs()
    selective_scan = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
    mamba_simple = importlib.import_module("mamba_ssm.modules.mamba_simple")
    if device.type == "cpu":
        # CPU fallback: use the pure PyTorch reference scan + regular Conv1d path.
        mamba_simple.causal_conv1d_fn = None
        mamba_simple.selective_scan_fn = selective_scan.selective_scan_ref
    sys.modules["mamba_ssm"].Mamba = mamba_simple.Mamba
    return load_module_from_file(
        "benchmark_sedmamba_baseline",
        REPO_ROOT / "SEDMamba" / "baseline" / "SEDMamba.py",
    )


def import_cog_module():
    install_fake_scipy()
    if str(CHAIN_DIR) not in sys.path:
        sys.path.insert(0, str(CHAIN_DIR))
    module = load_module_from_file(
        "benchmark_chain_of_gesture_models",
        CHAIN_DIR / "models.py",
    )
    module.clip = FakeClipModule()
    return module


def make_masked_feature_batches(
    *,
    seq_len: int,
    feature_dim: int,
    num_batches: int,
    seed_offset: int = 0,
) -> List[Dict[str, torch.Tensor]]:
    valid_lengths = [seq_len, max(seq_len - 2, 1), max(seq_len - 4, 1)]
    batches: List[Dict[str, torch.Tensor]] = []
    for idx in range(num_batches):
        valid_len = valid_lengths[idx % len(valid_lengths)]
        generator = torch.Generator().manual_seed(1000 + seed_offset + idx)
        features = torch.randn(1, seq_len, feature_dim, generator=generator)
        masks = torch.zeros(1, seq_len, dtype=torch.bool)
        masks[:, :valid_len] = True
        features[:, valid_len:] = 0.0
        batches.append({"features": features, "masks": masks})
    return batches


def make_kin_batches(seq_len: int, num_batches: int) -> List[Dict[str, torch.Tensor]]:
    base_batches = make_masked_feature_batches(
        seq_len=seq_len,
        feature_dim=RESNET_FEATURE_DIM,
        num_batches=num_batches,
        seed_offset=100,
    )
    batches: List[Dict[str, torch.Tensor]] = []
    for idx, batch in enumerate(base_batches):
        generator = torch.Generator().manual_seed(2000 + idx)
        kine = torch.randn(1, seq_len, KINEMATICS_DIM, generator=generator)
        kine[:, ~batch["masks"][0], :] = 0.0
        batches.append(
            {
                "features": batch["features"],
                "masks": batch["masks"],
                "kine": kine,
            }
        )
    return batches


def make_dual_feature_batches(seq_len: int, num_batches: int) -> List[Dict[str, torch.Tensor]]:
    base_batches = make_masked_feature_batches(
        seq_len=seq_len,
        feature_dim=RESNET_FEATURE_DIM,
        num_batches=num_batches,
        seed_offset=200,
    )
    batches: List[Dict[str, torch.Tensor]] = []
    for idx, batch in enumerate(base_batches):
        generator = torch.Generator().manual_seed(3000 + idx)
        ges_features = torch.randn(1, seq_len, RESNET_FEATURE_DIM, generator=generator)
        ges_features[:, ~batch["masks"][0], :] = 0.0
        batches.append(
            {
                "base_features": batch["features"],
                "ges_features": ges_features,
                "masks": batch["masks"],
            }
        )
    return batches


def make_plain_feature_batches(
    *,
    seq_len: int,
    feature_dim: int,
    num_batches: int,
    seed_offset: int = 0,
) -> List[Dict[str, torch.Tensor]]:
    batches: List[Dict[str, torch.Tensor]] = []
    for idx in range(num_batches):
        generator = torch.Generator().manual_seed(4000 + seed_offset + idx)
        features = torch.randn(1, seq_len, feature_dim, generator=generator)
        batches.append({"features": features})
    return batches


def make_sedmamba_batches(seq_len: int, num_batches: int) -> List[Dict[str, torch.Tensor]]:
    batches: List[Dict[str, torch.Tensor]] = []
    for idx in range(num_batches):
        generator = torch.Generator().manual_seed(5000 + idx)
        features = torch.randn(1, DINO_FEATURE_DIM, seq_len, generator=generator)
        batches.append({"features": features})
    return batches


def benchmark_model(
    *,
    model: torch.nn.Module,
    forward_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    batches: Sequence[Dict[str, torch.Tensor]],
    device: torch.device,
    warmup_iters: int,
    timing_iters: int,
) -> tuple[str, int, float, float]:
    model = model.to(device)
    model.eval()
    prepared_batches = [recursive_to_device(batch, device) for batch in batches]

    with torch.inference_mode():
        sample_output = forward_fn(prepared_batches[0])
    output_shape = tensor_shape_to_str(sample_output.detach().cpu())

    for idx in range(warmup_iters):
        with torch.inference_mode():
            forward_fn(prepared_batches[idx % len(prepared_batches)])
        maybe_sync(device)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_flops=True,
        acc_events=True,
    ) as profiler:
        with torch.inference_mode():
            forward_fn(prepared_batches[0])
        maybe_sync(device)
    flops = int(sum(event.flops or 0 for event in profiler.key_averages()))

    latency_samples_ms: List[float] = []
    for idx in range(timing_iters):
        batch = prepared_batches[idx % len(prepared_batches)]
        maybe_sync(device)
        start = time.perf_counter()
        with torch.inference_mode():
            forward_fn(batch)
        maybe_sync(device)
        latency_samples_ms.append((time.perf_counter() - start) * 1000.0)

    latency_mean = mean(latency_samples_ms)
    latency_std = pstdev(latency_samples_ms) if len(latency_samples_ms) > 1 else 0.0
    return output_shape, flops, latency_mean, latency_std


def build_sedmamba_spec(device: torch.device, num_batches: int):
    sed_module = import_sedmamba_module(device)
    model = sed_module.MultiStageModel(
        num_block=3,
        com_factor=64,
        dim=DINO_FEATURE_DIM,
        num_classes=1,
    )
    batches = make_sedmamba_batches(seq_len=SEGMENT_LENGTH, num_batches=num_batches)
    return {
        "model_family": "SEDMamba",
        "model_name": "MultiStageModel",
        "prompt_type": "n/a",
        "feature_source": "dinov2",
        "sequence_length": SEGMENT_LENGTH,
        "input_description": f"1x{DINO_FEATURE_DIM}x{SEGMENT_LENGTH}",
        "notes": "CPU benchmarks use the reference Mamba scan path; encoder excluded.",
        "model": model,
        "batches": batches,
        "forward_fn": lambda batch: model(batch["features"]),
    }


def build_cog_spec(device: torch.device, num_batches: int):
    cog_module = import_cog_module()
    gest_prompt_file = register_temp_file(
        tempfile.NamedTemporaryFile(prefix="cog_prompt_", suffix=".pt", delete=False).name
    )
    args = SimpleNamespace(
        train=1,
        k=SEGMENT_LENGTH,
        layers=10,
        stages=8,
        lambda_=0.15,
        dmodel=64,
        len_q=SEGMENT_LENGTH,
    )
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
    batches = make_plain_feature_batches(
        seq_len=SEGMENT_LENGTH,
        feature_dim=RESNET_FEATURE_DIM,
        num_batches=num_batches,
        seed_offset=600,
    )
    return {
        "model_family": "CoG",
        "model_name": "COG",
        "prompt_type": "default_gesture_list",
        "feature_source": "resnet50",
        "sequence_length": SEGMENT_LENGTH,
        "input_description": f"1x{SEGMENT_LENGTH}x{RESNET_FEATURE_DIM}",
        "notes": "CLIP text encoder excluded via synthetic prompt embeddings; len_q=10 and k=10 for one-window benchmarking.",
        "model": model,
        "batches": batches,
        "forward_fn": lambda batch: model(batch["features"])[0][0],
    }


def build_gvr_pred_specs(num_batches: int):
    gvr_module = import_gvr_module()
    specs = []
    for prompt_index, (prompt_type, prompts) in enumerate(PROMPT_SETS.items()):
        prompt_features = deterministic_prompt_features(
            len(prompts), GVR_TEXT_DIM, offset=100 * prompt_index
        )
        model = gvr_module.GVRModulePredFeatures(
            gesture_prompts=prompts,
            d_text=GVR_TEXT_DIM,
            d_model=RESNET_FEATURE_DIM,
            num_heads=1,
            segment_length=SEGMENT_LENGTH,
            position=True,
            dropout=0.5,
            prompt_features=prompt_features,
            use_pooled_branch=True,
        )
        batches = make_masked_feature_batches(
            seq_len=SEGMENT_LENGTH,
            feature_dim=RESNET_FEATURE_DIM,
            num_batches=num_batches,
            seed_offset=700 + prompt_index,
        )
        specs.append(
            {
                "model_family": "GVR",
                "model_name": "GVRModulePredFeatures",
                "prompt_type": prompt_type,
                "feature_source": "resnet50",
                "sequence_length": SEGMENT_LENGTH,
                "input_description": f"features=1x{SEGMENT_LENGTH}x{RESNET_FEATURE_DIM}, masks=1x{SEGMENT_LENGTH}",
                "notes": "Prompt-conditioned prediction head with synthetic precomputed prompt features; num_heads=1.",
                "model": model,
                "batches": batches,
                "forward_fn": lambda batch, _model=model: _model(
                    batch["features"], masks=batch["masks"]
                ),
            }
        )
    return specs


def build_gvr_kin_specs(num_batches: int):
    gvr_module = import_gvr_module()
    specs = []
    for prompt_index, (prompt_type, prompts) in enumerate(PROMPT_SETS.items()):
        prompt_features = deterministic_prompt_features(
            len(prompts), GVR_TEXT_DIM, offset=900 + 100 * prompt_index
        )
        model = gvr_module.GVRModulePredKinFeatures(
            gesture_prompts=prompts,
            d_text=GVR_TEXT_DIM,
            d_model=RESNET_FEATURE_DIM,
            num_heads=1,
            segment_length=SEGMENT_LENGTH,
            position=True,
            dropout=0.3,
            prompt_features=prompt_features,
            use_pooled_branch=False,
        )
        batches = make_kin_batches(seq_len=SEGMENT_LENGTH, num_batches=num_batches)
        specs.append(
            {
                "model_family": "GVR",
                "model_name": "GVRModulePredKinFeatures",
                "prompt_type": prompt_type,
                "feature_source": "resnet50+kinematics",
                "sequence_length": SEGMENT_LENGTH,
                "input_description": f"features=1x{SEGMENT_LENGTH}x{RESNET_FEATURE_DIM}, kine=1x{SEGMENT_LENGTH}x{KINEMATICS_DIM}, masks=1x{SEGMENT_LENGTH}",
                "notes": "Prompt-conditioned visual+kinematics head with synthetic precomputed prompt features; num_heads=1.",
                "model": model,
                "batches": batches,
                "forward_fn": lambda batch, _model=model: _model(
                    batch["features"], batch["kine"], masks=batch["masks"]
                ),
            }
        )
    return specs


def build_gvr_context_specs(num_batches: int):
    gvr_module = import_gvr_module()
    context_model = gvr_module.GVRModuleContexPredFeatures(
        d_model_ges=GVR_TEXT_DIM,
        d_model=RESNET_FEATURE_DIM,
        num_heads=1,
        segment_length=SEGMENT_LENGTH,
        layer_norm1=True,
        layer_norm2=True,
        layer_norm3=True,
        position=False,
        dropout=0.1,
    )
    dual_model = gvr_module.GVRModuleContexPredDualFeatures(
        base_dim=RESNET_FEATURE_DIM,
        ges_dim=RESNET_FEATURE_DIM,
        d_embed=RESNET_FEATURE_DIM,
        num_heads=1,
        segment_length=SEGMENT_LENGTH,
        layer_norm1=False,
        layer_norm2=True,
        layer_norm3=False,
        position=True,
        dropout=0.3,
    )
    return [
        {
            "model_family": "GVR",
            "model_name": "GVRModuleContexPredFeatures",
            "prompt_type": "n/a",
            "feature_source": "resnet50",
            "sequence_length": SEGMENT_LENGTH,
            "input_description": f"1x{SEGMENT_LENGTH}x{RESNET_FEATURE_DIM}",
            "notes": "Single-stream context model; no prompt type axis; num_heads=1.",
            "model": context_model,
            "batches": make_plain_feature_batches(
                seq_len=SEGMENT_LENGTH,
                feature_dim=RESNET_FEATURE_DIM,
                num_batches=num_batches,
                seed_offset=1200,
            ),
            "forward_fn": lambda batch, _model=context_model: _model(batch["features"]),
        },
        {
            "model_family": "GVR",
            "model_name": "GVRModuleContexPredDualFeatures",
            "prompt_type": "n/a",
            "feature_source": "resnet50+gesture_prompt_resnet50",
            "sequence_length": SEGMENT_LENGTH,
            "input_description": f"base=1x{SEGMENT_LENGTH}x{RESNET_FEATURE_DIM}, ges=1x{SEGMENT_LENGTH}x{RESNET_FEATURE_DIM}, masks=1x{SEGMENT_LENGTH}",
            "notes": "Dual-stream context model using 2048-d resnet gesture prompt features inferred from existing extracted data; num_heads=1.",
            "model": dual_model,
            "batches": make_dual_feature_batches(seq_len=SEGMENT_LENGTH, num_batches=num_batches),
            "forward_fn": lambda batch, _model=dual_model: _model(
                batch["base_features"],
                batch["ges_features"],
                masks=batch["masks"],
            ),
        },
    ]


def collect_model_specs(device: torch.device, num_batches: int) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [build_sedmamba_spec(device, num_batches), build_cog_spec(device, num_batches)]
    specs.extend(build_gvr_pred_specs(num_batches))
    specs.extend(build_gvr_kin_specs(num_batches))
    specs.extend(build_gvr_context_specs(num_batches))
    return specs


def pretty_print_results(results: Sequence[BenchmarkResult]) -> None:
    headers = (
        "family",
        "model",
        "prompt_type",
        "params",
        "flops",
        "latency_ms",
    )
    widths = [12, 30, 22, 14, 16, 14]
    print(
        f"{headers[0]:<{widths[0]}} {headers[1]:<{widths[1]}} "
        f"{headers[2]:<{widths[2]}} {headers[3]:>{widths[3]}} "
        f"{headers[4]:>{widths[4]}} {headers[5]:>{widths[5]}}"
    )
    print("-" * (sum(widths) + 5))
    for row in results:
        print(
            f"{row.model_family:<{widths[0]}} "
            f"{row.model_name:<{widths[1]}} "
            f"{row.prompt_type:<{widths[2]}} "
            f"{row.non_encoder_params:>{widths[3]}} "
            f"{row.flops_per_window:>{widths[4]}} "
            f"{row.latency_mean_ms:>{widths[5]}.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark non-encoder params, FLOPs, and latency for SEDMamba, CoG, and GVR feature models."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Benchmark device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--dummy_batches",
        type=int,
        default=DEFAULT_DUMMY_BATCHES,
        help="Number of distinct dummy batches to cycle through during timing.",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=DEFAULT_WARMUP_ITERS,
        help="Number of warmup forwards per model before timing.",
    )
    parser.add_argument(
        "--timing_iters",
        type=int,
        default=DEFAULT_TIMING_ITERS,
        help="Number of timed forwards per model.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=REPO_ROOT / "profile_reports" / "window_model_complexity_len10.csv",
        help="CSV path for benchmark results.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=REPO_ROOT / "profile_reports" / "window_model_complexity_len10.json",
        help="JSON path for benchmark results.",
    )
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)
    ensure_parent_dir(args.output_csv)
    ensure_parent_dir(args.output_json)

    results: List[BenchmarkResult] = []
    model_specs = collect_model_specs(device, args.dummy_batches)
    print(f"[INFO] Benchmark device: {device}")
    print(f"[INFO] Segment/window length: {SEGMENT_LENGTH}")
    print(f"[INFO] Dummy batches per model: {args.dummy_batches}")
    print(f"[INFO] Warmup iterations: {args.warmup_iters}")
    print(f"[INFO] Timing iterations: {args.timing_iters}")

    for spec in model_specs:
        print(f"[INFO] Benchmarking {spec['model_family']} / {spec['model_name']} / {spec['prompt_type']}")
        output_shape, flops, latency_mean, latency_std = benchmark_model(
            model=spec["model"],
            forward_fn=spec["forward_fn"],
            batches=spec["batches"],
            device=device,
            warmup_iters=args.warmup_iters,
            timing_iters=args.timing_iters,
        )
        results.append(
            BenchmarkResult(
                model_family=spec["model_family"],
                model_name=spec["model_name"],
                prompt_type=spec["prompt_type"],
                feature_source=spec["feature_source"],
                sequence_length=spec["sequence_length"],
                input_description=spec["input_description"],
                output_shape=output_shape,
                non_encoder_params=total_params(spec["model"]),
                trainable_params=trainable_params(spec["model"]),
                flops_per_window=flops,
                latency_mean_ms=latency_mean,
                latency_std_ms=latency_std,
                device=str(device),
                notes=spec["notes"],
            )
        )

    csv_rows = [asdict(row) for row in results]
    with args.output_csv.open("w", encoding="utf-8") as handle:
        headers = list(csv_rows[0].keys())
        handle.write(",".join(headers) + "\n")
        for row in csv_rows:
            values = []
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
        "segment_length": SEGMENT_LENGTH,
        "dummy_batches": args.dummy_batches,
        "warmup_iters": args.warmup_iters,
        "timing_iters": args.timing_iters,
        "results": csv_rows,
    }
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    pretty_print_results(results)
    print(f"\n[INFO] Wrote CSV:  {args.output_csv}")
    print(f"[INFO] Wrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
