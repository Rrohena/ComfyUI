"""
IP-Adapter Lite — ComfyUI custom node
======================================
A simplified, memory-efficient IP-Adapter node targeting SD1.5 on ≤8 GB VRAM.

Design choices vs. the full comfyui_ipadapter_plus:
  • fp16 throughout (weights, embeddings, projections).
  • Supports only the *standard* (non-Plus, non-FaceID) IP-Adapter SD1.5 model.
  • No insightface / AnimateDiff / mask / composition image inputs.
  • ImageProjModel (simple linear) instead of the heavier Resampler.
  • ImageProjModel is freed from GPU immediately after embedding computation.
  • One weight/scale slider, linear scheduling only.

Nodes exposed:
  IPAdapterLiteModelLoader  — loads an .safetensors/.bin IPA weights file.
  IPAdapterLiteApply        — patches a MODEL clone with IP-Adapter conditioning.
"""

import math
import os
import torch

import folder_paths
import comfy.utils
import comfy.model_management as model_management
from comfy.ldm.modules.attention import optimized_attention
from comfy.clip_vision import clip_preprocess

from .models import ImageProjModel, IPToKV


# ---------------------------------------------------------------------------
# Register the "ipadapter" folder so the file-list widget is populated.
# (The existing comfyui_ipadapter_plus node does the same thing.)
# ---------------------------------------------------------------------------
if "ipadapter" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["ipadapter"] = (
        [os.path.join(folder_paths.models_dir, "ipadapter")],
        folder_paths.supported_pt_extensions,
    )

# SD1.5 UNet blocks that contain a cross-attention (attn2) layer.
# Order must match the module_key numbering used in the IP-Adapter weights.
_SD15_INPUT_BLOCKS = [1, 2, 4, 5, 7, 8]          # 6 blocks
_SD15_OUTPUT_BLOCKS = [3, 4, 5, 6, 7, 8, 9, 10, 11]  # 9 blocks
# plus 1 middle block  →  16 total, module keys 1, 3, 5, …, 31


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_weights(path: str) -> dict:
    """
    Load an IP-Adapter weights file (.safetensors or .bin) and return a dict
    with "image_proj" and "ip_adapter" sub-dicts, matching the format used
    by the reference implementation.
    """
    raw = comfy.utils.load_torch_file(path, safe_load=True)

    if path.lower().endswith(".safetensors"):
        weights = {"image_proj": {}, "ip_adapter": {}}
        for key, tensor in raw.items():
            if key.startswith("image_proj."):
                weights["image_proj"][key[len("image_proj."):]] = tensor
            elif key.startswith("ip_adapter."):
                weights["ip_adapter"][key[len("ip_adapter."):]] = tensor
    else:
        # .bin files already carry the nested structure
        weights = raw

    if not weights.get("ip_adapter"):
        raise ValueError(
            f"'{os.path.basename(path)}' does not look like a valid IP-Adapter "
            "weights file (missing 'ip_adapter' key)."
        )
    return weights


def _encode_image(clip_vision, image: torch.Tensor) -> torch.Tensor:
    """
    Run the CLIP vision encoder on *image* and return the pooled image embedding.

    image: [B, H, W, C] float32 in [0, 1]  (ComfyUI convention)
    returns: [B, clip_dim] on intermediate_device
    """
    model_management.load_model_gpu(clip_vision.patcher)
    pixel_values = clip_preprocess(image.to(clip_vision.load_device)).float()
    # intermediate_output=-2  →  (last_hidden, penultimate_hidden, image_embeds)
    out = clip_vision.model(pixel_values=pixel_values, intermediate_output=-2)
    embeds = out[2].to(model_management.intermediate_device())  # pooled embedding
    del pixel_values, out
    torch.cuda.empty_cache()
    return embeds


def _set_patch(model, key: tuple, patch) -> None:
    """
    Install *patch* into model.model_options["transformer_options"]["patches_replace"]["attn2"][key].

    All intermediate dicts are shallow-copied so the original model's options
    are never modified.
    """
    to = model.model_options.get("transformer_options", {}).copy()
    patches_replace = to.get("patches_replace", {}).copy()
    attn2 = patches_replace.get("attn2", {}).copy()
    attn2[key] = patch
    patches_replace["attn2"] = attn2
    to["patches_replace"] = patches_replace
    model.model_options["transformer_options"] = to


# ---------------------------------------------------------------------------
# Attention patch
# ---------------------------------------------------------------------------

class _IPALiteAttnPatch:
    """
    Replaces a single cross-attention layer with:
        out = base_attention(q, k, v) + ip_attention(q, ip_k, ip_v) * weight

    ComfyUI calls this as patch(q, k, v, extra_options) and expects the
    *full* attention output (not just a delta), because it is a
    "patches_replace" entry, not a "patches" entry.
    """

    def __init__(
        self,
        ip_to_kv: IPToKV,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        weight: float,
        sigma_start: float,
        sigma_end: float,
        module_key: str,
    ):
        self.ip_to_kv = ip_to_kv
        self.cond = cond        # [1, num_tokens, cross_attn_dim]
        self.uncond = uncond    # [1, num_tokens, cross_attn_dim]
        self.weight = weight
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.module_key = module_key  # e.g. "1", "3", …, "31"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        extra_options: dict,
    ) -> torch.Tensor:
        dtype = q.dtype

        # ── base cross-attention (text conditioning) ──────────────────────
        base_out = optimized_attention(q, k, v, extra_options["n_heads"])

        # ── sigma-range gating ────────────────────────────────────────────
        if "sigmas" in extra_options:
            sigma = extra_options["sigmas"].detach().cpu()[0].item()
        else:
            sigma = 999_999_999.0

        if sigma > self.sigma_start or sigma < self.sigma_end:
            return base_out.to(dtype=dtype)

        # ── build IP-Adapter K, V for this layer ──────────────────────────
        cond_or_uncond = extra_options["cond_or_uncond"]
        batch_prompt = q.shape[0] // len(cond_or_uncond)
        device = q.device

        k_key = self.module_key + "_to_k_ip"
        v_key = self.module_key + "_to_v_ip"

        # Project in the weight dtype (fp16), then cast to q.dtype for attention.
        ip_dtype = next(iter(self.ip_to_kv.parameters())).dtype
        cond_d = self.cond.to(device, dtype=ip_dtype)
        uncond_d = self.uncond.to(device, dtype=ip_dtype)

        # Repeat along batch dimension to match the number of image prompts.
        k_cond = self.ip_to_kv.layers[k_key](cond_d).to(dtype).repeat(batch_prompt, 1, 1)
        k_uncond = self.ip_to_kv.layers[k_key](uncond_d).to(dtype).repeat(batch_prompt, 1, 1)
        v_cond = self.ip_to_kv.layers[v_key](cond_d).to(dtype).repeat(batch_prompt, 1, 1)
        v_uncond = self.ip_to_kv.layers[v_key](uncond_d).to(dtype).repeat(batch_prompt, 1, 1)

        # Interleave cond / uncond to match the batch ordering used by ComfyUI
        # (cond_or_uncond is a list of 0s and 1s: 0 = cond, 1 = uncond).
        ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0)
        ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0)

        ip_out = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])

        return (base_out + ip_out * self.weight).to(dtype=dtype)


# ---------------------------------------------------------------------------
# Node: loader
# ---------------------------------------------------------------------------

class IPAdapterLiteModelLoader:
    """
    Load an IP-Adapter weights file from the models/ipadapter directory.

    Supports:
      • ip_adapter_sd15.safetensors  (standard, 16-layer)
      • ip_adapter_sd15.bin          (same, PyTorch format)

    Does NOT support:
      • IP-Adapter Plus / Plus Face (use Resampler, not ImageProjModel)
      • IP-Adapter FaceID (requires InsightFace embeddings)
      • SDXL variants
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ipadapter_file": (folder_paths.get_filename_list("ipadapter"),),
            }
        }

    RETURN_TYPES = ("IPA_LITE_WEIGHTS",)
    RETURN_NAMES = ("ipa_weights",)
    FUNCTION = "load"
    CATEGORY = "ip_adapter_lite"

    def load(self, ipadapter_file: str):
        path = folder_paths.get_full_path("ipadapter", ipadapter_file)
        weights = _load_weights(path)

        image_proj = weights["image_proj"]

        # Reject unsupported variants early with a clear message.
        if "latents" in image_proj or "perceiver_resampler.proj_in.weight" in image_proj:
            raise ValueError(
                f"'{ipadapter_file}' is an IP-Adapter Plus model (uses Resampler). "
                "ip_adapter_lite only supports the standard ImageProjModel. "
                "Download 'ip_adapter_sd15.safetensors'."
            )
        if "proj.3.weight" in image_proj:
            raise ValueError(
                f"'{ipadapter_file}' is an IP-Adapter Full Face model. "
                "ip_adapter_lite does not support this variant."
            )
        if "0.to_q_lora.down.weight" in weights["ip_adapter"]:
            raise ValueError(
                f"'{ipadapter_file}' is an IP-Adapter FaceID model. "
                "ip_adapter_lite does not support FaceID."
            )
        if "proj.weight" not in image_proj:
            raise ValueError(
                f"'{ipadapter_file}': unrecognised image_proj format. "
                "Expected standard IP-Adapter SD1.5 weights."
            )

        cross_attn_dim = weights["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        if cross_attn_dim == 2048:
            raise ValueError(
                f"'{ipadapter_file}' is an SDXL IP-Adapter model. "
                "ip_adapter_lite targets SD1.5 only (cross_attention_dim=768)."
            )

        print(
            f"[ip_adapter_lite] Loaded '{ipadapter_file}' — "
            f"cross_attn_dim={cross_attn_dim}, "
            f"ip_adapter_layers={len([k for k in weights['ip_adapter'] if k.endswith('.to_k_ip.weight')])}"
        )
        return (weights,)


# ---------------------------------------------------------------------------
# Node: apply
# ---------------------------------------------------------------------------

class IPAdapterLiteApply:
    """
    Condition an SD1.5 MODEL with a reference image using IP-Adapter (fp16).

    Pipeline:
      1. Encode reference image with CLIP Vision → pooled image embedding.
      2. Project embedding through ImageProjModel → [1, num_tokens, 768].
      3. Build per-layer K,V projection layers (IPToKV) from IPA weights.
      4. Patch each cross-attention layer in the UNet clone with an
         _IPALiteAttnPatch that adds the IP-Adapter contribution.

    Memory notes (8 GB GPU):
      • All weights and activations are kept in fp16.
      • ImageProjModel is moved to GPU only for the projection step, then freed.
      • IPToKV layers (~19 MB fp16) stay on GPU during sampling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip_vision": ("CLIP_VISION",),
                "ipa_weights": ("IPA_LITE_WEIGHTS",),
                "image": ("IMAGE",),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "start_at": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_at": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "ip_adapter_lite"

    def apply(
        self,
        model,
        clip_vision,
        ipa_weights: dict,
        image: torch.Tensor,
        weight: float,
        start_at: float,
        end_at: float,
    ):
        device = model_management.get_torch_device()
        dtype = torch.float16

        # ── 1. Encode reference image ──────────────────────────────────────
        # Take first frame only if a video/batch is provided.
        ref_image = image[:1]
        img_embeds = _encode_image(clip_vision, ref_image)          # [1, clip_dim]
        uncond_embeds = torch.zeros_like(img_embeds)                # zero uncond

        # ── 2. Project through ImageProjModel ─────────────────────────────
        image_proj_sd = ipa_weights["image_proj"]
        cross_attn_dim = ipa_weights["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        proj_w = image_proj_sd["proj.weight"]
        clip_dim = proj_w.shape[1]
        num_tokens = proj_w.shape[0] // cross_attn_dim

        proj_model = ImageProjModel(cross_attn_dim, clip_dim, num_tokens)
        proj_model.load_state_dict(
            {k: v.to(dtype=dtype) for k, v in image_proj_sd.items()}
        )
        proj_model = proj_model.to(device, dtype=dtype)

        with torch.inference_mode():
            cond = proj_model(img_embeds.to(device, dtype=dtype))       # [1, T, D]
            uncond = proj_model(uncond_embeds.to(device, dtype=dtype))  # [1, T, D]

        # Free projection model immediately to reclaim VRAM.
        del proj_model
        torch.cuda.empty_cache()

        # ── 3. Build K,V projection layers ────────────────────────────────
        ip_to_kv = IPToKV(
            {k: v.to(dtype=dtype) for k, v in ipa_weights["ip_adapter"].items()}
        ).to(device, dtype=dtype)

        # ── 4. Compute sigma schedule ──────────────────────────────────────
        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_at)
        sigma_end = model_sampling.percent_to_sigma(end_at)

        # ── 5. Patch all SD1.5 cross-attention layers ─────────────────────
        patched = model.clone()
        number = 0

        def _make_patch(mk: str) -> _IPALiteAttnPatch:
            return _IPALiteAttnPatch(
                ip_to_kv=ip_to_kv,
                cond=cond,
                uncond=uncond,
                weight=weight,
                sigma_start=sigma_start,
                sigma_end=sigma_end,
                module_key=mk,
            )

        for block_id in _SD15_INPUT_BLOCKS:
            _set_patch(patched, ("input", block_id), _make_patch(str(number * 2 + 1)))
            number += 1

        for block_id in _SD15_OUTPUT_BLOCKS:
            _set_patch(patched, ("output", block_id), _make_patch(str(number * 2 + 1)))
            number += 1

        _set_patch(patched, ("middle", 0), _make_patch(str(number * 2 + 1)))

        print(
            f"[ip_adapter_lite] Patched {number + 1} attention layers "
            f"(weight={weight:.2f}, steps {start_at:.0%}–{end_at:.0%})."
        )
        return (patched,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "IPAdapterLiteModelLoader": IPAdapterLiteModelLoader,
    "IPAdapterLiteApply": IPAdapterLiteApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterLiteModelLoader": "Load IP-Adapter Lite Weights",
    "IPAdapterLiteApply": "Apply IP-Adapter Lite (SD1.5)",
}
