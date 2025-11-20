import torch
import os
import comfy.utils
import folder_paths
import numpy as np
import math
import cv2
import time
import PIL.Image
from .resampler import Resampler
from .CrossAttentionPatch import Attn2Replace, instantid_attention
from .utils import tensor_to_image

from insightface.app import FaceAnalysis

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

import torch.nn.functional as F

MODELS_DIR = os.path.join(folder_paths.models_dir, "instantid")

if "instantid" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["instantid"]

folder_paths.folder_names_and_paths["instantid"] = (current_paths, folder_paths.supported_pt_extensions)

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    h, w, _ = image_pil.shape
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

class InstantID(torch.nn.Module):
    def __init__(self, instantid_model, cross_attention_dim=1280, output_cross_attention_dim=1024, clip_embeddings_dim=512, clip_extra_context_tokens=16):
        super().__init__()

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens

        self.image_proj_model = self.init_proj()

        self.image_proj_model.load_state_dict(instantid_model["image_proj"])
        self.ip_layers = To_KV(instantid_model["ip_adapter"])

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.clip_extra_context_tokens,
            embedding_dim=self.clip_embeddings_dim,
            output_dim=self.output_cross_attention_dim,
            ff_mult=4
        )
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        #image_prompt_embeds = clip_embed.clone().detach()
        image_prompt_embeds = self.image_proj_model(clip_embed)
        #uncond_image_prompt_embeds = clip_embed_zeroed.clone().detach()
        uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)

        return image_prompt_embeds, uncond_image_prompt_embeds

class ImageProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class To_KV(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.to_kvs = torch.nn.ModuleDict()
        for key, value in state_dict.items():
            k = key.replace(".weight", "").replace(".", "_")
            self.to_kvs[k] = torch.nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[k].weight.data = value

def _set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"].copy()
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    else:
        to["patches_replace"]["attn2"] = to["patches_replace"]["attn2"].copy()

    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = Attn2Replace(instantid_attention, **patch_kwargs)
        model.model_options["transformer_options"] = to
    else:
        to["patches_replace"]["attn2"][key].add(instantid_attention, **patch_kwargs)

class InstantIDModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "instantid_file": (folder_paths.get_filename_list("instantid"), )}}

    RETURN_TYPES = ("INSTANTID",)
    FUNCTION = "load_model"
    CATEGORY = "sunxAI_facetools"

    def load_model(self, instantid_file):
        ckpt_path = folder_paths.get_full_path("instantid", instantid_file)

        model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model

        model = InstantID(
            model,
            cross_attention_dim=1280,
            output_cross_attention_dim=model["ip_adapter"]["1.to_k_ip.weight"].shape[1],
            clip_embeddings_dim=512,
            clip_extra_context_tokens=16,
        )

        return (model,)

def extractFeatures(insightface, image, extract_kps=False):
    face_img = tensor_to_image(image)
    out = []

    insightface.det_model.input_size = (640,640) # reset the detection size

    for i in range(face_img.shape[0]):
        for size in [(size, size) for size in range(640, 128, -64)]:
            insightface.det_model.input_size = size # TODO: hacky but seems to be working
            face = insightface.get(face_img[i])
            if face:
                face = sorted(face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]

                if extract_kps:
                    out.append(draw_kps(face_img[i], face['kps']))
                else:
                    out.append(torch.from_numpy(face['embedding']).unsqueeze(0))

                if 640 not in size:
                    print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
                break

    if out:
        if extract_kps:
            out = torch.stack(T.ToTensor()(out), dim=0).permute([0,2,3,1])
        else:
            out = torch.stack(out, dim=0)
    else:
        out = None

    return out

class InstantIDFaceAnalysis:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM", "CoreML"], ),
            },
        }

    RETURN_TYPES = ("FACEANALYSIS",)
    FUNCTION = "load_insight_face"
    CATEGORY = "sunxAI_facetools"

    def load_insight_face(self, provider):
        model = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',]) # alternative to buffalo_l
        model.prepare(ctx_id=0, det_size=(640, 640))

        return (model,)


def add_noise(image, factor):
    seed = int(torch.sum(image).item()) % 1000000007
    torch.manual_seed(seed)
    mask = (torch.rand_like(image) < factor).float()
    noise = torch.rand_like(image)
    noise = torch.zeros_like(image) * (1-mask) + noise * mask

    return factor*noise

class ApplyInstantID:
    """
    Main node for applying InstantID
    Function: Apply face identity information to the model to achieve face swap effect

    Parameter description:
    - image: Reference face image for extracting face feature embeddings
    - image_kps: Optional keypoint image for generating ControlNet control signals
      * If provided: directly use this image to extract face keypoints
      * If not provided: use the first image from the reference image (image) to extract keypoints
    - face_embed: Optional pre-computed face embedding, if provided skips face detection and feature extraction
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "instantid": ("INSTANTID", ),  # InstantID model
                "insightface": ("FACEANALYSIS", ),  # InsightFace face analysis model
                "control_net": ("CONTROL_NET", ),  # ControlNet model
                "image": ("IMAGE", ),  # Reference face image
                "model": ("MODEL", ),  # Base diffusion model
                "positive": ("CONDITIONING", ),  # Positive conditioning
                "negative": ("CONDITIONING", ),  # Negative conditioning
                "weight": ("FLOAT", {"default": .8, "min": 0.0, "max": 5.0, "step": 0.01, }),  # Weight
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, }),  # Start timestep
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, }),  # End timestep
            },
            "optional": {
                "image_kps": ("IMAGE",),  # Optional: keypoint image for ControlNet
                "mask": ("MASK",),  # Optional: mask
                "face_embed": ("FACE_EMBEDS",),  # Optional: pre-computed face embedding
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "FACE_EMBEDS", "BOOLEAN")
    RETURN_NAMES = ("MODEL", "positive", "negative", "face_embed", "has_face")
    FUNCTION = "apply_instantid"
    CATEGORY = "sunxAI_facetools"

    def apply_instantid(self, instantid, insightface, control_net, image, model, positive, negative, start_at, end_at, weight=.8, ip_weight=None, cn_strength=None, noise=0.35, image_kps=None, mask=None, combine_embeds='average', face_embed=None):
        """
        Apply InstantID face swap effect

        Processing flow:
        1. Face embedding processing:
           - If face_embed is provided: directly use pre-computed embedding, skip face detection
           - If not provided: detect faces from image and extract feature embeddings

        2. Keypoint processing:
           - If image_kps is provided: use this image to extract face keypoints for ControlNet
           - If not provided: use the first image from reference image (image) to extract keypoints

        3. Model patching: Inject face embeddings into the model's attention layers
        4. ControlNet application: Use face keypoints to control the generation process
        """
        # If end_at is 0, return original data directly, skip all processing
        if end_at == 0 or weight == 0:
            print(f"\033[33mINFO: end_at=0 or weight=0, skipping InstantID processing\033[0m")
            return (model, positive, negative, None, False)


        start_total = time.time()
        print(f"\033[36m=== InstantID processing started ===\033[0m")

        # Set data type and device
        start_setup = time.time()
        dtype = comfy.model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set weight parameters
        ip_weight = weight if ip_weight is None else ip_weight  # IP-Adapter weight
        cn_strength = weight if cn_strength is None else cn_strength  # ControlNet strength
        print(f"\033[36mInitialization setup time: {time.time() - start_setup:.3f}s\033[0m")

        # === Face embedding processing ===
        start_embed = time.time()
        # If pre-computed face_embed is provided, use it directly; otherwise extract from image
        if face_embed is not None:
            # Check if it's a no-face marker
            if face_embed.get('no_face', False):
                print(f"\033[33mINFO: Using pre-computed no-face marker, skipping InstantID processing\033[0m")
                return (model, positive, negative, face_embed, False)

            print(f"\033[32mINFO: Using pre-computed face embedding (already on GPU)\033[0m")
            # Data already loaded to GPU in LoadFaceEmbeds, use directly
            image_prompt_embeds = face_embed['cond']
            uncond_image_prompt_embeds = face_embed['uncond']
            output_face_embed = None  # Already have embed, no need to output new one
        else:
            print(f"\033[32mINFO: Extracting face features from reference image\033[0m")
            # Extract face features from reference image
            start_face_detect = time.time()
            face_embed_raw = extractFeatures(insightface, image)
            print(f"\033[36mFace detection time: {time.time() - start_face_detect:.3f}s\033[0m")

            if face_embed_raw is None:
                print(f"\033[33mWARNING: No face detected in reference image, creating no-face marker and returning\033[0m")
                # Create no-face marker to avoid repeated detection next time
                no_face_embed = {
                    'no_face': True,
                    'timestamp': int(time.time())
                }
                return (model, positive, negative, no_face_embed, False)

            start_embed_process = time.time()
            clip_embed = face_embed_raw
            # InstantID works better with average embedding
            if clip_embed.shape[0] > 1:
                if combine_embeds == 'average':
                    clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)
                elif combine_embeds == 'norm average':
                    clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=0, keepdim=True), dim=0).unsqueeze(0)

            # Add noise to negative embedding
            if noise > 0:
                seed = int(torch.sum(clip_embed).item()) % 1000000007
                torch.manual_seed(seed)
                clip_embed_zeroed = noise * torch.rand_like(clip_embed)
            else:
                clip_embed_zeroed = torch.zeros_like(clip_embed)
            print(f"\033[36mEmbedding preprocessing time: {time.time() - start_embed_process:.3f}s\033[0m")

            # Process embedding with InstantID model
            start_instantid = time.time()
            instantid_model = instantid
            instantid_model.to(device, dtype=dtype)

            image_prompt_embeds, uncond_image_prompt_embeds = instantid_model.get_image_embeds(clip_embed.to(device, dtype=dtype), clip_embed_zeroed.to(device, dtype=dtype))

            image_prompt_embeds = image_prompt_embeds.to(device, dtype=dtype)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(device, dtype=dtype)
            print(f"\033[36mInstantID model processing time: {time.time() - start_instantid:.3f}s\033[0m")

            # Save generated face_embed for next use
            output_face_embed = { "cond": image_prompt_embeds, "uncond": uncond_image_prompt_embeds }

        print(f"\033[36mTotal face embedding time: {time.time() - start_embed:.3f}s\033[0m")

        # === Keypoint processing ===
        start_kps = time.time()
        # If keypoint image is not provided, use the first image from reference image to extract keypoints
        # image_kps is used for ControlNet to control generated face pose and expression
        if image_kps is not None:
            print(f"\033[32mINFO: Using provided keypoint image\033[0m")
            face_kps = extractFeatures(insightface, image_kps, extract_kps=True)
        else:
            print(f"\033[32mINFO: Extracting keypoints from reference image\033[0m")
            face_kps = extractFeatures(insightface, image[0].unsqueeze(0), extract_kps=True)

        # If keypoint extraction fails, use zero tensor as placeholder
        if face_kps is None:
            face_kps = torch.zeros_like(image) if image_kps is None else image_kps
            print(f"\033[33mWARNING: No face detected in keypoint image, may affect control effect\033[0m")
        print(f"\033[36mKeypoint extraction time: {time.time() - start_kps:.3f}s\033[0m")

        # === Model patching ===
        start_patch = time.time()
        # Clone model to avoid affecting original model
        work_model = model.clone()

        # Calculate timestep range
        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        if mask is not None:
            mask = mask.to(device)

        # Prepare patching parameters
        patch_kwargs = {
            "ipadapter": instantid,
            "weight": ip_weight,
            "cond": image_prompt_embeds,
            "uncond": uncond_image_prompt_embeds,
            "mask": mask,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
        }

        # Patch model's attention layers
        number = 0
        # Input blocks
        for id in [4,5,7,8]:
            block_indices = range(2) if id in [4, 5] else range(10)
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                _set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                number += 1
        # Output blocks
        for id in range(6):
            block_indices = range(2) if id in [3, 4, 5] else range(10)
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                _set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                number += 1
        # Middle blocks
        for index in range(10):
            patch_kwargs["module_key"] = str(number*2+1)
            _set_model_patch_replace(work_model, patch_kwargs, ("middle", 1, index))
            number += 1
        print(f"\033[36mModel patching time: {time.time() - start_patch:.3f}s\033[0m")

        # === ControlNet application ===
        start_controlnet = time.time()
        # Handle mask dimensions
        if mask is not None and len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        cnets = {}
        cond_uncond = []

        # Apply ControlNet to positive and negative conditioning separately
        is_cond = True
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                # Get or create ControlNet
                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    # Use face keypoints as ControlNet control signal
                    c_net = control_net.copy().set_cond_hint(face_kps.movedim(-1,1), cn_strength, (start_at, end_at))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                # Set cross-attention ControlNet embedding
                d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype) if is_cond else uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype)

                # Apply mask (only for positive conditioning)
                if mask is not None and is_cond:
                    d['mask'] = mask
                    d['set_area_to_bounds'] = False

                n = [t[0], d]
                c.append(n)
            cond_uncond.append(c)
            is_cond = False
        print(f"\033[36mControlNet application time: {time.time() - start_controlnet:.3f}s\033[0m")

        total_time = time.time() - start_total
        print(f"\033[36m=== InstantID processing completed, total time: {total_time:.3f}s ===\033[0m")

        # Optimization suggestion
        if output_face_embed is not None:
            print(f"\033[33m💡 Optimization tip: Saving generated face_embed can save {time.time() - start_embed:.3f}s of face processing time\033[0m")

        return(work_model, cond_uncond[0], cond_uncond[1], output_face_embed , True)


class SaveFaceEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_embed": ("FACE_EMBEDS",),
                "name": ("STRING", {"default": "face_embed"}),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_face_embed"
    CATEGORY = "sunxAI_facetools"
    OUTPUT_NODE = True

    def save_face_embed(self, face_embed, name):
        # Check if face_embed is None or empty
        if face_embed is None:
            print(f"\033[33mWARNING: Face embed is None, skipping save.\033[0m")
            return {}

        # Create save directory
        save_dir = os.path.join(folder_paths.models_dir, "face_embeds")
        os.makedirs(save_dir, exist_ok=True)

        # Ensure filename ends with .pt
        filename = name + '.pt'
        filepath = os.path.join(save_dir, filename)

        # Check if file already exists
        if os.path.exists(filepath):
            print(f"\033[33mWARNING: File {filepath} already exists, skipping save.\033[0m")
            return {}

        # Handle different types of face_embed data
        save_data = {}

        # Check if it's a no-face marker
        if face_embed.get('no_face', False):
            print(f"\033[32mINFO: Saving no-face marker to {filepath}\033[0m")
            save_data = {
                "no_face": True,
                "timestamp": face_embed.get('timestamp', int(time.time()))
            }
        else:
            # Normal face embedding data - save to disk, immediately release memory
            if 'cond' in face_embed and 'uncond' in face_embed:
                print(f"\033[32mINFO: Saving face embeddings to {filepath}\033[0m")
                save_data = {
                    "cond": face_embed["cond"].cpu(),
                    "uncond": face_embed["uncond"].cpu(),
                    "timestamp": int(time.time())
                }
            else:
                print(f"\033[33mWARNING: Invalid face_embed format, missing 'cond' or 'uncond' fields\033[0m")
                return {}

        # Save to disk
        torch.save(save_data, filepath)

        # Immediately clear memory
        del save_data
        print(f"\033[32mINFO: Face embed data saved successfully and memory cleared\033[0m")

        return {}

class LoadFaceEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": ("STRING", {"default": "face_embed"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("FACE_EMBEDS",)
    RETURN_NAMES = ("face_embed",)
    FUNCTION = "load_face_embed"
    CATEGORY = "sunxAI_facetools"

    def load_face_embed(self, name, seed=0):
        face_embeds_dir = os.path.join(folder_paths.models_dir, "face_embeds")

        # Ensure filename ends with .pt
        filename = name + '.pt'
        filepath = os.path.join(face_embeds_dir, filename)

        if not os.path.exists(filepath):
            print(f"\033[33mWARNING: Face embed file not found: {filepath}, returning None\033[0m")
            return (None,)

        try:
            print(f"\033[36mINFO: Loading face embed (reload={seed}): {filename}\033[0m")

            # Load face embedding data to CPU
            save_data = torch.load(filepath, map_location="cpu")

            # Check if it's a no-face marker
            if save_data.get('no_face', False):
                print(f"\033[32mINFO: Loaded no-face marker from {filepath}\033[0m")
                face_embed = {
                    "no_face": True,
                    "timestamp": save_data.get('timestamp', 0)
                }
                del save_data
                return (face_embed,)

            # Normal face embedding data - directly load to GPU
            if 'cond' in save_data and 'uncond' in save_data:
                print(f"\033[32mINFO: Loading face embeddings to GPU from {filepath}\033[0m")

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32

                face_embed = {
                    "cond": save_data["cond"].to(device, dtype=dtype),
                    "uncond": save_data["uncond"].to(device, dtype=dtype),
                    "timestamp": save_data.get('timestamp', 0)
                }

                # Clear CPU data
                del save_data
                print(f"\033[32mINFO: Face embeddings loaded to {device} and CPU cache cleared\033[0m")

                return (face_embed,)
            else:
                print(f"\033[33mWARNING: Invalid saved data format, missing 'cond' or 'uncond' fields\033[0m")
                del save_data
                return (None,)

        except Exception as e:
            print(f"\033[31mERROR: Failed to load face embed from {filepath}: {e}\033[0m")
            return (None,)



