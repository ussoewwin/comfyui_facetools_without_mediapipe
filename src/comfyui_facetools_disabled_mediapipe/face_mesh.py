"""
Face Mesh Preprocessor using InsightFace (replacing MediaPipe)
Adapted from comfyui_controlnet_aux
"""

import os
import sys
import subprocess
import torch
import numpy as np
import cv2
from PIL import Image

# Import the common face mesh generation function
from .face_mesh_common import generate_annotation

# Simple cache for InsightFace app instances (shared with other modules)
_insightface_cache = {}


def install_package(package):
    """Install package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")


def pil2tensor(image):
    """Convert PIL Image to torch Tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(image):
    """Convert torch Tensor to PIL Image"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def resize_image_with_pad(input_image, target_size, upscale_method="INTER_CUBIC"):
    """
    Resize image to target size while maintaining aspect ratio, with padding.
    Returns (resized_image, remove_pad_function)
    """
    if isinstance(input_image, Image.Image):
        img_np = np.array(input_image.convert('RGB'))
    else:
        img_np = input_image
    
    h, w = img_np.shape[:2]
    
    # Calculate scaling factor
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize
    if upscale_method == "INTER_CUBIC":
        resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif upscale_method == "INTER_LINEAR":
        resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Pad to target size
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    padded = cv2.copyMakeBorder(resized, pad_h, target_size - new_h - pad_h,
                               pad_w, target_size - new_w - pad_w,
                               cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    def remove_pad(image):
        """Remove padding and resize back to original size"""
        if isinstance(image, Image.Image):
            img_np = np.array(image.convert('RGB'))
        else:
            img_np = image
        
        # Remove padding
        unpadded = img_np[pad_h:pad_h+new_h, pad_w:pad_w+new_w]
        
        # Resize back to original size
        restored = cv2.resize(unpadded, (w, h), interpolation=cv2.INTER_CUBIC)
        return restored
    
    return padded, remove_pad


def HWC3(x):
    """Ensure image has 3 channels"""
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W = x.shape[:2]
    if x.shape[2] == 1:
        x = np.concatenate([x, x, x], axis=2)
    elif x.shape[2] == 4:
        x = x[:, :, :3]
    assert x.shape[2] == 3
    return x


class FacetoolsFaceMesh:
    """
    Face Mesh Preprocessor using InsightFace (replacing MediaPipe)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_faces": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "min_confidence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect"
    CATEGORY = "facetools_disabled_mediapipe"
    
    def detect(self, image, max_faces=10, min_confidence=0.5, resolution=512):
        # Install dependencies if needed
        try:
            import insightface
        except ImportError:
            install_package("insightface")
            import insightface
        
        # Process each image in the batch
        results = []
        for img in image:
            # Convert tensor to PIL
            img_pil = tensor2pil(img.unsqueeze(0))
            
            # Convert PIL to numpy (RGB)
            img_np = np.array(img_pil.convert('RGB'))
            
            # Resize with padding
            resized_img, remove_pad = resize_image_with_pad(img_np, resolution, "INTER_CUBIC")
            
            # Generate annotation (face mesh)
            detected_map = generate_annotation(resized_img, max_faces, min_confidence)
            
            # Remove padding and restore original size
            detected_map = remove_pad(detected_map)
            
            # Ensure 3 channels
            detected_map = HWC3(detected_map)
            
            # Convert back to tensor
            detected_tensor = pil2tensor(Image.fromarray(detected_map))
            results.append(detected_tensor)
        
        return (torch.cat(results, dim=0),)


NODE_CLASS_MAPPINGS = {
    "FacetoolsFaceMesh": FacetoolsFaceMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FacetoolsFaceMesh": "Facetools Face Mesh (InsightFace)",
}

