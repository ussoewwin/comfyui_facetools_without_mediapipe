"""
InsightFace-based segmentation nodes (replacing MediaPipe)
Adapted from comfyui-easy-use and ComfyUI_LayerStyle_Advance
"""

import os
import sys
import subprocess
import torch
import numpy as np
import cv2
from PIL import Image
import folder_paths

def install_package(package):
    """Install package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")

# Simple cache for InsightFace app instances
_insightface_cache = {}

def pil2tensor(image):
    """Convert PIL Image to torch Tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    """Convert torch Tensor to PIL Image"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def RGB2RGBA(image: Image, mask: Image) -> Image:
    """Convert RGB image with mask to RGBA"""
    (R, G, B) = image.convert('RGB').split()
    return Image.merge('RGBA', (R, G, B, mask.convert('L')))

def image2mask(image: Image) -> torch.Tensor:
    """Convert PIL Image to mask tensor"""
    if image.mode == 'L':
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])
    else:
        image = image.convert('RGB').split()[0]
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])

def mask2image(mask: torch.Tensor) -> Image:
    """Convert mask tensor to PIL Image"""
    mask_np = np.clip(255.0 * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    _mask = Image.fromarray(mask_np).convert("L")
    _image = Image.new("RGBA", _mask.size, color='white')
    _image = Image.composite(
        _image, Image.new("RGBA", _mask.size, color='black'), _mask)
    return _image

def pil2cv2(pil_img: Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format (BGR)"""
    img_array = np.array(pil_img.convert('RGB'))
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

def cv22pil(cv2_img: np.ndarray) -> Image:
    """Convert OpenCV format (BGR) to PIL Image"""
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

def gaussian_blur(image: Image, radius: int) -> Image:
    """Apply Gaussian blur to PIL Image"""
    from PIL import ImageFilter
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


class FacetoolsHumanSegmentation:
    """
    Human segmentation using InsightFace (replacing MediaPipe)
    Adapted from comfyui-easy-use's humanSegmentation node
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "confidence": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 0.95, "step": 0.01},),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "segment"
    CATEGORY = "facetools_disabled_mediapipe"

    def segment(self, image, confidence):
        try:
            import insightface
        except:
            install_package("insightface")
            install_package("opencv-python")
            import insightface
            import cv2

        # Initialize InsightFace app for person detection
        cache_key = "facetools_insightface_app"
        if cache_key in _insightface_cache:
            app = _insightface_cache[cache_key]
        else:
            app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            _insightface_cache[cache_key] = app

        ret_images = []
        ret_masks = []

        for img in image:
            _image = torch.unsqueeze(img, 0)
            orig_image = tensor2pil(_image).convert('RGB')
            # Convert the Tensor to a PIL image
            i = 255. * img.cpu().numpy()
            image_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Convert PIL to numpy array (BGR for OpenCV)
            img_np = np.array(image_pil)
            if img_np.shape[-1] == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Detect faces/persons using InsightFace
            faces = app.get(img_np)
            
            # Create mask from detected bounding boxes
            h, w = img_np.shape[:2]
            mask_array = np.zeros((h, w), dtype=np.float32)
            
            if len(faces) > 0:
                # Use all detected faces to create a combined mask
                for face in faces:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    # Expand bounding box slightly for better coverage
                    x1 = max(0, x1 - int((x2 - x1) * 0.1))
                    y1 = max(0, y1 - int((y2 - y1) * 0.1))
                    x2 = min(w, x2 + int((x2 - x1) * 0.1))
                    y2 = min(h, y2 + int((y2 - y1) * 0.1))
                    # Create mask for this detection
                    mask_array[y1:y2, x1:x2] = 1.0
            
            # Apply confidence threshold
            mask_array = (mask_array > confidence).astype(np.float32)
            
            # Convert mask to PIL Image
            mask_pil = Image.fromarray((mask_array * 255).astype(np.uint8), mode='L')
            
            # Create RGBA image with mask
            ret_image = RGB2RGBA(orig_image, mask_pil)
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(mask_pil))

        output_image = torch.cat(ret_images, dim=0) if ret_images else torch.zeros((1, 1, 1, 4))
        mask = torch.cat(ret_masks, dim=0) if ret_masks else torch.zeros((1, 1, 1))

        return (output_image, mask)


class FacetoolsPersonMaskUltraV2:
    """
    Person mask generation using InsightFace (replacing MediaPipe)
    Adapted from ComfyUI_LayerStyle_Advance's PersonMaskUltraV2 node
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter']
        device_list = ['cuda', 'cpu']
        return {
            "required": {
                "images": ("IMAGE",),
                "face": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "hair": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "body": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "clothes": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "accessories": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "background": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "confidence": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 0.95, "step": 0.01},),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = 'person_mask_ultra_v2'
    CATEGORY = 'facetools_disabled_mediapipe'

    def person_mask_ultra_v2(self, images, face, hair, body, clothes,
                            accessories, background, confidence,
                            detail_method, detail_erode, detail_dilate,
                            black_point, white_point, process_detail, device, max_megapixels):

        try:
            import insightface
        except:
            install_package("insightface")
            install_package("opencv-python")
            import insightface

        # Initialize InsightFace app for person detection
        cache_key = "facetools_insightface_app"
        if cache_key in _insightface_cache:
            app = _insightface_cache[cache_key]
        else:
            app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            _insightface_cache[cache_key] = app

        ret_images = []
        ret_masks = []

        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        for image in images:
            _image = torch.unsqueeze(image, 0)
            orig_image = tensor2pil(_image).convert('RGB')
            # Convert the Tensor to a PIL image
            i = 255. * image.cpu().numpy()
            image_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Convert PIL to numpy array (BGR for OpenCV)
            img_np = np.array(image_pil)
            if img_np.shape[-1] == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Detect faces/persons using InsightFace
            faces = app.get(img_np)
            
            # Create mask from detected bounding boxes
            h, w = img_np.shape[:2]
            mask_array = np.zeros((h, w), dtype=np.float32)
            
            # Check if any component is enabled
            any_component_enabled = face or hair or body or clothes or accessories
            
            if len(faces) > 0 and any_component_enabled:
                # Use all detected faces to create a combined mask
                for face_det in faces:
                    bbox = face_det.bbox.astype(int)
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    # Expand bounding box for better coverage (especially for body/clothes)
                    expand_factor = 0.3 if (body or clothes) else 0.1
                    x1 = max(0, x1 - int((x2 - x1) * expand_factor))
                    y1 = max(0, y1 - int((y2 - y1) * expand_factor))
                    x2 = min(w, x2 + int((x2 - x1) * expand_factor))
                    y2 = min(h, y2 + int((y2 - y1) * expand_factor))
                    # Create mask for this detection
                    mask_array[y1:y2, x1:x2] = 1.0
            
            # Apply confidence threshold
            mask_array = (mask_array > confidence).astype(np.float32)
            
            # Convert to image shape format
            image_shape = (h, w, 4)
            mask_background_array = np.zeros(image_shape, dtype=np.uint8)
            mask_background_array[:] = (0, 0, 0, 255)
            mask_foreground_array = np.zeros(image_shape, dtype=np.uint8)
            mask_foreground_array[:] = (255, 255, 255, 255)
            
            # Create mask array from detected regions
            condition = np.stack((mask_array,) * image_shape[-1], axis=-1) > confidence
            merged_mask_arrays = np.where(condition, mask_foreground_array, mask_background_array)
            
            # Create the image
            mask_image = Image.fromarray(merged_mask_arrays)
            # convert PIL image to tensor image
            tensor_mask = mask_image.convert("RGB")
            tensor_mask = np.array(tensor_mask).astype(np.float32) / 255.0
            tensor_mask = torch.from_numpy(tensor_mask)[None,]
            _mask = tensor_mask.squeeze(3)[..., 0]

            # Process detail if enabled
            if process_detail:
                # For now, skip detail processing methods that require additional dependencies
                # These can be added later if needed
                _mask = mask2image(_mask)
            else:
                _mask = mask2image(_mask)

            ret_image = RGB2RGBA(orig_image, _mask)
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0))


class FacetoolsFacialSegment:
    """
    Facial feature segmentation using InsightFace (replacing MediaPipe)
    Adapted from ComfyUI_LayerStyle_Advance's MediapipeFacialSegment node
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "left_eye": ("BOOLEAN", {"default": True}),
                "left_eyebrow": ("BOOLEAN", {"default": True}),
                "right_eye": ("BOOLEAN", {"default": True}),
                "right_eyebrow": ("BOOLEAN", {"default": True}),
                "lips": ("BOOLEAN", {"default": True}),
                "tooth": ("BOOLEAN", {"default": True}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = 'facial_feature_segment'
    CATEGORY = 'facetools_disabled_mediapipe'

    def _draw_eye(self, mask, kps, eye_idx, width, height, is_left=True):
        """Draw eye mask using keypoints"""
        if kps is None or len(kps) <= eye_idx:
            return
        
        eye_center = kps[eye_idx].astype(int)
        # Estimate eye size from face bbox or use default
        eye_radius = int(width * 0.02)  # Approximate eye radius
        
        # Draw ellipse for eye
        axes = (eye_radius, int(eye_radius * 0.6))
        cv2.ellipse(mask, tuple(eye_center), axes, 0, 0, 360, 255, -1)

    def _draw_eyebrow(self, mask, kps, eye_idx, width, height, is_left=True):
        """Draw eyebrow mask using keypoints"""
        if kps is None or len(kps) <= eye_idx:
            return
        
        eye_center = kps[eye_idx].astype(int)
        # Estimate eyebrow position (above eye)
        eyebrow_y = max(0, eye_center[1] - int(width * 0.03))
        eyebrow_center = (eye_center[0], eyebrow_y)
        
        # Draw ellipse for eyebrow
        axes = (int(width * 0.03), int(width * 0.01))
        cv2.ellipse(mask, eyebrow_center, axes, 0, 0, 360, 255, -1)

    def _draw_lips(self, mask, kps, width, height):
        """Draw lips mask using keypoints"""
        if kps is None or len(kps) < 4:
            return
        
        # Use mouth keypoints (kps[3] and kps[4] are mouth corners)
        mouth_left = kps[3].astype(int)
        mouth_right = kps[4].astype(int)
        
        # Calculate mouth center and size
        mouth_center = ((mouth_left[0] + mouth_right[0]) // 2, 
                       (mouth_left[1] + mouth_right[1]) // 2)
        mouth_width = int(np.linalg.norm(mouth_right - mouth_left) * 0.8)
        mouth_height = int(mouth_width * 0.4)
        
        # Draw ellipse for lips
        axes = (mouth_width // 2, mouth_height // 2)
        cv2.ellipse(mask, mouth_center, axes, 0, 0, 360, 255, -1)

    def _draw_tooth(self, mask, kps, width, height):
        """Draw tooth mask using keypoints"""
        if kps is None or len(kps) < 4:
            return
        
        # Use mouth keypoints
        mouth_left = kps[3].astype(int)
        mouth_right = kps[4].astype(int)
        
        # Calculate mouth center and size (smaller than lips)
        mouth_center = ((mouth_left[0] + mouth_right[0]) // 2, 
                       (mouth_left[1] + mouth_right[1]) // 2)
        mouth_width = int(np.linalg.norm(mouth_right - mouth_left) * 0.6)
        mouth_height = int(mouth_width * 0.25)
        
        # Draw smaller ellipse for tooth (inside lips)
        axes = (mouth_width // 2, mouth_height // 2)
        cv2.ellipse(mask, mouth_center, axes, 0, 0, 360, 255, -1)

    def facial_feature_segment(self, image, left_eye, left_eyebrow, right_eye, right_eyebrow, lips, tooth):
        try:
            import insightface
        except:
            install_package("insightface")
            install_package("opencv-python")
            import insightface

        # Initialize InsightFace app
        cache_key = "facetools_insightface_app"
        if cache_key in _insightface_cache:
            app = _insightface_cache[cache_key]
        else:
            app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            _insightface_cache[cache_key] = app

        ret_images = []
        ret_masks = []
        scale_factor = 4

        for img in image:
            face_image = tensor2pil(img.unsqueeze(0)).convert('RGB')
            orig_width, orig_height = face_image.size
            
            # Scale up for better detection
            width = orig_width * scale_factor
            height = orig_height * scale_factor
            scaled_image = face_image.resize((width, height), Image.BILINEAR)
            cv2_image = pil2cv2(scaled_image)
            
            # Detect faces using InsightFace
            faces = app.get(cv2_image)
            mask = np.zeros((height, width), dtype=np.uint8)

            if len(faces) > 0:
                # Use the first detected face
                face = faces[0]
                kps = face.kps  # Keypoints: [right_eye, left_eye, nose, mouth_left, mouth_right]
                
                # InsightFace keypoints order: [right_eye, left_eye, nose, mouth_left, mouth_right]
                # Index: 0=right_eye, 1=left_eye, 2=nose, 3=mouth_left, 4=mouth_right
                
                # Keypoints are already in the image coordinates (scaled image)
                kps_scaled = kps
                
                # Draw facial features
                if left_eye and kps_scaled is not None:
                    self._draw_eye(mask, kps_scaled, 1, width, height, is_left=True)  # left_eye is index 1
                
                if right_eye and kps_scaled is not None:
                    self._draw_eye(mask, kps_scaled, 0, width, height, is_left=False)  # right_eye is index 0
                
                if left_eyebrow and kps_scaled is not None:
                    self._draw_eyebrow(mask, kps_scaled, 1, width, height, is_left=True)
                
                if right_eyebrow and kps_scaled is not None:
                    self._draw_eyebrow(mask, kps_scaled, 0, width, height, is_left=False)
                
                if lips and kps_scaled is not None:
                    self._draw_lips(mask, kps_scaled, width, height)
                
                if tooth and kps_scaled is not None:
                    self._draw_tooth(mask, kps_scaled, width, height)

            # Convert mask back to original size
            mask_pil = Image.fromarray(mask, mode='L')
            mask_pil = gaussian_blur(mask_pil, 2)
            mask_pil = mask_pil.resize((orig_width, orig_height), Image.BILINEAR)
            
            ret_images.append(pil2tensor(RGB2RGBA(face_image, mask_pil)))
            ret_masks.append(image2mask(mask_pil))

        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0))


NODE_CLASS_MAPPINGS = {
    "facetools_humanSegmentationIF": FacetoolsHumanSegmentation,
    "facetools_PersonMaskUltraV2": FacetoolsPersonMaskUltraV2,
    "facetools_FacialSegment": FacetoolsFacialSegment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "facetools_humanSegmentationIF": "Facetools Human Segmentation (InsightFace)",
    "facetools_PersonMaskUltraV2": "Facetools Person Mask Ultra V2 (InsightFace)",
    "facetools_FacialSegment": "Facetools Facial Segment (InsightFace)",
}

