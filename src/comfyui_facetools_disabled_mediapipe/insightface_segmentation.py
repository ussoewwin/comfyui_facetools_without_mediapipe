"""
InsightFace-based segmentation nodes (replacing MediaPipe)
Adapted from comfyui-easy-use and ComfyUI_LayerStyle_Advance
"""

import os
import sys
import subprocess
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import folder_paths
from urllib.parse import urlparse
from torch.hub import download_url_to_file
from torchvision.transforms import Resize, CenterCrop, ToPILImage
from torchvision.transforms.functional import to_pil_image
from comfy_extras.nodes_compositing import JoinImageWithAlpha

def install_package(package):
    """Install package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")

# Simple cache for InsightFace app instances
_insightface_cache = {}

# Human parsing models configuration (from comfyui-easy-use)
HUMANPARSING_MODELS = {
    "parsing_lip": {
        "model_url": "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/parsing_lip.onnx",
    },
    "human-parts": {
        "model_url": "https://huggingface.co/Metal3d/deeplabv3p-resnet50-human/resolve/main/deeplabv3p-resnet50-human.onnx",
    },
    "segformer_b3_clothes": {
        "model_name": "sayeed99/segformer_b3_clothes",
    },
    "segformer_b3_fashion": {
        "model_name": "sayeed99/segformer-b3-fashion",
    },
    "face_parsing": {
        "model_name": "jonathandinu/face-parsing"
    }
}

def get_local_filepath(url, dirname, local_file_name=None):
    """Get local file path when is already downloaded or download it"""
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)
    destination = os.path.join(dirname, local_file_name)
    if not os.path.exists(destination):
        try:
            print(f'Downloading {url} to {destination}')
            download_url_to_file(url, destination)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            raise
    return destination

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
        method_list = ["selfie_multiclass_256x256", "human_parsing_lip", "human_parts (deeplabv3p)", "segformer_b3_clothes", "segformer_b3_fashion", "face_parsing"]
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (method_list, {"default": "selfie_multiclass_256x256"}),
                "confidence": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 0.95, "step": 0.01},),
                "crop_multi": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001},),
                "mask_components": ("STRING", {"default": "0", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOX")
    RETURN_NAMES = ("image", "mask", "bbox")
    FUNCTION = "parsing"
    CATEGORY = "facetools_disabled_mediapipe"

    def parsing(self, image, method, confidence, crop_multi, mask_components):
        if method == 'selfie_multiclass_256x256':
            try:
                import insightface
                import cv2
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

        elif method == "human_parsing_lip":
            # Parse mask_components string to list of integers
            try:
                mask_components_list = [int(x.strip()) for x in mask_components.split(',') if x.strip()]
            except:
                mask_components_list = [0]
            
            try:
                from .human_parsing.run_parsing import HumanParsing
                
                cache_key = f"human_parsing_lip_{method}"
                if cache_key in _insightface_cache:
                    parsing = _insightface_cache[cache_key]
                else:
                    onnx_path = os.path.join(folder_paths.models_dir, 'onnx')
                    model_path = get_local_filepath(HUMANPARSING_MODELS['parsing_lip']['model_url'], onnx_path)
                    parsing = HumanParsing(model_path=model_path)
                    _insightface_cache[cache_key] = parsing

                model_image = image.squeeze(0)
                model_image = model_image.permute((2, 0, 1))
                model_image = to_pil_image(model_image)

                map_image, mask = parsing(model_image, mask_components_list)
                mask = mask[:, :, :, 0]
                alpha = 1.0 - mask
                try:
                    output_image, = JoinImageWithAlpha().execute(image, alpha)
                except:
                    output_image, = JoinImageWithAlpha().join_image_with_alpha(image, alpha)
            except Exception as e:
                print(f"Error in human_parsing_lip: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to empty output
                output_image = image
                mask = torch.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=torch.float32)

        elif method == "human_parts (deeplabv3p)":
            # Parse mask_components string to list of integers
            try:
                mask_components_list = [int(x.strip()) for x in mask_components.split(',') if x.strip()]
            except:
                mask_components_list = [0]
            
            try:
                from .human_parsing.run_parsing import HumanParts
                
                cache_key = f"human_parts_{method}"
                if cache_key in _insightface_cache:
                    parsing = _insightface_cache[cache_key]
                else:
                    onnx_path = os.path.join(folder_paths.models_dir, 'onnx')
                    human_parts_path = os.path.join(onnx_path, 'human-parts')
                    model_path = get_local_filepath(HUMANPARSING_MODELS['human-parts']['model_url'], human_parts_path)
                    parsing = HumanParts(model_path=model_path)
                    _insightface_cache[cache_key] = parsing

                ret_images = []
                ret_masks = []
                for img in image:
                    mask, = parsing(img, mask_components_list)
                    _mask = tensor2pil(mask).convert('L')
                    ret_image = RGB2RGBA(tensor2pil(img).convert('RGB'), _mask.convert('L'))
                    ret_images.append(pil2tensor(ret_image))
                    ret_masks.append(image2mask(_mask))

                output_image = torch.cat(ret_images, dim=0)
                mask = torch.cat(ret_masks, dim=0)
            except Exception as e:
                print(f"Error in human_parts: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to empty output
                output_image = image
                mask = torch.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=torch.float32)

        elif method in ["segformer_b3_clothes", "segformer_b3_fashion", "face_parsing"]:
            try:
                from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
                import shutil
                
                # Parse mask_components string to list of integers
                try:
                    mask_components_list = [int(x.strip()) for x in mask_components.split(',') if x.strip()]
                except:
                    mask_components_list = [0]

                def get_segmentation_from_model(tensor_image, model, processor):
                    cloth = tensor2pil(tensor_image)
                    inputs = processor(images=cloth, return_tensors="pt")
                    outputs = model(**inputs)
                    logits = outputs.logits.cpu()
                    upsampled_logits = F.interpolate(logits, size=cloth.size[::-1], mode="bilinear", align_corners=False)
                    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
                    return pred_seg, cloth

                cache_key = f"segformer_{method}"
                if cache_key in _insightface_cache:
                    processor, model = _insightface_cache[cache_key]
                else:
                    model_folder_path = os.path.join(folder_paths.models_dir, method)
                    if not os.path.exists(model_folder_path):
                        try:
                            from huggingface_hub import snapshot_download
                            print(f"Model not found locally. Downloading {method}...")
                            model_path_cache = os.path.join(folder_paths.models_dir, "cache-"+method)
                            snapshot_download(
                                repo_id=HUMANPARSING_MODELS[method]['model_name'],
                                local_dir=model_path_cache,
                                local_dir_use_symlinks=False,
                                resume_download=True
                            )
                            shutil.move(model_path_cache, model_folder_path)
                            print(f"Model downloaded to {model_folder_path}...")
                        except Exception as e:
                            print(f"Error downloading model: {e}")
                            raise

                    processor = SegformerImageProcessor.from_pretrained(model_folder_path)
                    model = AutoModelForSemanticSegmentation.from_pretrained(model_folder_path)
                    _insightface_cache[cache_key] = (processor, model)

                ret_images = []
                ret_masks = []

                if method == "face_parsing":
                    import matplotlib
                    transform = ToPILImage()
                    device = model.device
                    results = []
                    for img in image:
                        size = img.shape[:2]
                        inputs = processor(images=transform(img.permute(2, 0, 1)), return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        outputs = model(**inputs)
                        logits = outputs.logits
                        upsampled_logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
                        pred_seg = upsampled_logits.argmax(dim=1)[0]
                        pred_seg_np = pred_seg.cpu().detach().numpy().astype(np.uint8)
                        results.append(torch.tensor(pred_seg_np))

                    results_out = torch.stack(results, dim=0)
                    for img, result_item in zip(image, results_out):
                        mask = torch.zeros(result_item.shape, dtype=torch.uint8)
                        for i in mask_components_list:
                            mask = mask | torch.where(result_item == i, 1, 0)
                        mask_np = (mask * 255).numpy().astype(np.uint8)
                        _mask = Image.fromarray(mask_np)
                        ret_image = RGB2RGBA(tensor2pil(img).convert('RGB'), _mask.convert('L'))
                        ret_images.append(pil2tensor(ret_image))
                        ret_masks.append(image2mask(_mask))
                else:
                    for img in image:
                        pred_seg, cloth = get_segmentation_from_model(img, model, processor)
                        mask = np.isin(pred_seg, mask_components_list).astype(np.uint8)
                        _mask = Image.fromarray(mask * 255)
                        ret_image = RGB2RGBA(tensor2pil(img).convert('RGB'), _mask.convert('L'))
                        ret_images.append(pil2tensor(ret_image))
                        ret_masks.append(image2mask(_mask))

                output_image = torch.cat(ret_images, dim=0)
                mask = torch.cat(ret_masks, dim=0)
            except Exception as e:
                print(f"Error in {method}: {e}")
                # Fallback to empty output
                output_image = image
                mask = torch.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=torch.float32)

        # use crop
        bbox = [[0, 0, 0, 0]]
        if crop_multi > 0.0:
            crop_instance = ImageCropFromMask()
            output_image, mask, bbox_raw = crop_instance.crop(output_image, mask, crop_multi, crop_multi, 1.0)
            # Convert bbox from (x, y, w, h) tuples to [[x1, y1, x2, y2]] list format
            if bbox_raw and len(bbox_raw) > 0:
                bbox = []
                for b in bbox_raw:
                    if isinstance(b, (tuple, list)) and len(b) >= 4:
                        # Convert from (x, y, w, h) to [x1, y1, x2, y2]
                        x, y, w, h = b[0], b[1], b[2], b[3]
                        bbox.append([int(x), int(y), int(x + w), int(y + h)])
                    else:
                        bbox.append([0, 0, 0, 0])
            else:
                bbox = [[0, 0, 0, 0]]

        return (output_image, mask, bbox)


class ImageCropFromMask:
    """Image crop from mask - adapted from comfyui-easy-use's imageCropFromMask"""
    
    def smooth_bbox_size(self, prev_bbox_size, curr_bbox_size, alpha):
        if alpha == 0:
            return prev_bbox_size
        return round(alpha * curr_bbox_size + (1 - alpha) * prev_bbox_size)

    def smooth_center(self, prev_center, curr_center, alpha=0.5):
        if alpha == 0:
            return prev_center
        return (
            round(alpha * curr_center[0] + (1 - alpha) * prev_center[0]),
            round(alpha * curr_center[1] + (1 - alpha) * prev_center[1])
        )

    def image2mask(self, image):
        return image[:, :, :, 0]

    def mask2image(self, mask):
        return mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

    def cropimage(self, original_images, masks, crop_size_mult, bbox_smooth_alpha):
        bounding_boxes = []
        cropped_images = []

        self.max_bbox_width = 0
        self.max_bbox_height = 0

        # First, calculate the maximum bounding box size across all masks
        curr_max_bbox_width = 0
        curr_max_bbox_height = 0
        for mask in masks:
            _mask = tensor2pil(mask)
            non_zero_indices = np.nonzero(np.array(_mask))
            if len(non_zero_indices[0]) > 0 and len(non_zero_indices[1]) > 0:
                min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
                min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
                width = max_x - min_x
                height = max_y - min_y
                curr_max_bbox_width = max(curr_max_bbox_width, width)
                curr_max_bbox_height = max(curr_max_bbox_height, height)

        # Smooth the changes in the bounding box size
        self.max_bbox_width = self.smooth_bbox_size(self.max_bbox_width, curr_max_bbox_width, bbox_smooth_alpha)
        self.max_bbox_height = self.smooth_bbox_size(self.max_bbox_height, curr_max_bbox_height, bbox_smooth_alpha)

        # Apply the crop size multiplier
        self.max_bbox_width = round(self.max_bbox_width * crop_size_mult)
        self.max_bbox_height = round(self.max_bbox_height * crop_size_mult)
        bbox_aspect_ratio = self.max_bbox_width / self.max_bbox_height if self.max_bbox_height > 0 else 1.0

        # Then, for each mask and corresponding image...
        for i, (mask, img) in enumerate(zip(masks, original_images)):
            _mask = tensor2pil(mask)
            non_zero_indices = np.nonzero(np.array(_mask))
            if len(non_zero_indices[0]) > 0 and len(non_zero_indices[1]) > 0:
                min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
                min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])

                # Calculate center of bounding box
                center_x = np.mean(non_zero_indices[1])
                center_y = np.mean(non_zero_indices[0])
                curr_center = (round(center_x), round(center_y))

                # If this is the first frame, initialize prev_center with curr_center
                if not hasattr(self, 'prev_center'):
                    self.prev_center = curr_center

                # Smooth the changes in the center coordinates from the second frame onwards
                if i > 0:
                    center = self.smooth_center(self.prev_center, curr_center, bbox_smooth_alpha)
                else:
                    center = curr_center

                # Update prev_center for the next frame
                self.prev_center = center

                # Create bounding box using max_bbox_width and max_bbox_height
                half_box_width = round(self.max_bbox_width / 2)
                half_box_height = round(self.max_bbox_height / 2)
                min_x = max(0, center[0] - half_box_width)
                max_x = min(img.shape[1], center[0] + half_box_width)
                min_y = max(0, center[1] - half_box_height)
                max_y = min(img.shape[0], center[1] + half_box_height)

                # Append bounding box coordinates
                bounding_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))

                # Crop the image from the bounding box
                cropped_img = img[min_y:max_y, min_x:max_x, :]

                # Calculate the new dimensions while maintaining the aspect ratio
                new_height = min(cropped_img.shape[0], self.max_bbox_height)
                new_width = round(new_height * bbox_aspect_ratio)

                # Resize the image
                resize_transform = Resize((new_height, new_width))
                resized_img = resize_transform(cropped_img.permute(2, 0, 1))

                # Perform the center crop to the desired size
                crop_transform = CenterCrop((self.max_bbox_height, self.max_bbox_width))
                cropped_resized_img = crop_transform(resized_img)

                cropped_images.append(cropped_resized_img.permute(1, 2, 0))
            else:
                # No mask found, return empty bbox
                bounding_boxes.append((0, 0, 0, 0))
                cropped_images.append(img)

        return cropped_images, bounding_boxes

    def crop(self, image, mask, image_crop_multi, mask_crop_multi, bbox_smooth_alpha):
        cropped_images, bounding_boxes = self.cropimage(image, mask, image_crop_multi, bbox_smooth_alpha)
        cropped_mask_image, _ = self.cropimage(self.mask2image(mask), mask, mask_crop_multi, bbox_smooth_alpha)

        cropped_image_out = torch.stack(cropped_images, dim=0)
        cropped_mask_out = torch.stack(cropped_mask_image, dim=0)

        return (cropped_image_out, cropped_mask_out[:, :, :, 0], bounding_boxes)


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

