import torch
from torch import Tensor
from torchvision.transforms import functional

from PIL import Image
import numpy as np
import comfy.utils
import time
from io import BytesIO


from ..utils import *

class DetectFaces:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'threshold': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'min_size': ('INT', {'default': 64, 'max': 512, 'step': 8}),
                'max_size': ('INT', {'default': 512, 'min': 512, 'step': 8}),
            },
            'optional': {
                'mask': ('MASK',),
            }
        }

    RETURN_TYPES = ('FACE', 'BOOLEAN')
    RETURN_NAMES = ('faces', 'has_face')
    FUNCTION = 'run'
    CATEGORY = 'facetools_disabled_mediapipe'

    def run(self, image, threshold, min_size, max_size, mask=None):
        faces = []
        masked = image
        if mask is not None:
            masked = image * tv.transforms.functional.resize(1-mask, image.shape[1:3])[..., None]
        masked = (masked * 255).type(torch.uint8)
        for i, img in enumerate(masked):
            unfiltered_faces = detect_faces(img, threshold)
            for face in unfiltered_faces:
                a, b, c, d = face.bbox
                h = abs(d-b)
                w = abs(c-a)
                if (h <= max_size or w <= max_size) and (min_size <= h or min_size <= w):
                    face.image_idx = i
                    face.img = image[i]
                    faces.append(face)

        # Return only the face with the largest area, or empty list if no faces found

        if faces:
            largest_face = max(faces, key=lambda f: abs(f.bbox[2] - f.bbox[0]) * abs(f.bbox[3] - f.bbox[1]))
            faces = [largest_face]
        else:
            faces = []

        has_face = len(faces) > 0
        return (faces, has_face)


class DetectFaceByIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'threshold': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'tooltip': 'Face detection confidence threshold, higher values are more strict'}),
                'min_size': ('INT', {'default': 64, 'max': 512, 'step': 8, 'tooltip': 'Minimum face size, filters out detection results that are too small'}),
                'max_size': ('INT', {'default': 512, 'min': 512, 'step': 8, 'tooltip': 'Maximum face size, filters out detection results that are too large'}),
                'face_index': ('INT', {'default': 0, 'min': 0, 'max': 10, 'step': 1, 'tooltip': 'Face index: 0=leftmost first, 1=second, and so on'}),
                'gender_filter': ('INT', {'default': 0, 'min': 0, 'max': 2, 'step': 1, 'tooltip': 'Gender filter: 0=any gender, 1=detect male only, 2=detect female only'}),
                'priority_mode': ('INT', {'default': 0, 'min': 0, 'max': 1, 'step': 1, 'tooltip': 'Priority mode: 0=index priority (select position first then check gender), 1=gender priority (filter gender first then select position)'}),
            },
            'optional': {
                'mask': ('MASK',),
            }
        }

    RETURN_TYPES = ('FACE', 'BOOLEAN')
    RETURN_NAMES = ('faces', 'has_face')
    FUNCTION = 'run'
    CATEGORY = 'facetools_disabled_mediapipe'

    def run(self, image, threshold, min_size, max_size, face_index, gender_filter, priority_mode, mask=None):
        faces = []
        masked = image
        if mask is not None:
            masked = image * tv.transforms.functional.resize(1-mask, image.shape[1:3])[..., None]
        masked = (masked * 255).type(torch.uint8)

        # First perform YOLO face detection
        for i, img in enumerate(masked):
            unfiltered_faces = detect_faces(img, threshold, detect_gender=False)  # Don't detect gender
            for face in unfiltered_faces:
                a, b, c, d = face.bbox
                h = abs(d-b)
                w = abs(c-a)
                if (h <= max_size or w <= max_size) and (min_size <= h or min_size <= w):
                    face.image_idx = i
                    face.img = image[i]
                    faces.append(face)

        print(f"[DetectFaceByIndex] YOLO detected {len(faces)} faces")

        # Directly call Models.gender for gender detection
        insightface_genders = Models.gender(image[0])  # Detect gender on original image
        print(f"[DetectFaceByIndex] InsightFace detected {len(insightface_genders)} faces with genders")

        # Check if face count and gender count match
        if len(faces) != len(insightface_genders):
            print(f"[DetectFaceByIndex] Warning: Face count mismatch! YOLO={len(faces)}, InsightFace={len(insightface_genders)}")
            # If counts don't match, use heuristic method
            faces.sort(key=lambda f: f.bbox[0])
            for i, face in enumerate(faces):
                face.gender = "man" if i == 0 else "woman"  # Left is male, right is female
                print(f"[DetectFaceByIndex] Face {i}: Using heuristic gender={face.gender}")
        else:
            # Counts match, assign genders in order
            faces.sort(key=lambda f: f.bbox[0])  # Sort by x coordinate
            for i, face in enumerate(faces):
                face.gender = insightface_genders[i]['gender']
                print(f"[DetectFaceByIndex] Face {i}: YOLO bbox={face.bbox}, assigned gender={face.gender}")

        # Sort by x coordinate first (left to right)
        faces.sort(key=lambda f: f.bbox[0])

        # Print gender information for all faces
        print(f"[DetectFaceByIndex] Priority mode: {priority_mode} ({'index priority' if priority_mode == 0 else 'gender priority'})")
        print(f"[DetectFaceByIndex] Gender filter: {gender_filter}")
        print(f"[DetectFaceByIndex] Face index: {face_index}")
        print(f"[DetectFaceByIndex] Total faces before filtering: {len(faces)}")
        for i, face in enumerate(faces):
            print(f"[DetectFaceByIndex] Face {i}: bbox={face.bbox}, gender={face.gender}")

        if priority_mode == 0:  # Index priority: select position first then check gender
            print(f"[DetectFaceByIndex] Using index priority mode")
            if faces and face_index < len(faces):
                selected_face = faces[face_index]
                print(f"[DetectFaceByIndex] Selected face {face_index}: bbox={selected_face.bbox}, gender={selected_face.gender}")

                # Check if gender matches requirement
                if gender_filter == 1:  # Must be male
                    if selected_face.gender == "man":
                        faces = [selected_face]
                        print(f"[DetectFaceByIndex] Face {face_index} is male, keeping it")
                    else:
                        faces = []
                        print(f"[DetectFaceByIndex] Face {face_index} is not male, returning empty")
                elif gender_filter == 2:  # Must be female
                    if selected_face.gender == "woman":
                        faces = [selected_face]
                        print(f"[DetectFaceByIndex] Face {face_index} is female, keeping it")
                    else:
                        faces = []
                        print(f"[DetectFaceByIndex] Face {face_index} is not female, returning empty")
                else:  # gender_filter == 0, no gender check
                    faces = [selected_face]
                    print(f"[DetectFaceByIndex] No gender filter, keeping face {face_index}")
            else:
                faces = []  # Index out of range, return empty list
                print(f"[DetectFaceByIndex] Face index {face_index} out of range, returning empty")

        else:  # Gender priority: filter gender first then select position
            print(f"[DetectFaceByIndex] Using gender priority mode")

            # First filter by gender
            if gender_filter == 1:  # Select males only
                filtered_faces = [face for face in faces if face.gender == "man"]
                print(f"[DetectFaceByIndex] After filtering for men: {len(filtered_faces)} faces")
            elif gender_filter == 2:  # Select females only
                filtered_faces = [face for face in faces if face.gender == "woman"]
                print(f"[DetectFaceByIndex] After filtering for women: {len(filtered_faces)} faces")
            else:  # gender_filter == 0, no gender filtering
                filtered_faces = faces
                print(f"[DetectFaceByIndex] No gender filtering, keeping all faces")

            # Then select by face_index
            if filtered_faces and face_index < len(filtered_faces):
                selected_face = filtered_faces[face_index]
                faces = [selected_face]
                print(f"[DetectFaceByIndex] Selected face {face_index} from filtered faces: bbox={selected_face.bbox}, gender={selected_face.gender}")
            else:
                faces = []
                print(f"[DetectFaceByIndex] Face index {face_index} out of range in filtered faces, returning empty")

        has_face = len(faces) > 0
        return (faces, has_face)



class CropFaces:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'faces': ('FACE',),
                'crop_size': ('INT', {'default': 512, 'min': 512, 'max': 1024, 'step': 128}),
                'crop_factor': ('FLOAT', {'default': 1.5, 'min': 1.0, 'max': 3, 'step': 0.1}),
                'mask_type': (mask_types,)
            }
        }

    RETURN_TYPES = ('IMAGE', 'MASK', 'WARP')
    RETURN_NAMES = ('crops', 'masks', 'warps')
    FUNCTION = 'run'
    CATEGORY = 'facetools_disabled_mediapipe'

    def run(self, faces, crop_size, crop_factor, mask_type):
        if len(faces) == 0:
            empty_crop = torch.zeros((1,512,512,3))
            empty_mask = torch.zeros((1,512,512))
            empty_warp = np.array([
                [1,0,-512],
                [0,1,-512],
            ], dtype=np.float32)
            return (empty_crop, empty_mask, [empty_warp])

        crops = []
        masks = []
        warps = []
        for face in faces:
            M, crop = face.crop(crop_size, crop_factor)
            mask = mask_crop(face, M, crop, mask_type)
            crops.append(np.array(crop[0]))
            masks.append(np.array(mask[0]))
            warps.append(M)
        crops = torch.from_numpy(np.array(crops)).type(torch.float32)
        masks = torch.from_numpy(np.array(masks)).type(torch.float32)
        return (crops, masks, warps)

class WarpFaceBack:
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'run'
    CATEGORY = 'facetools_disabled_mediapipe'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'images': ('IMAGE',),
                'face': ('FACE',),
                'crop': ('IMAGE',),
                'mask': ('MASK',),
                'warp': ('WARP',),
            },
            'optional': {
                'has_face': ('BOOLEAN',),
            }
        }

    def run(self, images, face, crop, mask, warp, has_face=True):
        # If has_face is False, return original image directly
        if not has_face:
            return (images,)

        # Process single face
        if len(face) == 0:
            return (images,)

        single_face = face[0]
        single_crop = crop[0]
        single_mask = mask[0]
        single_warp = warp[0]

        results = []
        for i, image in enumerate(images):
            if i != single_face.image_idx:
                result = image
            else:
                warped_mask = np.clip(cv2.warpAffine(single_mask.numpy(),
                                cv2.invertAffineTransform(single_warp),
                                image.shape[1::-1],
                                flags=cv2.INTER_LANCZOS4), 0, 1)

                swapped = np.clip(cv2.warpAffine(single_crop.cpu().numpy(),
                                cv2.invertAffineTransform(single_warp),
                                image.shape[1::-1],
                                flags=cv2.INTER_LANCZOS4), 0, 1)

                result = (swapped * warped_mask[..., None] +
                         (1 - warped_mask[..., None]) * image.numpy())
                result = torch.from_numpy(result)
            results.append(result)

        results = torch.stack(results)
        return (results, )

class VAEDecodeNew:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The latent to be decoded."}),
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."})
            },
            'optional': {
                'has_face': ('BOOLEAN',),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The decoded image.",)
    FUNCTION = "decode"

    CATEGORY = "facetools_disabled_mediapipe"
    DESCRIPTION = "Decodes latent images back into pixel space images."

    def decode(self, vae, samples, has_face=True):
        # If has_face is False, return blank canvas to save VAE decode time
        if not has_face:
            # Create 512x512 black blank canvas
            blank_canvas = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (blank_canvas,)
        images = vae.decode(samples["samples"])
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images, )


class VAEEncodeNew:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "pixels": ("IMAGE", ), "vae": ("VAE", )},
            'optional': {
                'has_face': ('BOOLEAN',),
            }
            }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "facetools_disabled_mediapipe"

    def encode(self, vae, pixels, has_face=True):
        # If has_face is False, return blank latent to save VAE encode time
        if not has_face:
            # Create blank latent corresponding to 512x512 (512//8 = 64)
            blank_latent = torch.zeros([1, 4, 64, 64])
            return ({"samples": blank_latent},)

        t = vae.encode(pixels[:,:,:,:3])
        return ({"samples":t}, )


class SelectFloatByBool:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond": ("BOOLEAN",),
                "true_value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "false_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"
    CATEGORY = "facetools_disabled_mediapipe"

    def run(self, cond, true_value, false_value):
        return (true_value if cond else false_value,)



class ColorAdjust:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 255,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number"
                }),
                "brightness": ("FLOAT", {
                    "default": 1.0,
                    "min": -255,
                    "max": 255,
                    "step": 0.01,
                    "round": 0.001, #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number"
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 255,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number"
                }),
                "hue": ("FLOAT", {
                    "default": 0,
                    "min": -0.5,
                    "max": 0.5,
                    "step": 0.001,
                    "round": 0.001,
                    "display": "number"
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 255,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number"
                }),
            },
            'optional': {
                'has_face': ('BOOLEAN',),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "main"

    #OUTPUT_NODE = False

    CATEGORY = "facetools_disabled_mediapipe"

    def main(self,
             image: Tensor,
             contrast: float = 1,
             brightness: float = 1,
             saturation: float = 1,
             hue: float = 0,
             gamma: float = 1,
             has_face: bool = True):

        if not has_face:
            return (image,)

        permutedImage = image.permute(0, 3, 1, 2)

        if (contrast != 1):
            permutedImage = functional.adjust_contrast(permutedImage, contrast)

        if (brightness != 1):
            permutedImage = functional.adjust_brightness(permutedImage, brightness)

        if (saturation != 1):
            permutedImage = functional.adjust_saturation(permutedImage, saturation)

        if (hue != 0):
            permutedImage = functional.adjust_hue(permutedImage, hue)

        if (gamma != 1):
            permutedImage = functional.adjust_gamma(permutedImage, gamma)

        result = permutedImage.permute(0, 2, 3, 1)

        return (result,)


class SaveImageWebsocket:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "jpeg_quality": (
                    "INT",
                    {
                        "default": 95,
                        "min": 60,
                        "max": 100,
                        "step": 1,
                        "tooltip": "JPEG compression quality (60=low quality, 100=high quality)"
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "facetools_disabled_mediapipe"

    def save_images(self, images, jpeg_quality):
        pbar = comfy.utils.ProgressBar(images.shape[0])

        for idx, image in enumerate(images):
            try:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                # Perform JPEG compression directly in memory
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=jpeg_quality)
                buffer.seek(0)
                jpg_img = Image.open(buffer).convert("RGB").copy()

                # Send JPEG format image
                pbar.update_absolute(idx, images.shape[0], ("JPEG", jpg_img, None))

            except Exception as e:
                print(f"[SaveImageWebsocket] ❌ Skipped idx={idx} due to error: {e}")
                continue

        return {}

    @classmethod
    def IS_CHANGED(s, images, jpeg_quality):
        return time.time()
