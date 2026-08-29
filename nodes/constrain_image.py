import torch
import numpy as np
from PIL import Image


class FacetoolsConstrainImage:
    """
    A node that constrains an image to a maximum and minimum size while maintaining aspect ratio.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "max_width": ("INT", {"default": 1024, "min": 0, "max": 16384, "step": 8}),
                "max_height": ("INT", {"default": 1024, "min": 0, "max": 16384, "step": 8}),
                "min_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "min_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "crop_if_required": (["yes", "no"], {"default": "no"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "constrain_image"
    CATEGORY = "facetools/image"
    OUTPUT_IS_LIST = (True,)

    def constrain_image(self, images, max_width, max_height, min_width, min_height, crop_if_required):
        crop_if_required = crop_if_required == "yes"
        results = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB")

            current_width, current_height = img.size
            if current_width <= 0 or current_height <= 0:
                results.append(image if image.ndim == 4 else image.unsqueeze(0))
                continue
            aspect_ratio = current_width / current_height

            eff_max_w = max_width if max_width > 0 else current_width
            eff_max_h = max_height if max_height > 0 else current_height
            eff_min_w = min_width if min_width > 0 else 0
            eff_min_h = min_height if min_height > 0 else 0

            constrained_width = max(min(max(current_width, eff_min_w), eff_max_w), 1)
            constrained_height = max(min(max(current_height, eff_min_h), eff_max_h), 1)

            if constrained_width / constrained_height > aspect_ratio:
                constrained_width = max(int(constrained_height * aspect_ratio), max(eff_min_w, 1))
                if crop_if_required:
                    div = current_width / constrained_width
                    constrained_height = max(int(current_height / div) if div > 0 else constrained_height, 1)
            else:
                constrained_height = max(int(constrained_width / max(aspect_ratio, 1e-6)), max(eff_min_h, 1))
                if crop_if_required:
                    div = current_height / constrained_height
                    constrained_width = max(int(current_width / div) if div > 0 else constrained_width, 1)

            resized_image = img.resize((constrained_width, constrained_height), Image.LANCZOS)

            if crop_if_required and (constrained_width > max_width or constrained_height > max_height):
                left = max((constrained_width - max_width) // 2, 0)
                top = max((constrained_height - max_height) // 2, 0)
                right = min(constrained_width, max_width) + left
                bottom = min(constrained_height, max_height) + top
                resized_image = resized_image.crop((left, top, right, bottom))

            resized_image = np.array(resized_image).astype(np.float32) / 255.0
            resized_image = torch.from_numpy(resized_image)[None,]
            results.append(resized_image)

        return (results,)


NODE_CLASS_MAPPINGS = {
    "FacetoolsConstrainImage": FacetoolsConstrainImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FacetoolsConstrainImage": "Constrain Image (Facetools)",
}
