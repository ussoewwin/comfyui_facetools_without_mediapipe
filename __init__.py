from .nodes.nodes import *
from .InstantID import *

# Image Resize
try:
    from .nodes.image_resize import (
        NODE_CLASS_MAPPINGS as IMAGE_RESIZE_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as IMAGE_RESIZE_NODE_DISPLAY_NAME_MAPPINGS,
    )
except Exception as e:
    print(f"[comfyui_facetools] Failed to load image_resize: {e}")
    import traceback
    traceback.print_exc()
    IMAGE_RESIZE_NODE_CLASS_MAPPINGS = {}
    IMAGE_RESIZE_NODE_DISPLAY_NAME_MAPPINGS = {}

# Constrain Image
try:
    from .nodes.constrain_image import (
        NODE_CLASS_MAPPINGS as CONSTRAIN_IMAGE_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as CONSTRAIN_IMAGE_NODE_DISPLAY_NAME_MAPPINGS,
    )
except Exception as e:
    print(f"[comfyui_facetools] Failed to load constrain_image: {e}")
    import traceback
    traceback.print_exc()
    CONSTRAIN_IMAGE_NODE_CLASS_MAPPINGS = {}
    CONSTRAIN_IMAGE_NODE_DISPLAY_NAME_MAPPINGS = {}


# SAM3 removed: ComfyUI now supports SAM3 natively.

try:
    from .src.comfyui_facetools_disabled_mediapipe.insightface_segmentation import (
        NODE_CLASS_MAPPINGS as INSIGHTFACE_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as INSIGHTFACE_NODE_DISPLAY_NAME_MAPPINGS,
    )
except ImportError:
    INSIGHTFACE_NODE_CLASS_MAPPINGS = {}
    INSIGHTFACE_NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .src.comfyui_facetools_disabled_mediapipe.face_mesh import (
        NODE_CLASS_MAPPINGS as FACE_MESH_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as FACE_MESH_NODE_DISPLAY_NAME_MAPPINGS,
    )
except ImportError:
    FACE_MESH_NODE_CLASS_MAPPINGS = {}
    FACE_MESH_NODE_DISPLAY_NAME_MAPPINGS = {}

# Face Parsing (integrated from nodes/face_parsing)
try:
    from .nodes.face_parsing import (
        NODE_CLASS_MAPPINGS as FACE_PARSING_NODE_CLASS_MAPPINGS,
    )
    FACE_PARSING_NODE_DISPLAY_NAME_MAPPINGS = {
        key: key for key in FACE_PARSING_NODE_CLASS_MAPPINGS
    }
except Exception as e:
    print(f"[comfyui_facetools] Failed to load face_parsing: {e}")
    import traceback
    traceback.print_exc()
    FACE_PARSING_NODE_CLASS_MAPPINGS = {}
    FACE_PARSING_NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS = {
    'DetectFaces': DetectFaces,
    'DetectFaceByIndex': DetectFaceByIndex,
    'CropFaces': CropFaces,
    'WarpFacesBack': WarpFaceBack,
    "SelectFloatByBool": SelectFloatByBool,


    "InstantIDModelLoader": InstantIDModelLoader,
    "InstantIDFaceAnalysis": InstantIDFaceAnalysis,
    "ApplyInstantID": ApplyInstantID,
    "SaveFaceEmbeds": SaveFaceEmbeds,
    "LoadFaceEmbeds": LoadFaceEmbeds,

    "VAEDecodeNew": VAEDecodeNew,
    "VAEEncodeNew": VAEEncodeNew,

    'ColorAdjustNew(FaceParsing)': ColorAdjust,

    "SaveImageWebsocketNew": SaveImageWebsocket,

}

# Add Image Resize nodes
NODE_CLASS_MAPPINGS.update(IMAGE_RESIZE_NODE_CLASS_MAPPINGS)

# Add Constrain Image nodes
NODE_CLASS_MAPPINGS.update(CONSTRAIN_IMAGE_NODE_CLASS_MAPPINGS)

# Add InsightFace segmentation nodes
NODE_CLASS_MAPPINGS.update(INSIGHTFACE_NODE_CLASS_MAPPINGS)

# Add Face Mesh nodes
NODE_CLASS_MAPPINGS.update(FACE_MESH_NODE_CLASS_MAPPINGS)

# Add Face Parsing nodes
NODE_CLASS_MAPPINGS.update(FACE_PARSING_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    'DetectFaces': 'DetectFaces',
    'DetectFaceByIndex': 'Detect Face By Index',
    'CropFaces': 'CropFaces',
    'WarpFacesBack': 'Warp Faces Back',
    "SelectFloatByBool": "Select Float (Bool)",

    "InstantIDModelLoader": "Load InstantID Model",
    "InstantIDFaceAnalysis": "InstantID Face Analysis",
    "ApplyInstantID": "Apply InstantID",
    "SaveFaceEmbeds": "Save Face Embeds",
    "LoadFaceEmbeds": "Load Face Embeds",

    "VAEDecodeNew": "VAE Decode New",
    "VAEEncodeNew": "VAE Encode New",

    'ColorAdjustNew(FaceParsing)': 'Color Adjust (Face Parsing) New',
    "SaveImageWebsocketNew": "Save Image Websocket New To JPG",

}

# Add Image Resize node display names
NODE_DISPLAY_NAME_MAPPINGS.update(IMAGE_RESIZE_NODE_DISPLAY_NAME_MAPPINGS)

# Add Constrain Image node display names
NODE_DISPLAY_NAME_MAPPINGS.update(CONSTRAIN_IMAGE_NODE_DISPLAY_NAME_MAPPINGS)

# Add InsightFace segmentation node display names
NODE_DISPLAY_NAME_MAPPINGS.update(INSIGHTFACE_NODE_DISPLAY_NAME_MAPPINGS)

# Add Face Mesh node display names
NODE_DISPLAY_NAME_MAPPINGS.update(FACE_MESH_NODE_DISPLAY_NAME_MAPPINGS)

# Add Face Parsing node display names
NODE_DISPLAY_NAME_MAPPINGS.update(FACE_PARSING_NODE_DISPLAY_NAME_MAPPINGS)


# Load ComfyUI frontend extensions (e.g. COLORCODE widget)
WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
