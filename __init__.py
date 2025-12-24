from .nodes import *
from .InstantID import *

# SAM3 (local) - keep optional so the rest of the extension still loads even if SAM3 deps are missing.
try:
    from .AILab_SAM3Segment import (
        NODE_CLASS_MAPPINGS as SAM3_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as SAM3_NODE_DISPLAY_NAME_MAPPINGS,
    )
except Exception:
    SAM3_NODE_CLASS_MAPPINGS = {}
    SAM3_NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .src.comfyui_facetools_disabled_mediapipe.insightface_segmentation import (
        NODE_CLASS_MAPPINGS as INSIGHTFACE_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as INSIGHTFACE_NODE_DISPLAY_NAME_MAPPINGS,
    )
except ImportError:
    INSIGHTFACE_NODE_CLASS_MAPPINGS = {}
    INSIGHTFACE_NODE_DISPLAY_NAME_MAPPINGS = {}

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

# Add SAM3 nodes (if available)
NODE_CLASS_MAPPINGS.update(SAM3_NODE_CLASS_MAPPINGS)

# Add InsightFace segmentation nodes
NODE_CLASS_MAPPINGS.update(INSIGHTFACE_NODE_CLASS_MAPPINGS)

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

# Add SAM3 node display names (if available)
NODE_DISPLAY_NAME_MAPPINGS.update(SAM3_NODE_DISPLAY_NAME_MAPPINGS)

# Add InsightFace segmentation node display names
NODE_DISPLAY_NAME_MAPPINGS.update(INSIGHTFACE_NODE_DISPLAY_NAME_MAPPINGS)




# Load ComfyUI frontend extensions (e.g. COLORCODE widget)
WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
