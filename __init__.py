from .nodes import *
from .InstantID import *

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

# Add InsightFace segmentation node display names
NODE_DISPLAY_NAME_MAPPINGS.update(INSIGHTFACE_NODE_DISPLAY_NAME_MAPPINGS)




__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
