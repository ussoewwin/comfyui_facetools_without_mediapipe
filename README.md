# comfyui_facetools_disabled_mediapipe

Face detection & restoration tools for ComfyUI by Sunx.ai

> [!NOTE]
> This projected was created with a [cookiecutter](https://github.com/Comfy-Org/cookiecutter-comfy-extension) template. It helps you start writing custom nodes without worrying about the Python setup.

## Quickstart

**⚠️ Note**: This extension is not available in ComfyUI Manager. You must install it manually.

1. Install [ComfyUI](https://docs.comfy.org/get_started).
2. Clone this repository into your `ComfyUI/custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/ussoewwin/comfyui_facetools_disabled_mediapipe.git
   ```
3. Install dependencies from `requirements.txt`:
   ```bash
   cd comfyui_facetools_disabled_mediapipe
   pip install -r requirements.txt
   ```
   **Important**: For Python 3.13 Windows users, see the [Installation Dependencies](#installation-dependencies) section below for special instructions.
4. Restart ComfyUI.

# Features

## Face Detection & Processing
- **DetectFaces**: Detects faces in images and returns the face with the largest area
- **DetectFaceByIndex**: Detects faces by index, supporting left-to-right selection of specific faces (0=leftmost, 1=second, etc.), with gender filtering support (0=any gender, 1=male, 2=female). Gender detection is only enabled when needed for improved performance.
- **DetectFaceByGender**: Detects faces by gender and index, supporting filtering of male/female faces and selection in left-to-right order
- **CropFaces**: Crops detected face regions
- **WarpFaceBack**: Warps processed faces back to the original image
- **InstantID**: Face identity preservation functionality
- **ColorAdjust**: Face color adjustment
- **SaveImageWebsocket**: Saves images via WebSocket

## Segmentation (InsightFace-based)
- **Facetools Human Segmentation (InsightFace)**: Human segmentation using InsightFace face detection. Detects faces and generates segmentation masks with configurable confidence threshold. Returns RGBA image with mask and separate mask output.
- **Facetools Person Mask Ultra V2 (InsightFace)**: Advanced person mask generation with detailed component selection (face, hair, body, clothes, accessories, background). Supports multiple detail processing methods (VITMatte, PyMatting, GuidedFilter) with configurable erode/dilate operations, black/white point adjustment, and device selection (CUDA/CPU).
- **Facetools Facial Segment (InsightFace)**: Facial feature segmentation for individual components. Supports selective segmentation of left/right eyes, eyebrows, lips, and teeth. Uses InsightFace keypoints for precise feature detection and generates masks with Gaussian blur for smooth edges.

## Installation Dependencies

This extension uses InsightFace for high-precision gender detection (replacing mediapipe for Python 3.13 compatibility). Models will be automatically downloaded on first use:

### Standard Installation (Python 3.10-3.12)

```bash
pip install insightface
```

### Python 3.13 Users (Windows Only)

**⚠️ Important**: The standard `pip install insightface` does not provide Python 3.13 wheels. Windows users running Python 3.13 must download the pre-built wheel from:

**https://huggingface.co/ussoewwin/Insightface_for_windows/tree/main**

Download the appropriate wheel file (`insightface-0.7.3-cp313-cp313-win_amd64.whl`) and install it directly:

```bash
pip install insightface-0.7.3-cp313-cp313-win_amd64.whl
```

**Note**: This is Windows-only. Linux/macOS Python 3.13 users will need to build from source or use Python 3.12 or earlier.

**Note**: InsightFace requires additional model files that will be automatically downloaded on first run.

### Gender Detection Features
- Uses InsightFace for high-precision gender recognition
- Supports both GPU and CPU modes with automatic device selection
- Provides age detection and confidence information
- Supports fallback mechanism (based on facial aspect ratio)
- Prevents division by zero errors to ensure stable operation

## Develop

To install the dev dependencies and pre-commit (will run the ruff hook), do:

```bash
cd comfyui_facetools_disabled_mediapipe
pip install -e .[dev]
pre-commit install
```

The `-e` flag above will result in a "live" install, in the sense that any changes you make to your node extension will automatically be picked up the next time you run ComfyUI.

## Publish to Github

Install Github Desktop or follow these [instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) for ssh.

1. Create a Github repository that matches the directory name.
2. Push the files to Git
```
git add .
git commit -m "project scaffolding"
git push
```

## Writing custom nodes

An example custom node is located in [node.py](src/comfyui_facetools_disabled_mediapipe/nodes.py). To learn more, read the [docs](https://docs.comfy.org/essentials/custom_node_overview).


## Tests

This repo contains unit tests written in Pytest in the `tests/` directory. It is recommended to unit test your custom node.

- [build-pipeline.yml](.github/workflows/build-pipeline.yml) will run pytest and linter on any open PRs
- [validate.yml](.github/workflows/validate.yml) will run [node-diff](https://github.com/Comfy-Org/node-diff) to check for breaking changes

