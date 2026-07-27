# Face Parsing (integrated)

This directory is the integrated face parsing node set from
[comfyui_face_parsing](https://github.com/Ryuukeisyou/comfyui_face_parsing).

It is registered by the parent extension (`comfyui_facetools_disabled_mediapipe`)
via the root `__init__.py`. Do **not** install this folder as a separate
ComfyUI `custom_nodes` package.

## Models (first run)

On first import, required files are downloaded to:

- `ComfyUI/models/face_parsing/` — face parsing model (`model.safetensors`, configs)
- `ComfyUI/models/ultralytics/bbox/` — `face_yolov8m.pt`

If Hugging Face download fails, place the files manually from:

- https://huggingface.co/jonathandinu/face-parsing
- https://huggingface.co/Bingsu/adetailer/

## License

See `LICENSE` in this directory (original package license).
