# SAM3 Integration Notes (Fully English)

This document is a Markdown snapshot of what has been confirmed so far in the chat.

- Everything stated here is **based on code that was actually read** (fact-based assertions).
- For anything not read, we explicitly mark it as **Unread** and do not assert details.

---

## Table of Contents (Chapter Titles)

- [Prerequisites (Facts)](#sec-01)
- ["Shortest Execution Path for SAM3 to Work" (Assertable Range)](#sec-02)
- [Additions So Far (Mapping between `state` and model outputs)](#sec-03)
- [Meaning of "Component Files" Read So Far (Assertions)](#sec-04)
- [Progress Notes](#sec-05)
- [Additional Details (Assertions on propagation loop and memory behavior)](#sec-06)
- [Meaning of `sam3/train/configs/*.yaml` (Read portion)](#sec-07)
- [Unread (Not asserted at this point)](#sec-08)
- [ComfyUI Node Integration + `COLORCODE` (JS widget) (Assertion with code evidence)](#sec-09)
- [Meaning of `sam3/model_builder.py` (Assertion: read)](#sec-10)
- [Core files for video inference (purpose / I/O / call relations)](#sec-11)
- [Image features, geometric prompts, segmentation (Assertions)](#sec-12)
- [Transformer encoder/decoder and backbone integration (Assertions)](#sec-13)
- [SAM2-compatible transformer and SAM1-compatible predictor (Assertions)](#sec-14)
- [SAM2 prompt encoder / mask decoder and RoPE implementation (Assertions)](#sec-15)
- [Overall picture of ComfyUI integration (complete explanation of files actually modified)](#sec-16)
- [Missed items (files found mechanically but not mentioned earlier)](#sec-17)

---

<a id="sec-01"></a>
## Prerequisites (Facts)

- **SAM3 lives in this extension**: `ComfyUI/custom_nodes/comfyui_facetools_without_mediapipe/sam3/`.
- **ComfyUI node implementation**: `ComfyUI/custom_nodes/comfyui_facetools_without_mediapipe/AILab_SAM3Segment.py`.
- **Frontend (COLORCODE widget)**:
  - `ComfyUI/custom_nodes/comfyui_facetools_without_mediapipe/__init__.py` sets `WEB_DIRECTORY = "./js"`.
  - `ComfyUI/custom_nodes/comfyui_facetools_without_mediapipe/js/sam3_color_widget.js` provides the `COLORCODE` widget.
- **SAM3 package markers**: `sam3/__init__.py` exists; `sam3/_init_.py` also exists and is empty.
- **Confirmed file counts under `sam3/`**:
  - Python: 84 files
  - YAML: 37 files (`sam3/train/configs/**.yaml`)
  - gz: 1 (`sam3/assets/bpe_simple_vocab_16e6.txt.gz`)
  - tiff: 1 (`sam3/perflib/tests/assets/masks.tiff`)

---

<a id="sec-02"></a>
## "Shortest Execution Path for SAM3 to Work" (Assertable Range)

### `AILab_SAM3Segment.py` (ComfyUI node entry)

- **Role**: Converts ComfyUI inputs (`IMAGE`, prompt, params) into SAM3 inference calls, then returns outputs in ComfyUI formats (`IMAGE`, `MASK`, `MASK_IMAGE`).
- **Main dependencies**:
  - `build_sam3_image_model` from `sam3/model_builder.py`
  - `Sam3Processor` from `sam3/model/sam3_image_processor.py`
- **Key conversions (node-side)**:
  - ComfyUI `IMAGE` tensor → PIL (`tensor2pil`)
  - `state["masks"]` (bool/float) → 0–255 `L` mask image → optional post-processing (invert/blur/offset) → 0–1 ComfyUI `MASK` tensor
  - Background modes: `Alpha` (RGBA) / `Color` (RGB via `Image.composite`)

### `sam3/model/sam3_image_processor.py` (state-driven inference orchestration)

- **Role**: A wrapper that progresses inference step-by-step using a mutable `state` dict.
- **Important methods (asserted)**:
  - `set_image`: runs preprocessing + `model.backbone.forward_image()` and writes `state["backbone_out"]`.
  - `reset_all_prompts`: clears prompt-related state (`language_*`, `masks/boxes/scores`, etc.).
  - `set_text_prompt`: runs `model.backbone.forward_text()` then calls `_forward_grounding()` to populate results into `state`.
  - `_forward_grounding`:
    - `outputs = self.model.forward_grounding(...)`
    - builds `out_probs = sigmoid(pred_logits) * sigmoid(presence_logit_dec)`
    - filters with `confidence_threshold`
    - boxes: `box_ops.box_cxcywh_to_xyxy` then scale back to original image size
    - masks: `data_misc.interpolate(...).sigmoid()` then threshold to bool (`> 0.5`)

#### Keys written into `state` (Assertion: code evidence)

`Sam3Processor._forward_grounding` writes the following keys into `state`:

- `state["masks_logits"]`
  - **Meaning**: mask probability (float), upsampled to original `(H, W)` and passed through `sigmoid()`.
- `state["masks"]`
  - **Meaning**: bool mask (`state["masks_logits"] > 0.5`).
- `state["boxes"]`
  - **Meaning**: `pred_boxes` (normalized cxcywh) converted to `xyxy` and rescaled to pixel coordinates.
- `state["scores"]`
  - **Meaning**: filtered scores from `sigmoid(pred_logits) * sigmoid(presence_logit_dec)`.
  - **Important**: because presence is multiplied here, **low-presence detections are dropped together with masks/boxes**.

### `sam3/model/sam3_image.py` (image inference core)

- **Role**: Core image inference model (`Sam3Image`).
- **Confirmed facts (from imports and call sites)**:
  - Imports `nms_masks` from `sam3/perflib/nms.py` (mask NMS is part of the design).
  - Imports `Prompt` from `sam3/model/geometry_encoders.py` (geometric prompts are part of the design).

#### Dict keys returned by `forward_grounding` (Assertion)

The `outputs` dict consumed by `Sam3Processor` comes from `Sam3Image.forward_grounding(...)`.
The following keys are **definitely populated** in that dict (based on code that adds them):

- `encoder_hidden_states`: encoder output memory (the `"memory"` from `transformer.encoder`).
- `pred_logits`: updated via `_update_scores_and_boxes` → `_update_out(out, "pred_logits", ...)`.
- `pred_boxes` (normalized cxcywh): updated via `_update_out(out, "pred_boxes", ...)`.
- `pred_boxes_xyxy`: `pred_boxes` converted via `box_cxcywh_to_xyxy` and added.
- `presence_logit_dec`: added if decoder returns `dec_presence_out`.
- `pred_masks`: added by `_run_segmentation_heads` (seg head returns `pred_masks` and it is stored into `out["pred_masks"]`).

#### Processing stages inside `forward_grounding` (Assertion)

`Sam3Image.forward_grounding` builds `out` in four stages (stage names match the `record_function` labels in code):

1. `_encode_prompt`
2. `_run_encoder`
3. `_run_decoder`
4. `_run_segmentation_heads`

Finally, only when `training` or `num_interactive_steps_val > 0`, it adds matcher assignments via `_compute_matching`.

---

<a id="sec-03"></a>
## Additions So Far (Mapping between `state` and model outputs)

This section exists to prevent confusion: it maps what the node code actually uses vs what the SAM3 core returns.

### The node uses `state["masks"]` (bool) as the final mask

- After `state = processor.set_text_prompt(text, state)`, the node fetches `masks = state.get("masks")`.
- If `masks` is missing/empty, it returns an "empty result".
- If present, it converts to float, collapses to one mask via `amax(dim=0)`, and converts to an `L` mask image (0–255).

So the node does not directly use `pred_logits` or `pred_boxes`; it relies on **the final `state["masks"]` produced by the processor**.

### `state["masks"]` is derived from `outputs["pred_masks"]`

- `Sam3Processor._forward_grounding` calls `outputs = self.model.forward_grounding(...)`.
- It interpolates `outputs["pred_masks"]` to original size, applies `sigmoid()` into `state["masks_logits"]`.
- Then it creates `state["masks"] = state["masks_logits"] > 0.5`.

### Scores are computed as `sigmoid(pred_logits) * sigmoid(presence_logit_dec)`

- The processor computes `out_probs = outputs["pred_logits"].sigmoid()`.
- It multiplies by `presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)`.
- Then it filters by `confidence_threshold` to produce `keep`.

Thus, **presence acts as a gating factor**: low presence drops detections at the score stage, excluding masks/boxes together.

---

<a id="sec-04"></a>
## Meaning of "Component Files" Read So Far (Assertions)

This section describes the purpose of files that were read, and what they contribute.

### `sam3/model/position_encoding.py` (PositionEmbeddingSine)

- **Meaning**: Adds positional information to image features (H×W grid) so transformers can distinguish spatial locations.
- **Important**: When `precompute_resolution` is used, it caches positional encodings (stride ~4/8/16/32) to avoid `torch.compile` shape-tracing pitfalls.

### `sam3/model/tokenizer_ve.py` (SimpleTokenizer)

- **Meaning**: Reads `assets/bpe_simple_vocab_16e6.txt.gz` and converts text into token id sequences using BPE.
- **Important**:
  - Reads gz via `g_pathmgr.open()` and builds the BPE merge table (encoder/decoder).
  - `__call__` packs `[SOT] + encode(text) + [EOT]` into `context_length=77` (with truncation).

### `sam3/model/text_encoder_ve.py` (VETextEncoder / TextTransformer)

- **Meaning**: Encodes token id sequences into text features via a transformer.
- **Important**:
  - `TextTransformer` uses token embedding + positional embedding + causal mask (default).
  - `text_global_pool(..., pool_type="argmax")` selects an EOT-like embedding (typical CLIP behavior).
  - Designed to work with `torch.compile` and activation checkpointing.

### `sam3/model/memory.py` (mask memory / fuser)

- **Meaning**: Used mainly for video/tracking; it downsamples masks and fuses them with visual features to build memory representations.
- **Confirmed structure (within read scope)**:
  - `SimpleMaskDownSampler`: multi-stage conv downsampling → projection to embedding dim
  - `CXBlock`: ConvNeXt-like block (depthwise conv + linear + GELU + residual)
  - `SimpleFuser`: applies CXBlocks over multiple layers
  - `SimpleMaskEncoder`: projects pixel features, adds mask, runs fuser, adds positional encoding

### `sam3/model/io_utils.py` (I/O utilities)

- **Meaning**: Converts diverse inputs (images/videos/folders/PIL lists) into normalized tensor sequences suitable for the model.
- **Important**:
  - A single image can be handled via a "single-frame video" path.
  - Supports CPU offloading via `offload_video_to_cpu`.
  - Includes async loading mechanisms for demos / large inputs.

### `sam3/perflib/masks_ops.py`

- **Meaning**: Mask operations (IoU, mask→box) used as a foundation for NMS and related utilities.
- **Assertable functions**:
  - `mask_iou(pred_masks, gt_masks)`: IoU matrix for bool masks
  - `masks_to_boxes(masks, obj_ids)`: compute bounding boxes from masks

### `sam3/perflib/nms.py`

- **Meaning**: Runs NMS to remove duplicate masks in inference results.
- **Important**:
  - Uses CUDA `torch_generic_nms` if available
  - Else falls back to Triton implementation
  - Else falls back to CPU implementation

### `sam3/perflib/connected_components.py`

- **Meaning**: Connected-components labeling + component size computation using the fastest available backend.
- **Important**:
  - CUDA: uses `cc_torch` if present, else Triton fallback
  - CPU: uses `skimage.measure.label` fallback

### `sam3/model/sam3_tracking_predictor.py` (interactive video tracking demo controller)

- **Role**: Inherits `Sam3TrackerBase` and orchestrates the interactive workflow: accept user clicks/boxes/masks, run inference per frame, consolidate temporary outputs, and propagate.
- **Meaning of key options (asserted)**:
  - `clear_non_cond_mem_around_input`: clears non-conditioning memory around input frames when adding clicks.
  - `clear_non_cond_mem_for_multi_obj`: applies the above for multiple objects.
  - `fill_hole_area`: fills small holes below a threshold area in final masks.
  - `always_start_from_first_ann_frame`: forces propagation start at the first annotated frame.
  - `max_point_num_in_prompt_enc`: limits number of points fed into prompt encoder to match training distribution.
  - `non_overlap_masks_for_output`: applies non-overlap constraint at output time.
  - Keeps an open `torch.autocast` bf16 context for memory/perf.
- **State initialization (`init_state`) includes**:
  - video meta (H/W/frames), offload flags, `storage_device`
  - per-object input dictionaries (`point_inputs_per_obj`, `mask_inputs_per_obj`)
  - split outputs into conditioning vs non-conditioning (`output_dict`, etc.)
  - mapping between client `obj_id` and server-side indices (OrderedDict)
  - temp outputs (`temp_output_dict_per_obj`) to stage user inputs before propagation
- **Object management (`_obj_id_to_idx`)**: allows creating new objects only before tracking starts; after start, new objects are rejected.
- **Input handlers (`add_new_points_or_box` / `add_new_mask`)**:
  - tensorize points/labels/boxes; optionally expand box into 2 points (labels 2/3); scale coords to model resolution
  - runs `_run_single_frame_inference(..., run_mem_encoder=False)` (skips mem encoder on user input)
  - passes previous mask logits via `prev_sam_mask_logits` to enable iterative refinement
  - stores result into `temp_output_dict_per_obj`, then consolidates across objects
  - returns `(frame_idx, obj_ids, masks_lowres=None, masks_video_res)` after resizing to video resolution
  - `add_new_mask` uses the provided mask as output at video resolution to avoid jitter; overlaps are neutralized via `NO_OBJ_SCORE`.
- **Output post-processing (`_get_orig_video_res_output`)**:
  - resize to video resolution; apply non-overlap; fill holes / remove sprinkles via `fill_holes_in_mask_scores`.

### `sam3/model/sam3_tracker_utils.py` (click reconditioning / error sampling / post-processing utilities)

- **Role**: Utility set used in interactive tracking: click generation, box extraction, frame selection, mask hole filling, etc.
- **Main functions (asserted)**:
  - `sample_box_points(...)`: creates two box corner points (labels 2/3) from masks with noise.
  - `mask_to_box(masks)`: computes `[min_x, min_y, max_x, max_y]` from masks.
  - `sample_random_points_from_errors(...)`: splits FP/FN regions, samples error points with FP negative / FN positive labels.
  - `sample_one_point_from_error_center(...)`: uses distance transform (Triton EDT) to pick a center-most error point.
  - `get_next_point(...)`: selects method `"uniform"` vs `"center"`.
  - `select_closest_cond_frames(...)`: selects temporally closest conditioning frames under a max budget.
  - `get_1d_sine_pe(...)`: 1D sine positional encoding.
  - `get_best_gt_match_from_multimasks(...)`: selects best mask by IoU (ties by score).
  - `fill_holes_in_mask_scores(...)`: fills background holes and removes tiny foreground sprinkles using `cc_torch` → Triton → skimage backends.

---

<a id="sec-05"></a>
## Progress Notes

- Read `sam3/model/sam3_tracking_predictor.py` and `sam3/model/sam3_tracker_utils.py`, and added asserted explanations of the interactive video tracking workflow.
- Remaining work (at the time this note was written) was to ensure every `sam3/` file is covered; if any files were unread, they were to be read progressively.

---

<a id="sec-06"></a>
## Additional Details (Assertions on propagation loop and memory behavior)

### Propagation loop details (`propagate_in_video`)

- If `propagate_preflight=True`, it runs `propagate_in_video_preflight` first to merge temp outputs into final outputs before propagation begins.
- Processing order is decided by `_get_processing_order` (supports forward/backward and max tracking length).
- Frames that already have conditioning output are skipped (no recomputation).
- Uncomputed frames run `_run_single_frame_inference`, store into `output_dict[non_cond_frame_outputs]`, and also build per-object slices via `_add_output_per_object`.
- Each frame yields masks resized to video resolution, with non-overlap + hole-filling applied; it also returns `object_score_logits`.

### Temp output consolidation and memory regeneration (`_consolidate_temp_output_across_obj`)

- Consolidates per-object temporary outputs into a single frame tensor; missing objects are filled with `NO_OBJ_SCORE`.
- Only when `run_mem_encoder=True`, it regenerates memory (`maskmem_features/pos_enc`) after resizing, applying non-overlap, and rerunning the memory encoder.

### Conditioning vs non-conditioning memory handling

- Input frames (clicks/masks) become "conditioning"; there are options to clear non-conditioning memory around them to discard past errors.
- Whether to treat "all frames to correct" as conditioning frames is controlled by `add_all_frames_to_correct_as_cond`.

### Cached image features reuse (`_get_image_feature`)

- Caches `(image, backbone_out)` to avoid recomputation for the same frame.
- Expands FPN outputs/pos embeddings to match object batch size and prepares shapes for decoder.

### CPU offload and bf16 memory savings

- Wraps inference in bf16 autocast for reduced memory.
- When `storage_device="cpu"`, keeps large state tensors on CPU while maintaining a path that still produces video-res masks.

### Input clearing (`clear_all_points_in_frame` / `clear_all_points_in_video`)

- Clears inputs for specific frames/objects and can downgrade frames from conditioning to non-conditioning if no conditioning remains.
- When all conditioning is removed, `_reset_tracking_results` resets state to a clean initial tracking state.

### `sam3/perflib/fa3.py`

- **Meaning**: Wraps FlashAttention 3 (FA3) as a `torch.library.custom_op`.
- **Important**:
  - Calls `flash_attn_interface.flash_attn_func`
  - Provides `register_fake` to support compilation/shape inference

### `sam3/model/box_ops.py`

- **Meaning**: Basic box transforms (cxcywh↔xyxy, etc.) and IoU/GIoU operations.
- **Confirmed use**: `Sam3Processor._forward_grounding` uses `box_cxcywh_to_xyxy`.

### `sam3/model/data_misc.py`

- **Meaning**: Provides data structures and helper functions for inference.
- **Important**:
  - `FindStage`: dataclass bundling inference target ids/inputs
  - `interpolate`: wrapper around `F.interpolate` that handles edge cases; used in `Sam3Processor._forward_grounding`

### `sam3/model/model_misc.py`

- **Meaning**: Shared building blocks (MLP, cloning, activations, scoring, etc.)
- **Important**:
  - `DotProductScoring`: pools prompt features (mean over masked region) and scores by dot product (with stabilization)
  - `TransformerWrapper`: wraps encoder/decoder together
  - `MLP`: feed-forward network implementation

### `sam3/model/necks.py` (Sam3DualViTDetNeck)

- **Meaning**: Converts ViT outputs into multi-scale FPN-like features with `d_model=256`.
- **Important**:
  - If `add_sam2_neck=True`, it deep-copies the neck to create a SAM2 path as well.

### `sam3/model/vl_combiner.py` (SAM3VLBackbone)

- **Meaning**: A wrapper that returns vision and language outputs in a shared dict (not a "fusion" in the strict sense).
- **Important**:
  - `forward_image` produces `vision_features/vision_pos_enc/backbone_fpn/sam2_backbone_out`
  - `forward_text` runs the language backbone and produces `language_features/language_mask/language_embeds`

### `sam3/model/encoder.py` / `sam3/model/decoder.py`

- **Meaning**: SAM3 transformer encoder/decoder components.
- **Important (asserted from read scope)**:
  - Encoder supports pre/post norm; has self-attn → cross-attn(image) → FFN.
  - Decoder updates queries; includes presence token insertion and DAC-style branching.

### `sam3/sam/transformer.py`

- **Meaning**: SAM-style `TwoWayTransformer` / `Attention` implementation.
- **Important**:
  - `Attention` has `use_fa3`, enabling a FA3 path when available.

### `sam3/perflib/triton/nms.py`

- **Meaning**: Triton CUDA fallback used when `torch_generic_nms` is unavailable; runs generic NMS on IoU matrices.
- **Important**:
  - Sorts by score descending; uses upper-triangular suppression against later boxes.

### `sam3/perflib/triton/connected_components.py`

- **Meaning**: Triton CUDA fallback for connected components when `cc_torch` is unavailable.
- **Important (asserted from the beginning of the file)**:
  - Initializes labels with linear indices for foreground; background is -1.
  - Implements DSU (disjoint set union) with atomics and retry handling.

### `sam3/perflib/compile.py`

- **Meaning**: Wrappers to safely apply `torch.compile`.
- **Important**:
  - `compile_wrapper`: recursively makes inputs contiguous and clones outputs to avoid aliasing.
  - `shape_logging_wrapper`: logs new shape combinations on first occurrence (debug toggle).

### `sam3/perflib/associate_det_trk.py`

- **Meaning**: Associates detections (det) with existing tracks (trk) via IoU for video tracking.
- **Important**:
  - Computes `mask_iou(det_masks, track_masks)` and runs Hungarian (`linear_sum_assignment`) for 1-to-1 assignment on tracks.
  - Allows many-to-one on detections; can treat low-IoU dets as new detections.
  - When mask resolutions differ, interpolates to the smaller resolution to save memory.

### `sam3/model/sam3_video_base.py`

- **Meaning**: Core base for video inference; integrates detector + tracker, defines per-frame processing steps and SPMD structure.
- **Important (asserted)**:
  - Centralizes thresholds linking detection→tracking (`det_nms_thresh`, `assoc_iou_thresh`, `trk_assoc_iou_thresh`, `new_det_thresh`, etc.)
  - Tracks confirmation state via `MaskletConfirmationStatus`.
  - Per-frame flow is organized into steps: detect → propagate → plan updates → apply updates → build outputs.

### `sam3/model/sam3_video_inference.py`

- **Meaning**: Inference API implementation that extends `Sam3VideoBase`; builds `inference_state` and provides prompt addition + propagation generator.
- **Important**:
  - `init_state` uses `load_resource_as_video_frames(...)` to normalize inputs into frame sequences + original H/W.
  - Builds per-frame `FindStage` and packs into `BatchedDatapoint` before moving to device.
  - Initializes an empty geometric `Prompt` (`empty_geometric_prompt`).

### `sam3/model/sam3_tracker_base.py`

- **Meaning**: Base tracking (propagation) implementation; maintains memory bank and SAM-style prompt encoder / mask decoder.
- **Important (asserted)**:
  - Uses `SimpleMaskEncoder` (`model/memory.py`) to build mask memory.
  - Combines `PromptEncoder` / `MaskDecoder` / `TwoWayTransformer` to output masks in SAM style.
  - Includes video-specific time handling (`maskmem_tpos_enc`, `no_mem_embed/no_mem_pos_enc`).

### `sam3/model/sam3_video_predictor.py`

- **Meaning**: Session manager that dispatches external requests (server/app) into `start_session/add_prompt/propagate_in_video/...`.
- **Important (asserted)**:
  - `Sam3VideoPredictor` builds the video model and keeps it in `.cuda().eval()` mode.
  - Stores session states in `_ALL_INFERENCE_STATES` and returns `propagate_in_video` as a generator/stream.
  - `Sam3VideoPredictorMultiGPU` spawns workers and sets up NCCL process groups for multi-GPU inference.

### `sam3/model/utils/misc.py`

- **Meaning**: Moves nested structures (dataclasses/dicts/lists/namedtuples) to device recursively via `.to(device)`.

### `sam3/model/utils/sam2_utils.py`

- **Meaning**: Demo video frame loading (JPEG folder/MP4) and async loading to avoid blocking session start.
- **Important**:
  - `AsyncVideoFrameLoader` loads the first frame first, then loads the rest in a background thread.
  - MP4 path may use `decord`.

### `sam3/model/utils/sam1_utils.py`

- **Meaning**: SAM2-compatible transforms and mask post-processing (hole filling, small region removal).
- **Important**:
  - `postprocess_masks` uses connected components to correct small holes/sprinkles by threshold.
  - On exceptions, it warns and skips post-processing rather than breaking outputs.

### `sam3/train/train.py`

- **Meaning**: Training entry point using Hydra + Submitit; launches `Trainer.run()` in single/multi-GPU/SLURM.
- **Important**:
  - `instantiate(cfg.trainer, _recursive_=False)` → `trainer.run()` is the execution core.
  - For Submitit, it sets `MASTER_*` and `RANK/WORLD_SIZE` env vars for distributed launch.

### `sam3/train/trainer.py`

- **Meaning**: DDP-oriented generic trainer skeleton: env → distributed init → dataloader → optimization → checkpoints → logging.
- **Important**:
  - Uses dataclass configs like `DistributedConf/CudaConf/CheckpointConf/LoggingConf`.

### `sam3/train/matcher.py`

- **Meaning**: Training-time assignment of predictions to GT (Hungarian / BinaryHungarian).
- **Important**:
  - Uses SciPy `linear_sum_assignment`, combining class/bbox/giou costs.

### `sam3/train/data/collator.py`

- **Meaning**: Packs training/inference API inputs into a model-consumable `BatchedDatapoint`.
- **Important**:
  - Builds `FindStage` / `BatchedFindTarget` / `BatchedInferenceMetadata` per stage, packing/padding ids, boxes, points.

---

<a id="sec-07"></a>
## Meaning of `sam3/train/configs/*.yaml` (Read portion)

- **Meaning**: Hydra experiment configs for training/evaluation.
- **What they define (high level)**:
  - model parameters and architecture toggles
  - dataset sources and paths
  - transforms / sampling strategies
  - loss composition and weights
  - distributed settings

---

<a id="sec-08"></a>
## Unread (Not asserted at this point)

If any file under `sam3/` was not explicitly read at the time of writing, it must be treated as **Unread** and not asserted beyond what is provable (e.g., file presence / import references).

---

<a id="sec-09"></a>
## ComfyUI Node Integration + `COLORCODE` (JS widget) (Assertion with code evidence)

### Python side

- `ComfyUI/custom_nodes/comfyui_facetools_without_mediapipe/__init__.py`
  - Registers nodes into `NODE_CLASS_MAPPINGS`.
  - Exposes `WEB_DIRECTORY = "./js"` so frontend JS is loaded.

### JS side (`COLORCODE`)

- `ComfyUI/custom_nodes/comfyui_facetools_without_mediapipe/js/sam3_color_widget.js`
  - Registers a ComfyUI extension with a unique name.
  - Provides `getCustomWidgets()` to define a `COLORCODE` widget.
  - Uses `beforeRegisterNodeDef` and `nodeCreated` to detect `COLORCODE` inputs and place the widget at the top.

---

<a id="sec-10"></a>
## Meaning of `sam3/model_builder.py` (Assertion: read)

- **Meaning**: "Factory" entry point for building SAM3 models (image/video), including checkpoint loading and optional ablations.
- **Assertable composition from imports**:
  - Vision: `sam3/model/vitdet.py` (`ViT`)
  - Neck: `sam3/model/necks.py` (`Sam3DualViTDetNeck`)
  - Positional encoding: `sam3/model/position_encoding.py` (`PositionEmbeddingSine`)
  - Text: `sam3/model/tokenizer_ve.py` + `sam3/model/text_encoder_ve.py`
  - VL backbone: `sam3/model/vl_combiner.py` (`SAM3VLBackbone`)
  - Transformer: `sam3/model/encoder.py` + `sam3/model/decoder.py` (wrapped via `sam3/model/model_misc.py`)
  - Segmentation head: `sam3/model/maskformer_segmentation.py` (`PixelDecoder` + `UniversalSegmentationHead`)
  - Geometry: `sam3/model/geometry_encoders.py` (`SequenceGeometryEncoder`)
  - Scoring: `sam3/model/model_misc.py` (`DotProductScoring`, etc.)

---

<a id="sec-11"></a>
## Core files for video inference (purpose / I/O / call relations)

Covered in earlier asserted sections; the core set is:

- `sam3/model/sam3_video_base.py`
- `sam3/model/sam3_video_inference.py`
- `sam3/model/sam3_video_predictor.py`
- `sam3/model/sam3_tracker_base.py`
- `sam3/model/sam3_tracking_predictor.py`
- `sam3/model/sam3_tracker_utils.py`

---

<a id="sec-12"></a>
## Image features, geometric prompts, segmentation (Assertions)

Key files (purpose-level assertions):

- `sam3/model/vitdet.py`: vision transformer backbone (ViTDet-style), with RoPE/window attention design.
- `sam3/model/geometry_encoders.py`: geometric `Prompt` dataclass + prompt encoders for points/boxes/masks.
- `sam3/model/maskformer_segmentation.py`: pixel decoder + segmentation heads including `UniversalSegmentationHead`.
- `sam3/model/box_ops.py`: bbox conversions/IoU ops used in processor output mapping.

---

<a id="sec-13"></a>
## Transformer encoder/decoder and backbone integration (Assertions)

- `sam3/model/encoder.py`: transformer encoder layers (self-attn, cross-attn(image), FFN, norm variants).
- `sam3/model/decoder.py`: transformer decoder layers (query updates, cross-attn(image/text), presence token, branching).
- `sam3/model/vl_combiner.py`: returns vision + language features in a shared dict.
- `sam3/model/necks.py`: builds FPN-like multi-scale features for decoder consumption.

---

<a id="sec-14"></a>
## SAM2-compatible transformer and SAM1-compatible predictor (Assertions)

- `sam3/sam/transformer.py`: SAM-style `TwoWayTransformer` and attention blocks.
- `sam3/model/sam1_task_predictor.py`: SAM1-compatible image predictor using SAM2-like transforms.

---

<a id="sec-15"></a>
## SAM2 prompt encoder / mask decoder and RoPE implementation (Assertions)

- `sam3/sam/prompt_encoder.py`: encodes points/boxes/masks into sparse/dense prompt embeddings.
- `sam3/sam/mask_decoder.py`: predicts masks and mask quality using a transformer decoder.
- `sam3/sam/rope.py`: 2D rotary position embedding implementation for vision transformers.

---

<a id="sec-16"></a>
## Overall picture of ComfyUI integration (complete explanation of files actually modified)

In this workspace, SAM3 is integrated into `comfyui_facetools_without_mediapipe`:

- `ComfyUI/custom_nodes/comfyui_facetools_without_mediapipe/__init__.py`
  - merges SAM3 node mappings into the extension
  - sets `WEB_DIRECTORY = "./js"` to load JS widgets
- `ComfyUI/custom_nodes/comfyui_facetools_without_mediapipe/AILab_SAM3Segment.py`
  - provides the ComfyUI SAM3 segmentation node (current node id is `SAM3SegmentV3`)
  - converts between ComfyUI tensors and PIL for inference + post-processing
- `ComfyUI/custom_nodes/comfyui_facetools_without_mediapipe/js/sam3_color_widget.js`
  - provides the `COLORCODE` widget so `background_color` is selectable from UI

Note: SAM3 integration was removed from `ComfyUI/custom_nodes/ComfyUI-NunchakuFluxLoraStacker/` in this workspace.

---

<a id="sec-17"></a>
## Missed items (files found mechanically but not mentioned earlier)

If any file under `sam3/` was present but not listed in earlier notes, it must be:

1. Added to the documentation with its purpose and role
2. Marked **Unread** if it has not been opened and reviewed


