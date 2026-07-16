# VLA-Visualizer

> This repository is modified from [luo3300612/Visualizer](https://github.com/luo3300612/Visualizer). The original `visualizer.get_local` bytecode instrumentation is retained, while this fork focuses on SmolVLA attention cache capture and web-based visualization.

VLA-Visualizer is a local attention inspection tool for Vision-Language-Action models. The current implementation is centered on **SmolVLA**: it captures Phase 2 denoising attention from a selected LeRobot data frame and visualizes attention across steps, layers, and heads in a Flask dashboard.

![SmolVLA attention viewer demo](assets/smolvla_attention_viewer.gif)

## Current focus: SmolVLA attention visualization

The SmolVLA viewer is designed for analyzing how action tokens attend to visual, language, state, and action-prefix tokens during denoising.

Main capabilities:

- Select a SmolVLA checkpoint and LeRobot dataset path from persisted history.
- Preview dataset frames before capture; the frame picker samples frames by stride for faster browsing.
- Capture one selected data frame into an in-memory Phase 2 attention cache.
- Browse attention maps by `step`, `layer`, and `head_idx`.
- Use static pagination or dynamic playback across:
  - `steps`
  - `layers`
  - `heads`
- Render two modes:
  - matrix mode: full attention matrix with token boundary markers;
  - image overlay mode: image-token attention projected back onto the original observation image.
- Support multi-view image observations using keys under `observation.images.*`.

The common viewer logic is separated from model-specific capture logic:

```text
custom_visualizer/
  common/              # dataset loading, image collection, filtering, layouts, serialization
  web/                 # Flask app, API routes, HTML/CSS/JS frontend
  policies/smolvla/    # SmolVLA checkpoint loading, get_local instrumentation, cache annotation
visualizer/            # original get_local bytecode instrumentation utility
```

## Quick start

Run the SmolVLA web viewer from the repository root:

```powershell
conda activate lerobot
python -m custom_visualizer.policies.smolvla.viewer
```

By default the Flask server listens on `127.0.0.1:7860` and opens the browser automatically. If the port is occupied, it falls back to the next available port.

In this project, all Python commands should be executed inside the `lerobot` Conda environment.

## Default paths

The SmolVLA adapter currently uses these defaults:

```text
Checkpoint:
D:\CodeProject\ModelTrainRepo\Checkpoint\pick_pen_give_human\smolvla\checkpoints\020000\pretrained_model

Dataset:
D:\Huggingface Cache\lerobot\CozyFire\pick_pen_give_human
```

The web UI persists checkpoint and dataset history. After a successful capture, the last used paths become the default values for the next launch.

## Viewer workflow

1. Open the web viewer.
2. Select or enter:
   - SmolVLA checkpoint path;
   - LeRobot dataset path or repo;
   - frame index through the frame picker.
3. Click **确认并运行采集** / run capture.
4. Browse the resulting attention cache:
   - leave filters as `All` to inspect all available maps;
   - set `step`, `layer`, or `head_idx` to narrow the view;
   - switch between static mode and dynamic playback.
5. Toggle image overlay mode when you want to inspect image-token attention on the original observation image.

## Attention cache semantics

The SmolVLA adapter captures `probs` from:

```text
SmolVLMWithExpertModel.eager_attention_forward
```

Captured Phase 2 entries are annotated with:

- `phase`
- `step`
- `layer_idx`
- `type`
- `probs`

The expected denoising attention shapes are typically:

```text
(B, H, 50, 113)  # expert_cross_attn
(B, H, 50, 163)  # self_attn
```

In the current SmolVLA denoising path, the query axis corresponds to action tokens. The key axis contains image/language/state prefix tokens, and for self-attention also action tokens.

## Image overlay mode

When image overlay mode is enabled, each attention map is rendered by:

1. taking only the image-token columns from the selected head;
2. averaging attention over action tokens;
3. splitting image tokens across views when multiple `observation.images.*` fields exist;
4. reshaping patch scores to the inferred patch grid;
5. projecting the patch heatmap back to the original image area;
6. drawing the heatmap in the same card used by matrix mode.

The original dataset images are never modified.

## Development checks

Recommended checks:

```powershell
conda activate lerobot
python -m compileall -q custom_visualizer
python -m unittest discover -s tests
node --check custom_visualizer/web/static/app.js
```

If you run commands through a non-interactive shell, use:

```powershell
conda run -n lerobot python -m unittest discover -s tests
```

## Notes on the original Visualizer

The original project provides `get_local`, a bytecode-based helper for collecting local variables inside Python functions without manually returning them. This fork keeps that mechanism because SmolVLA attention probabilities are captured from inside the model's attention implementation.

For the upstream implementation and general-purpose examples, see [luo3300612/Visualizer](https://github.com/luo3300612/Visualizer).

## License

This repository keeps the upstream license unless otherwise stated.
