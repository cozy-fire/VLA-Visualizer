import socket
import threading
import webbrowser
from dataclasses import dataclass
from typing import Any

from flask import Flask, jsonify, render_template, request

from custom_visualizer.common import (
    ALL,
    CaptureResult,
    build_dynamic_frames,
    common_head_indices,
    filter_display_items,
    load_lerobot_frame_page,
    load_settings,
    phase2_entries,
    save_settings,
    serialize_capture_context,
    serialize_display_item,
)

PAGE_SIZE = 12
FRAME_PREVIEW_PAGE_SIZE = 24


@dataclass
class ViewerState:
    adapter: Any
    context: dict[str, Any] | None = None
    result: CaptureResult | None = None
    lock: threading.Lock = threading.Lock()


def create_app(adapter: Any, persist_settings: bool = True) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    state = ViewerState(adapter=adapter)
    app.config["VIEWER_STATE"] = state

    def load_adapter_settings():
        kwargs = {"path": adapter.settings_path} if getattr(adapter, "settings_path", None) else {}
        return load_settings(adapter.default_checkpoint, adapter.default_dataset, **kwargs)

    def save_adapter_settings(checkpoint: str, dataset: str) -> None:
        kwargs = {"path": adapter.settings_path} if getattr(adapter, "settings_path", None) else {}
        save_settings(checkpoint, dataset, **kwargs)

    @app.get("/")
    def index():
        return render_template("index.html", app_name=f"{adapter.name} Attention Viewer")

    @app.get("/api/settings")
    def api_settings():
        if persist_settings:
            settings = load_adapter_settings()
        else:
            settings = {
                "checkpoint": adapter.default_checkpoint,
                "dataset": adapter.default_dataset,
                "checkpoint_history": [adapter.default_checkpoint] if adapter.default_checkpoint else [],
                "dataset_history": [adapter.default_dataset] if adapter.default_dataset else [],
            }
        return jsonify({
            "adapter": adapter.name,
            "checkpoint": settings["checkpoint"],
            "dataset": settings["dataset"],
            "checkpoint_history": settings.get("checkpoint_history", []),
            "dataset_history": settings.get("dataset_history", []),
            "default_checkpoint": adapter.default_checkpoint,
            "default_dataset": adapter.default_dataset,
        })

    @app.get("/api/dataset/frames")
    def api_dataset_frames():
        dataset = str(request.args.get("dataset", "")).strip()
        if not dataset:
            return jsonify({"ok": False, "error": "Please provide dataset path or repo id"}), 400
        page = max(0, _int_arg("page", 0))
        page_size = max(1, _int_arg("page_size", FRAME_PREVIEW_PAGE_SIZE))
        stride = max(1, _int_arg("stride", 1))
        try:
            data = load_lerobot_frame_page(dataset, page=page, page_size=page_size, stride=stride)
            return jsonify({"ok": True, **data})
        except Exception as exc:
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500

    @app.post("/api/capture")
    def api_capture():
        payload = request.get_json(silent=True) or {}
        checkpoint = str(payload.get("checkpoint", "")).strip()
        dataset = str(payload.get("dataset", "")).strip()
        try:
            frame_idx = int(payload.get("frame_idx", 0))
            if frame_idx < 0:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "frame_idx must be a non-negative integer"}), 400
        if not checkpoint or not dataset:
            return jsonify({"ok": False, "error": "Please provide checkpoint and dataset"}), 400

        with state.lock:
            try:
                if persist_settings:
                    save_adapter_settings(checkpoint, dataset)
                if state.context and not adapter.can_reuse_context(state.context, checkpoint):
                    return jsonify({"ok": False, "error": adapter.restart_required_message}), 400
                if state.context is None:
                    state.context = adapter.load_context(checkpoint)
                state.result = adapter.capture(state.context, dataset, frame_idx)
                response = {"ok": True, "frame_idx": frame_idx, **_capture_summary(state.result)}
                response.update(serialize_capture_context(state.result))
                if persist_settings:
                    settings = load_adapter_settings()
                    response.update({
                        "checkpoint": settings["checkpoint"],
                        "dataset": settings["dataset"],
                        "checkpoint_history": settings.get("checkpoint_history", []),
                        "dataset_history": settings.get("dataset_history", []),
                    })
                return jsonify(response)
            except Exception as exc:
                return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500

    @app.get("/api/items")
    def api_items():
        result = _require_result(state)
        if isinstance(result, tuple):
            return result
        selections = _parse_selections(request.args)
        page = max(0, _int_arg("page", 0))
        items = filter_display_items(result.cache, **selections)
        start = page * PAGE_SIZE
        visible = items[start:start + PAGE_SIZE]
        return jsonify({
            "ok": True,
            "page": page,
            "page_size": PAGE_SIZE,
            "total": len(items),
            "total_pages": max(1, (len(items) + PAGE_SIZE - 1) // PAGE_SIZE),
            "items": _serialize_items(visible),
            "overlay_available": not bool(result.layout.overlay_reason),
            "overlay_reason": result.layout.overlay_reason,
        })

    @app.get("/api/frames")
    def api_frames():
        result = _require_result(state)
        if isinstance(result, tuple):
            return result
        mode = _mode_arg(request.args.get("mode", "static"))
        selections = _parse_selections(request.args)
        if mode == "static":
            return jsonify({"ok": True, "mode": mode, "frames": 0, "axis_values": []})
        try:
            frames = build_dynamic_frames(result.cache, mode, **selections)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc), "frames": 0, "axis_values": []}), 400
        return jsonify({
            "ok": True,
            "mode": mode,
            "frames": len(frames),
            "axis_values": [axis for axis, _ in frames],
        })

    @app.get("/api/dynamic")
    def api_dynamic():
        result = _require_result(state)
        if isinstance(result, tuple):
            return result
        mode = _mode_arg(request.args.get("mode", "steps"))
        selections = _parse_selections(request.args)
        try:
            frames = build_dynamic_frames(result.cache, mode, **selections)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc), "frames": []}), 400
        axis_name = {"steps": "step", "layers": "layer", "heads": "head"}[mode]
        return jsonify({
            "ok": True,
            "mode": mode,
            "axis_name": axis_name,
            "frame_count": len(frames),
            "frames": [
                {
                    "frame_index": idx,
                    "axis_value": axis_value,
                    "items": _serialize_items(items),
                }
                for idx, (axis_value, items) in enumerate(frames)
            ],
            "overlay_available": not bool(result.layout.overlay_reason),
            "overlay_reason": result.layout.overlay_reason,
        })

    @app.get("/api/frame")
    def api_frame():
        result = _require_result(state)
        if isinstance(result, tuple):
            return result
        mode = _mode_arg(request.args.get("mode", "steps"))
        selections = _parse_selections(request.args)
        frame_index = max(0, _int_arg("frame_index", 0))
        try:
            frames = build_dynamic_frames(result.cache, mode, **selections)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if not frames:
            return jsonify({"ok": False, "error": "No matching dynamic frame"}), 404
        frame_index = min(frame_index, len(frames) - 1)
        axis_value, items = frames[frame_index]
        axis_name = {"steps": "step", "layers": "layer", "heads": "head"}[mode]
        return jsonify({
            "ok": True,
            "frame_index": frame_index,
            "frame_count": len(frames),
            "axis_name": axis_name,
            "axis_value": axis_value,
            "items": _serialize_items(items),
            "overlay_available": not bool(result.layout.overlay_reason),
            "overlay_reason": result.layout.overlay_reason,
        })

    return app


def _capture_summary(result: CaptureResult) -> dict[str, Any]:
    entries = phase2_entries(result.cache)
    steps = sorted({e["step"] for e in entries})
    layers = sorted({e["layer_idx"] for e in entries})
    heads = common_head_indices(entries)
    return {
        "summary": result.summary or (
            f"Phase 2 {len(entries)} entries; steps={len(steps)}, layers={len(layers)}, "
            f"heads={len(heads)}; images={len(result.image_keys)}"
        ),
        "steps": steps,
        "layers": layers,
        "heads": heads,
        "image_count": len(result.image_keys),
        "overlay_available": not bool(result.layout.overlay_reason),
        "overlay_reason": result.layout.overlay_reason,
    }


def _serialize_items(items) -> list[dict[str, Any]]:
    return [serialize_display_item(item) for item in items]


def _require_result(state: ViewerState):
    if state.result is None:
        return jsonify({"ok": False, "error": "Cache has not been captured yet"}), 400
    return state.result


def _parse_selections(args) -> dict[str, int | None]:
    return {
        "step": _optional_int(args.get("step")),
        "layer": _optional_int(args.get("layer")),
        "head": _optional_int(args.get("head")),
    }


def _optional_int(value: Any) -> int | None:
    if value in (None, "", ALL, "all", "全部"):
        return None
    return int(value)


def _mode_arg(value: Any) -> str:
    mode = str(value or "static")
    return "static" if mode in {"static", ALL, "全部"} else mode


def _int_arg(name: str, default: int) -> int:
    try:
        return int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default


def find_free_port(host: str = "127.0.0.1", start_port: int = 7860, attempts: int = 20) -> int:
    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"Could not find a free port in {host}:{start_port}-{start_port + attempts - 1}")


def run_app(adapter: Any, host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True) -> None:
    app = create_app(adapter)
    actual_port = find_free_port(host, port)
    url = f"http://{host}:{actual_port}"
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"{adapter.name} Attention Viewer running at {url}")
    app.run(host=host, port=actual_port, debug=False, use_reloader=False)
