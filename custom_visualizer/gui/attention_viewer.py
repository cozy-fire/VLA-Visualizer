import math
import threading
from typing import Any, Sequence

import numpy as np

from custom_visualizer.common import (
    ALL,
    STATIC_COLUMNS,
    STATIC_VISIBLE_ROWS,
    CaptureResult,
    DisplayItem,
    TokenLayout,
    build_dynamic_frames,
    common_head_indices,
    compose_overlay_image,
    filter_display_items,
    load_settings,
    phase2_entries,
    save_settings,
    static_window,
)


class AttentionViewer:
    def __init__(self, root: Any, adapter: Any) -> None:
        import tkinter as tk
        from tkinter import font as tkfont
        from tkinter import ttk
        import matplotlib
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.root = root
        self.adapter = adapter
        font_family = "Microsoft YaHei UI"
        root.option_add("*Font", (font_family, 10))
        for font_name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont"):
            try:
                tkfont.nametofont(font_name).configure(family=font_family, size=10)
            except tk.TclError:
                pass
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("TButton", padding=(10, 6))
        style.configure("Primary.TButton", padding=(14, 8), font=(font_family, 10, "bold"))
        style.configure("TLabelframe", padding=4)
        style.configure("TLabelframe.Label", font=(font_family, 10, "bold"))
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        root.title(f"{adapter.name} Attention Viewer")
        root.geometry("1500x950")
        root.minsize(980, 700)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        self.cache: list[dict[str, Any]] = []
        self.layout: TokenLayout | None = None
        self.source_images: list[np.ndarray] = []
        self.model_context: dict[str, Any] | None = None
        self.static_row = 0
        self.current_frame = 0
        self.after_id: str | None = None
        self.playing = False

        settings = load_settings(adapter.default_checkpoint, adapter.default_dataset)
        self.checkpoint_var = tk.StringVar(value=settings["checkpoint"])
        self.dataset_var = tk.StringVar(value=settings["dataset"])
        self.frame_var = tk.StringVar(value="0")
        self.mode_var = tk.StringVar(value="静态")
        self.step_var = tk.StringVar(value=ALL)
        self.layer_var = tk.StringVar(value=ALL)
        self.head_var = tk.StringVar(value=ALL)
        self.overlay_var = tk.BooleanVar(value=False)
        self.loop_var = tk.BooleanVar(value=True)
        self.interval_var = tk.StringVar(value="600")
        self.status_var = tk.StringVar(value="请选择 checkpoint、数据集和数据帧")
        self.position_var = tk.StringVar(value="尚未采集 cache")

        source = ttk.LabelFrame(root, text="数据采集", padding=8)
        source.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        source.columnconfigure(1, weight=1)
        ttk.Label(source, text="Checkpoint").grid(row=0, column=0, sticky="w")
        ttk.Entry(source, textvariable=self.checkpoint_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(source, text="浏览", command=self._browse_checkpoint).grid(row=0, column=2)
        ttk.Label(source, text="数据集").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(source, textvariable=self.dataset_var).grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        ttk.Button(source, text="浏览", command=self._browse_dataset).grid(row=1, column=2, pady=(6, 0))
        ttk.Label(source, text="帧索引（0-based）").grid(row=0, column=3, padx=(14, 4))
        ttk.Entry(source, textvariable=self.frame_var, width=8).grid(row=0, column=4)
        self.capture_button = ttk.Button(source, text="运行并采集 Cache", command=self._start_capture, style="Primary.TButton")
        self.capture_button.grid(row=1, column=3, columnspan=2, sticky="ew", padx=(14, 0), pady=(6, 0))

        controls = ttk.LabelFrame(root, text="浏览与播放", padding=8)
        controls.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        specs = (
            ("mode", "模式", self.mode_var, ("静态", "steps", "layers", "heads")),
            ("step", "Step", self.step_var, (ALL,)),
            ("layer", "Layer", self.layer_var, (ALL,)),
            ("head", "Head", self.head_var, (ALL,)),
        )
        self.combos: dict[str, Any] = {}
        for column, (key, label, variable, values) in enumerate(specs):
            ttk.Label(controls, text=label).grid(row=0, column=column * 2, padx=(0 if column == 0 else 10, 4))
            combo = ttk.Combobox(controls, textvariable=variable, values=values, width=10, state="disabled")
            combo.grid(row=0, column=column * 2 + 1)
            combo.bind("<<ComboboxSelected>>", self._on_control_change)
            self.combos[key] = combo

        self.prev_button = ttk.Button(controls, text="上一帧", command=self._previous, state="disabled")
        self.prev_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.play_button = ttk.Button(controls, text="播放", command=self._toggle_play, state="disabled")
        self.play_button.grid(row=1, column=2, columnspan=2, sticky="ew", padx=6, pady=(8, 0))
        self.next_button = ttk.Button(controls, text="下一帧", command=self._next, state="disabled")
        self.next_button.grid(row=1, column=4, columnspan=2, sticky="ew", pady=(8, 0))
        self.overlay_check = ttk.Checkbutton(
            controls, text="叠加到原图（关闭=矩阵热力图）", variable=self.overlay_var,
            command=self._on_control_change, state="disabled"
        )
        self.overlay_check.grid(row=1, column=6, columnspan=2, sticky="w", padx=(14, 4), pady=(8, 0))
        ttk.Label(controls, text="间隔(ms)").grid(row=1, column=8, padx=(10, 4), pady=(8, 0))
        self.interval_spin = ttk.Spinbox(controls, from_=100, to=5000, increment=100,
                                         width=7, textvariable=self.interval_var)
        self.interval_spin.grid(row=1, column=9, pady=(8, 0))
        ttk.Checkbutton(controls, text="循环", variable=self.loop_var).grid(row=1, column=10, padx=8, pady=(8, 0))
        ttk.Label(controls, textvariable=self.position_var).grid(
            row=2, column=0, columnspan=12, sticky="w", pady=(8, 0))

        plot_frame = ttk.Frame(root)
        plot_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        self.figure = Figure(figsize=(14, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")
        canvas_widget.bind("<MouseWheel>", self._on_mousewheel)
        self.static_scrollbar = ttk.Scrollbar(
            plot_frame, orient="vertical", command=self._on_static_scroll
        )
        self.static_scrollbar.grid(row=0, column=1, sticky="ns", padx=(6, 0))
        self.static_scrollbar.set(0.0, 1.0)
        self.static_scrollbar.state(["disabled"])
        ttk.Label(root, textvariable=self.status_var, anchor="w").grid(
            row=3, column=0, sticky="ew", padx=10, pady=(0, 8))
        self._show_message("采集 cache 后可浏览 attention map")
        root.protocol("WM_DELETE_WINDOW", self._close)

    @staticmethod
    def _selection(value: str) -> int | None:
        return None if value == ALL else int(value)

    def _browse_checkpoint(self) -> None:
        from tkinter import filedialog
        value = filedialog.askdirectory(title=f"选择 {self.adapter.name} checkpoint 目录")
        if value:
            self.checkpoint_var.set(value)

    def _browse_dataset(self) -> None:
        from tkinter import filedialog
        value = filedialog.askdirectory(title="选择 LeRobot 数据集目录")
        if value:
            self.dataset_var.set(value)

    def _start_capture(self) -> None:
        checkpoint, dataset_repo = self.checkpoint_var.get().strip(), self.dataset_var.get().strip()
        try:
            frame_idx = int(self.frame_var.get())
            if frame_idx < 0:
                raise ValueError
        except ValueError:
            self.status_var.set("帧索引必须是非负整数")
            return
        if not checkpoint or not dataset_repo:
            self.status_var.set("请填写 checkpoint 和数据集")
            return
        try:
            save_settings(checkpoint, dataset_repo)
        except OSError as exc:
            self.status_var.set(f"设置保存失败：{exc}")
            return
        if self.model_context and not self.adapter.can_reuse_context(self.model_context, checkpoint):
            self.status_var.set(self.adapter.restart_required_message)
            return
        self._stop_playback()
        self.capture_button.configure(state="disabled")
        self.status_var.set("正在加载模型、数据集并采集 cache…")
        threading.Thread(target=self._capture_worker,
                         args=(checkpoint, dataset_repo, frame_idx), daemon=True).start()

    def _capture_worker(self, checkpoint: str, dataset_repo: str, frame_idx: int) -> None:
        try:
            if self.model_context is None:
                self.model_context = self.adapter.load_context(checkpoint)
            result = self.adapter.capture(self.model_context, dataset_repo, frame_idx)
            self.root.after(0, lambda: self._capture_done(result))
        except Exception as exc:
            self.root.after(0, lambda error=exc: self._capture_failed(error))

    def _capture_done(self, result: CaptureResult) -> None:
        self.cache, self.layout = result.cache, result.layout
        self.source_images = result.images
        entries = phase2_entries(self.cache)
        steps = sorted({e["step"] for e in entries})
        layers = sorted({e["layer_idx"] for e in entries})
        heads = common_head_indices(entries)
        for key, values in (("step", steps), ("layer", layers), ("head", heads)):
            self.combos[key].configure(values=(ALL, *map(str, values)), state="readonly")
        self.combos["mode"].configure(state="readonly")
        self.mode_var.set("静态")
        self.step_var.set(ALL); self.layer_var.set(ALL); self.head_var.set(ALL)
        self.capture_button.configure(state="normal")
        if self.layout.overlay_reason:
            self.overlay_var.set(False)
            self.overlay_check.configure(state="disabled", text="叠加到原图（当前数据不可用）")
        else:
            self.overlay_check.configure(state="normal", text="叠加到原图（关闭=矩阵热力图）")
        suffix = f"；叠加不可用：{self.layout.overlay_reason}" if self.layout.overlay_reason else ""
        summary = result.summary or (
            f"Phase 2 {len(entries)} 条，steps={len(steps)}, layers={len(layers)}, "
            f"heads={len(heads)}；底图={len(result.image_keys)} 张"
        )
        self.status_var.set(f"采集完成：{summary}{suffix}")
        self._update_control_states()
        self._render(reset=True)

    def _capture_failed(self, error: Exception) -> None:
        self.capture_button.configure(state="normal")
        self.status_var.set(f"采集失败：{type(error).__name__}: {error}")

    def _on_control_change(self, event: Any = None) -> None:
        if not self.cache:
            return
        self._stop_playback()
        self.static_row = self.current_frame = 0
        self._update_control_states()
        self._render(reset=True)

    def _update_control_states(self) -> None:
        for key in ("step", "layer", "head"):
            self.combos[key].configure(state="readonly")
        mode = self.mode_var.get()
        if mode != "静态":
            axis = {"steps": "step", "layers": "layer", "heads": "head"}[mode]
            {"step": self.step_var, "layer": self.layer_var, "head": self.head_var}[axis].set(ALL)
            self.combos[axis].configure(state="disabled")

    def _selections(self) -> dict[str, int | None]:
        return {"step": self._selection(self.step_var.get()),
                "layer": self._selection(self.layer_var.get()),
                "head": self._selection(self.head_var.get())}

    def _render(self, reset: bool = False) -> None:
        if not self.cache:
            return
        if self.mode_var.get() == "静态":
            self._render_static(reset)
        else:
            self._render_dynamic(reset)

    def _render_static(self, reset: bool) -> None:
        if reset:
            self.static_row = 0
        items = filter_display_items(self.cache, **self._selections())
        if not items:
            self.static_scrollbar.set(0.0, 1.0)
            self.static_scrollbar.state(["disabled"])
            self._show_message("当前筛选条件没有匹配的 attention map")
            return
        self.static_row, total_rows, visible = static_window(items, self.static_row)
        start = self.static_row * STATIC_COLUMNS
        self._draw_items(visible, f"Phase 2 Attention · {len(items)} maps · 滚动按需加载")
        self.position_var.set(
            f"当前显示 {start + 1}–{start + len(visible)} / {len(items)} 张 · "
            f"视口仅渲染 {len(visible)} 张"
        )
        if total_rows > STATIC_VISIBLE_ROWS:
            self.static_scrollbar.state(["!disabled"])
            self.static_scrollbar.set(
                self.static_row / total_rows,
                min(1.0, (self.static_row + STATIC_VISIBLE_ROWS) / total_rows),
            )
        else:
            self.static_scrollbar.set(0.0, 1.0)
            self.static_scrollbar.state(["disabled"])
        self.prev_button.configure(state="disabled")
        self.next_button.configure(state="disabled")
        self.play_button.configure(state="disabled", text="播放")

    def _frames(self) -> list[tuple[int, list[DisplayItem]]]:
        return build_dynamic_frames(self.cache, self.mode_var.get(), **self._selections())

    def _render_dynamic(self, reset: bool) -> None:
        self.static_scrollbar.set(0.0, 1.0)
        self.static_scrollbar.state(["disabled"])
        if reset:
            self.current_frame = 0
        try:
            frames = self._frames()
        except ValueError as exc:
            self._show_message(str(exc))
            self.position_var.set("请在另外两个参数中恰好指定一个")
            self.prev_button.configure(state="disabled")
            self.next_button.configure(state="disabled")
            self.play_button.configure(state="disabled", text="播放")
            return
        self.current_frame = min(self.current_frame, len(frames) - 1)
        axis_value, items = frames[self.current_frame]
        mode = self.mode_var.get()
        axis_name = {"steps": "step", "layers": "layer", "heads": "head"}[mode]
        self._draw_items(items, f"Dynamic {mode} · {axis_name}={axis_value}")
        self.position_var.set(
            f"第 {self.current_frame + 1}/{len(frames)} 帧 · {axis_name}={axis_value} · {len(items)} 张")
        self.prev_button.configure(state="normal" if self.current_frame else "disabled")
        self.next_button.configure(
            state="normal" if self.current_frame + 1 < len(frames) or self.loop_var.get() else "disabled")
        self.play_button.configure(state="normal", text="暂停" if self.playing else "播放")

    def _draw_items(self, items: Sequence[DisplayItem], title: str) -> None:
        self.figure.clear()
        cols = min(4, max(1, len(items)))
        rows = math.ceil(len(items) / cols)
        axes = np.asarray(self.figure.subplots(rows, cols, squeeze=False)).ravel()
        for axis, item in zip(axes, items):
            self._draw_item(axis, item)
        for axis in axes[len(items):]:
            axis.axis("off")
        self.figure.suptitle(title)
        self.figure.subplots_adjust(
            left=0.035, right=0.99, bottom=0.045, top=0.91,
            wspace=0.22, hspace=0.38,
        )
        self.canvas.draw_idle()

    def _draw_item(self, axis: Any, item: DisplayItem) -> None:
        entry, head = item.entry, item.head_idx
        title = f"S{entry['step']} · L{entry['layer_idx']} · H{head}\n{entry['type']}"
        if self.overlay_var.get() and self.layout and not self.layout.overlay_reason:
            self._draw_overlay(axis, entry, head)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
            return
        axis.imshow(np.asarray(entry["probs"])[0, head], cmap="viridis", aspect="auto")
        self._draw_partitions(axis, entry)
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Key tokens", fontsize=8)
        axis.set_ylabel("Action tokens", fontsize=8)
        axis.set_xticks([]); axis.set_yticks([])

    def _draw_partitions(self, axis: Any, entry: dict[str, Any]) -> None:
        if not self.layout:
            return
        image_end = self.layout.image_tokens
        language_end = image_end + self.layout.language_tokens
        axis.axvline(image_end - 0.5, color="white", linewidth=1.2, linestyle="--")
        axis.axvline(language_end - 0.5, color="white", linewidth=1.2, linestyle="--")
        if entry["type"] == "self_attn":
            axis.axvline(self.layout.prefix_tokens - 0.5, color="red", linewidth=1.6)

    def _draw_overlay(self, axis: Any, entry: dict[str, Any], head: int) -> None:
        if not self.source_images or self.layout is None:
            raise RuntimeError("原图或 token 布局尚未准备")
        axis.imshow(compose_overlay_image(self.source_images, entry, head, self.layout))

    def _show_message(self, message: str) -> None:
        self.figure.clear()
        axis = self.figure.subplots()
        axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
        axis.axis("off")
        self.canvas.draw_idle()

    def _on_static_scroll(self, *args: str) -> None:
        if not self.cache or self.mode_var.get() != "静态" or not args:
            return
        items = filter_display_items(self.cache, **self._selections())
        total_rows = max(1, math.ceil(len(items) / STATIC_COLUMNS))
        max_start = max(0, total_rows - STATIC_VISIBLE_ROWS)
        if args[0] == "moveto":
            self.static_row = min(max_start, max(0, round(float(args[1]) * total_rows)))
        elif args[0] == "scroll":
            amount = int(args[1])
            step = STATIC_VISIBLE_ROWS if args[2] == "pages" else 1
            self.static_row = min(max_start, max(0, self.static_row + amount * step))
        self._render(False)

    def _on_mousewheel(self, event: Any) -> str | None:
        if not self.cache or self.mode_var.get() != "静态":
            return None
        direction = -1 if event.delta > 0 else 1
        items = filter_display_items(self.cache, **self._selections())
        total_rows = max(1, math.ceil(len(items) / STATIC_COLUMNS))
        max_start = max(0, total_rows - STATIC_VISIBLE_ROWS)
        self.static_row = min(max_start, max(0, self.static_row + direction))
        self._render(False)
        return "break"

    def _previous(self) -> None:
        self._stop_playback()
        if self.mode_var.get() == "静态":
            return
        self.current_frame = max(0, self.current_frame - 1)
        self._render(False)

    def _next(self) -> None:
        self._stop_playback()
        if self.mode_var.get() == "静态":
            return
        frames = self._frames()
        self.current_frame = self.current_frame + 1 if self.current_frame + 1 < len(frames) else 0
        self._render(False)

    def _toggle_play(self) -> None:
        if self.playing:
            self._stop_playback(); self._render(False)
            return
        try:
            interval = int(self.interval_var.get())
            if interval < 50:
                raise ValueError("间隔必须至少为 50ms")
            self._frames()
        except (ValueError, TypeError) as exc:
            self.status_var.set(f"无法播放：{exc}")
            return
        self.playing = True
        self.play_button.configure(text="暂停")
        self.after_id = self.root.after(interval, self._tick)

    def _tick(self) -> None:
        if not self.playing:
            return
        frames = self._frames()
        if self.current_frame + 1 < len(frames):
            self.current_frame += 1
        elif self.loop_var.get():
            self.current_frame = 0
        else:
            self._stop_playback(); self._render(False)
            return
        self._render(False)
        try:
            interval = max(50, int(self.interval_var.get()))
        except ValueError:
            interval = 600
            self.interval_var.set("600")
        self.after_id = self.root.after(interval, self._tick)

    def _stop_playback(self) -> None:
        self.playing = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.play_button.configure(text="播放")

    def _close(self) -> None:
        self._stop_playback()
        try:
            save_settings(self.checkpoint_var.get().strip(), self.dataset_var.get().strip())
        except OSError:
            pass
        self.root.destroy()