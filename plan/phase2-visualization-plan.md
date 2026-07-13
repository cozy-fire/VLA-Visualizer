# SmolVLA Phase 2 可视化方案（新版，替代旧 5.1-5.5）

## 0. 前置假设

- 使用 `annotate_smolvla_cache`（两阶段版）产出的 `cache`，仅取 `phase=2` 条目
- 每个 entry: `{layer_idx, type, phase, step, probs}`
- probs shape:
  - `self_attn`: `(H, 50, 163)` — Q=action(50), K=prefix(113)+action(50)
  - `expert_cross_attn`: `(H, 50, 113)` — Q=action(50), K=prefix(113)
- prefix 内部分布: `image(64) + language([redacted] + state(1) = 113`

## 1. 分区常量

```
IMG       = 64    # image patches 结束位置
LANG_END  = 112   # language tokens 结束位置
PREFIX_END = 113  # prefix 结束位置（state token）
ACTION    = 50    # action tokens
```

## 2. 实施概要

- **删除**旧的可视化 cell（当前 5.1-5.5 全部 markdown + code cell），完全替代为新版
- 新版包含 1 个 helpers cell（分区线注入函数 + 常量）+ 4 个可视化 cell（每个含函数定义 + 示例调用）
- 4 个函数: `vis_step_layers` / `vis_step_layer_heads` / `vis_across_steps` / `vis_animate`

---

## 3. 四个可视化函数（伪代码 + 详细逻辑）

### 3.1 `draw_partitions(ax, entry)` — 分区线 helper

```
draw_partitions(ax, entry):
    # 横轴 K 方向画竖线
    ax.axvline(IMG - 0.5, color='white', lw=1.5, ls='--')         # image|lang 分界
    ax.axvline(LANG_END - 0.5, color='white', lw=1.5, ls='--')    # lang|state 分界
    if entry.type == "self_attn":
        # 多一条 prefix|suffix 分界（prefix 最后一格是 state token）
        # 横轴 0..113 是 prefix(113)，113..163 是 suffix(50)
        ax.axvline(PREFIX_END - 0.5, color='red', lw=2, ls='-')   # prefix|action 分界

    # 纵轴 Q 方向：始终为 action only，无横线
    # 可选: 在 y 轴标注 "Action tokens (50)"
```

### 3.2 `vis_step_layers` — 同一 step，所有 layer，同一 head

```
vis_step_layers(cache, step, attn_type, head_idx=0):
    entries = cache 中 phase=2, step=s, type=t 的条目   # 对 self_attn 8条，对 expert 8条
    if 空: print("无数据"); return

    cols = 4, rows = ceil(len / 4)
    创建 subplots(rows, cols, figsize=(4*cols, 3.5*rows))

    for (i, entry) in enumerate(entries):
        ax = axes[i]
        probs_2d = entry.probs[head_idx]       # (Lq, Lk)
        ax.imshow(probs_2d, cmap='viridis', aspect='auto')
        draw_partitions(ax, entry)
        ax.set_title(f"L{entry.layer_idx}")
        # 不画 ticks（格子多时避免拥挤）
        ax.set_xticks([]); ax.set_yticks([])

    对剩余空位 ax.axis('off')
    fig.suptitle(f"{attn_type} | Step {step} | Head {head_idx}")
    plt.tight_layout(); plt.show()
```

### 3.3 `vis_step_layer_heads` — 同一 step，单个 layer，全部 head

```
vis_step_layer_heads(cache, step, layer_idx):
    entry = cache 中 phase=2, step=s, layer=l 的第一个（取 type 不影响 probs shape）
    if 无: print("无数据"); return

    probs = entry.probs              # (H, Lq, Lk)
    H = probs.shape[0]
    cols = min(4, H)
    rows = ceil((H + 1) / cols)      # +1 为 mean

    创建 subplots(rows, cols, figsize=(4*cols, 3.5*rows))

    for i in range(H):
        ax = axes[i]
        ax.imshow(probs[i], cmap='viridis', aspect='auto')
        draw_partitions(ax, entry)
        ax.set_title(f"H{i}")
        ax.set_xticks([]); ax.set_yticks([])

    # 最后一格: 所有 head 的 mean
    mean_ax = axes[H]
    mean_probs = probs.mean(axis=0)   # (Lq, Lk)
    mean_ax.imshow(mean_probs, cmap='viridis', aspect='auto')
    draw_partitions(mean_ax, entry)
    mean_ax.set_title("Mean")
    mean_ax.set_xticks([]); mean_ax.set_yticks([])

    剩余空位 ax.axis('off')
    fig.suptitle(f"Layer {layer_idx} | Step {step} | {entry.type}")
    plt.tight_layout(); plt.show()
```

### 3.4 `vis_across_steps` — 不同 step，单个 layer，单个 head

```
vis_across_steps(cache, layer_idx, attn_type, head_idx=0):
    entries = cache 中 phase=2, layer=l, type=t 的条目，按 step 升序排列
    # 应有 10 条 (step 0..9)
    if 空: print("无数据"); return

    cols = 5, rows = 2
    创建 subplots(rows, cols, figsize=(5*cols, 3.5*rows))

    for (i, entry) in enumerate(entries):
        ax = axes[i]
        probs_2d = entry.probs[head_idx]
        ax.imshow(probs_2d, cmap='viridis', aspect='auto')
        draw_partitions(ax, entry)
        ax.set_title(f"Step {entry.step}")
        ax.set_xticks([]); ax.set_yticks([])

    剩余空位 ax.axis('off')
    fig.suptitle(f"L{layer_idx} {attn_type} Head {head_idx} — denoising across 10 steps")
    plt.tight_layout(); plt.show()
```

### 3.5 `vis_animate` — 动态动画：step×layer 双维度

**设计**: 使用 `matplotlib.animation.FuncAnimation`，在固定 figure 上逐帧更新 imshow 数据。提供两种模式:

- **mode="steps"**: 固定 layer，动画遍历 step 0→9，展示去噪过程演变
- **mode="layers"**: 固定 step，动画遍历 layer 0→15，展示深度变化

```
vis_animate(cache, attn_type, head_idx=0, mode="steps", fixed_layer=15):
    if mode == "steps":
        entries = cache 中 phase=2, type=t, layer=fixed_layer 的条目，按 step 排序
        title_prefix = f"L{fixed_layer} {attn_type} H{head_idx}"
    elif mode == "layers":
        step0_entries = cache 中 phase=2, type=t, step=0 的条目，按 layer 排序
        entries = step0_entries  # 展示 step=0 时各层情况
        title_prefix = f"{attn_type} Step0 H{head_idx}"

    创建单轴 fig, ax

    def update(frame):
        ax.clear()
        entry = entries[frame]
        probs_2d = entry.probs[head_idx]
        ax.imshow(probs_2d, cmap='viridis', aspect='auto')
        draw_partitions(ax, entry)
        if mode == "steps":
            ax.set_title(f"{title_prefix} | Step {entry.step}")
        elif mode == "layers":
            ax.set_title(f"{title_prefix} | Layer {entry.layer_idx}")
        ax.set_xticks([]); ax.set_yticks([])

    anim = FuncAnimation(fig, update, frames=len(entries), interval=800, repeat=True)

    # 在 notebook 中内嵌播放
    from IPython.display import HTML
    HTML(anim.to_jshtml())
```

**备选 mode="grid"**: 对于不需要动画的场合，提供一个静态的 step×layer 网格（10×N），可以通过参数切换:
```
vis_animate(cache, attn_type, head_idx=0, mode="grid"):
    entries = cache 中 phase=2, type=t 的条目
    按 step, layer 排序
    布局 rows=10 (step), cols=N (存数据的 layer 数)
    每格: imshow + 极细分区线 + step/layer 标注
    # 本质是 vis_across_steps 的扩展版
```

---

## 4. notebook cell 布局

删除当前 cell 24-33（markdown 5.1 ~ code 5.5），替换为:

| 新 cell | 类型 | 内容 |
|:---:|------|------|
| 1 | markdown | `## Step 5: Phase 2 可视化` |
| 2 | code | 分区常量 + `draw_partitions` helper |
| 3 | markdown | `### 5.1 同Step所有Layer` |
| 4 | code | `vis_step_layers` 定义 + 示例调用 |
| 5 | markdown | `### 5.2 同Step单Layer全部Head` |
| 6 | code | `vis_step_layer_heads` 定义 + 示例调用 |
| 7 | markdown | `### 5.3 跨Step单Layer单Head` |
| 8 | code | `vis_across_steps` 定义 + 示例调用 |
| 9 | markdown | `### 5.4 动态动画` |
| 10 | code | `vis_animate` 定义 + 示例调用（两个 mode 各一次） |

---

## 5. 验证标准

- `vis_step_layers(cache, step=0, attn_type="expert_cross_attn", head_idx=0)`: 输出 2×4 网格（8层有数据），每格横轴有 IMG 和 LANG_END 两条白线
- `vis_step_layers(cache, step=0, attn_type="self_attn", head_idx=0)`: 输出 2×4 网格，每格多一条红色 prefix|action 分界线
- `vis_step_layer_heads(cache, step=0, layer_idx=1)`: 输出 H+1 格，含 Mean，分区线正确
- `vis_across_steps(cache, layer_idx=1, attn_type="expert_cross_attn")`: 2×5 网格，Step0→9
- `vis_animate(cache, "expert_cross_attn", mode="steps", fixed_layer=15)`: 生成可播放动画，逐 step 显示 attention 变化
- `vis_animate(cache, "expert_cross_attn", mode="layers")`: 动画遍历 8 层
