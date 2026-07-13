# SmolVLA 注意力可视化方案（`@get_local` monkey-patch 版）

## 1. 背景与目标

**目标**: 利用 Visualizer 现有的 `get_local` 机制，不改 SmolVLA 源码一行，通过 monkey-patch 捕获前向推理中的所有 attention map。

**参考代码**: `D:\CodeProject\lerobot\src\lerobot\policies\smolvla`

**核心工具**: `D:\CodeProject\VLM-Visualizer\visualizer\visualizer.py` — `get_local` 类

---

## 2. SmolVLA 架构速览

```
输入: images + language_tokens + state + (noisy_actions + timestep)
  │
  ├─ Vision Encoder: SigLIP → connector
  ├─ Language Embedding: SmolVLM2 input embeddings
  ├─ State / Action+Time: 各自 Linear 投影
  │
  └─ SmolVLMWithExpertModel (核心 Transformer, smolvlm_with_expert.py)
       ├─ VLM Backbone: 16层 Gemma
       ├─ Action Expert: 较窄的 Gemma 副本 (interleaved)
       │
       └─ 两种 Attention 模式交替 (self_attn_every_n_layers=2):
            ├─ self_attn: VLM prefix + Expert suffix 拼接 → 联合 self-attn
            └─ cross_attn: VLM prefix self-attn → Expert cross-attends to VLM KV
```

---

## 3. 注意力计算点 — 唯一的拦截目标

所有 Q@K 注意力都汇聚到 **唯一一个函数**:

| 属性 | 值 |
|------|-----|
| **文件** | `smolvlm_with_expert.py` |
| **类** | `SmolVLMWithExpertModel` |
| **方法** | `eager_attention_forward` (第505–550行) |
| **Q@K 变量** | `att_weights` (第535行) — 原始分数, shape `(B, H, Lq, Lk)` |
| **Softmax 变量** | `probs` (第541行) — 注意力概率, shape `(B, H, Lq, Lk)` |
| **返回值** | `att_output` (第550行) |

### 3.1 调用链路

```
SmolVLMWithExpertModel.forward()           # layer_idx = 0..15 主循环
  ├─ forward_attn_layer()                  # self-attn 模式 (偶数层)
  │     └─ get_attention_interface()       # 返回 self.eager_attention_forward
  │           └─ eager_attention_forward() # ← 🎯 返回 att_output
  │
  └─ forward_cross_attn_layer()            # cross-attn 模式 (奇数层)
        ├─ get_attention_interface() → eager_attention_forward() # ← 🎯 prefix self-attn
        └─ get_attention_interface() → eager_attention_forward() # ← 🎯 expert cross-attn
```

### 3.2 一次 forward 的调用次数与顺序

默认配置 `num_vlm_layers=16, self_attn_every_n_layers=2`:

```
layer_idx  mode         调用
─────────────────────────────
0         self_attn     → cache[0]    (B, H, Lp+La, Lp+La)
1         cross_attn    → cache[1]    (B, H, Lp, Lp)      prefix self
                        → cache[2]    (B, H, La, Lp)      expert cross
2         self_attn     → cache[3]    (B, H, Lp+La, Lp+La)
3         cross_attn    → cache[4]    ...
                        → cache[5]
...
15        cross_attn    → cache[22]
                        → cache[23]
```

总计 **24 次**调用 = 8层 self_attn × 1 + 8层 cross_attn × 2

### 3.3 各注意力类型的语义

```
prefix = [image_patches × Nimg | lang_tokens × Ntxt | state_token × 1]
suffix = [action_tokens × Na]
```

| 类型 | Query 来自 | Key 来自 | 要回答的问题 |
|------|-----------|---------|-------------|
| `self_attn` | prefix + suffix | prefix + suffix | 图文与 action 如何交互 |
| `prefix_self_attn` | prefix | prefix | 图片 patch 间、图-文之间的注意力 |
| `expert_cross_attn` | suffix (action) | prefix KV | 🔥 **Action 在"看"图片的哪里？** |

---

## 4. 方案：`@get_local` monkey-patch（0 行源码改动）

### 4.1 为什么可行

`get_local.__call__` 可以作用于**任意函数**，不限于 `@` 语法糖：

```python
# @ 语法糖形式:
@get_local('probs')
def eager_attention_forward(self, ...):
    ...

# 等价于:
eager_attention_forward = get_local('probs')(eager_attention_forward)
```

`eager_attention_forward` 满足所有条件：
- `probs` 是局部变量 ✅
- `return att_output` 时 `probs` 仍存活 ✅
- 所有层共用同一方法 → cache 自动收集所有调用 ✅

### 4.2 完整使用流程

> **可执行 notebook**: [smolvla_attention_demo.ipynb](smolvla_attention_demo.ipynb) — 路径 A 和 B 的完整代码，
> 包含数据加载、monkey-patch、标注和所有可视化面板。

> **Cell 分块风格**: 参考 `demo.ipynb` 的组织方式 — 每个逻辑步骤用 `## StepN` 的 markdown cell 标识，
> 紧接一个代码 cell；可视化辅助函数集中在 import 之后的一个大 cell 中定义，可视化调用在后续独立的 cell 中执行。
> 这样便于按步骤讲解和执行。
> **两种路径**: (A) 从 LeRobot 数据集加载真实数据; (B) 构造假数据快速验证 monkey-patch 是否生效。

#### 路径 A: 从 LeRobot 数据集加载（推荐，用于实际分析）

```python
import torch
from visualizer import get_local

# ──── 步骤 1: 激活 ────
get_local.activate()

# ──── 步骤 2: 加载模型和预处理器 ────
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy.to(device)
model = policy.vla.vlm_with_expert  # SmolVLMWithExpertModel 实例

# 预处理器: 负责 tokenization、归一化、设备迁移
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path="lerobot/smolvla_base",
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# ──── 步骤 3: monkey-patch（必须在 forward 之前）───
model.eager_attention_forward = get_local('probs')(model.eager_attention_forward)

# ──── 步骤 4: 加载数据集 ────
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps

# 替换为实际数据集 repo，如 "danaaubakirova/svla_so100_task1_v3"
dataset_repo = "your_dataset_repo"

ds_meta = LeRobotDatasetMetadata(dataset_repo)
delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta)
dataset = LeRobotDataset(dataset_repo, delta_timestamps=delta_timestamps)

# ──── 步骤 5: 取一条数据 → 预处理 → forward ────
get_local.clear()  # 清空上次缓存

item = dataset[0]                              # dict: observation.state, observation.images.*, action, task
batch = preprocessor(item)                     # 添加 language tokens、归一化、移到 device

with torch.no_grad():
    action_chunk = policy.predict_action_chunk(batch)  # (1, 50, action_dim)
    # 内部调用 VLAFlowMatching.sample_actions → SmolVLMWithExpertModel.forward
    # 所有 eager_attention_forward 的 probs 被自动拦截

# ──── 步骤 6: 获取并标注结果 ────
raw_cache = get_local.cache
probs_list = raw_cache['SmolVLMWithExpertModel.eager_attention_forward']
# 每个元素 shape: (B, num_heads, Lq, Lk)

# 自动标注层号和类型
num_vlm_layers = model.num_vlm_layers
self_attn_every_n = model.self_attn_every_n_layers
cache = annotate_smolvla_cache(probs_list, num_vlm_layers, self_attn_every_n)

# 看各类型数量
from collections import Counter
print(Counter(e["type"] for e in cache))
# -> {'self_attn': 8, 'prefix_self_attn': 8, 'expert_cross_attn': 8}
```

#### 路径 B: 构造假数据快速验证

```python
from visualizer import get_local
import torch

get_local.activate()

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy.to(device)
model = policy.vla.vlm_with_expert

# monkey-patch
model.eager_attention_forward = get_local('probs')(model.eager_attention_forward)

# 构造预处理前的 batch（模拟 dataset[0] 的输出格式）
# 图片维度: (C, H, W), state 维度: (state_dim,)
batch = {
    "observation.state": torch.randn(policy.config.max_state_dim, dtype=torch.float32),
    "observation.images.cam1": torch.rand(3, 512, 512, dtype=torch.float32),
    "task": "pick the red block\n",
}
preprocessor, _ = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path="lerobot/smolvla_base",
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)
batch = preprocessor(batch)

# forward
get_local.clear()
with torch.no_grad():
    action_chunk = policy.predict_action_chunk(batch)

# 取结果
probs_list = get_local.cache['SmolVLMWithExpertModel.eager_attention_forward']
print(f"捕获到 {len(probs_list)} 个 attention maps")
# -> 24（或推理时的更多）

cache = annotate_smolvla_cache(probs_list, model.num_vlm_layers, model.self_attn_every_n_layers)
```

#### Observation Dict 格式说明

从 `dataset[i]` 直接取出的 raw item（预处理前）:

```python
item = {
    "observation.state":      torch.Tensor,  # (state_dim,) float32，物理单位
    "observation.images.cam": torch.Tensor,  # (C, H, W) float32, [0, 1]
    "action":                 torch.Tensor,  # (chunk_size, action_dim) float32
    "task":                   str,           # e.g. "pick the red block\n"
}
```

经过 `preprocessor` 后（送入 `predict_action_chunk` 的格式）:

```python
batch = {
    "observation.state":                      torch.Tensor,  # (1, state_dim), 归一化, 在 device 上
    "observation.images.cam":                 torch.Tensor,  # (1, 3, H, W), [0,1], 在 device 上
    "observation.language.tokens":            torch.Tensor,  # (1, 48), int
    "observation.language.attention_mask":    torch.Tensor,  # (1, 48), int
    "action":                                 torch.Tensor,  # (1, 50, action_dim) 可选
}
```

#### 推理模式的重要差异

`policy.predict_action_chunk()` 内部调用 `sample_actions()`，分两阶段:

| 阶段 | 目的 | 触发的 attention 调用 |
|------|------|----------------------|
| Phase 1: prefix encoding | 将图片+文本+状态编码进 KV cache | 24 次（所有层各一次 self_attn 或 prefix+expert） |
| Phase 2: 迭代去噪 (10步) | 从噪声中逐步恢复 action | 每步 8 次 expert_cross_attn (仅奇数层) |

**完整推理的 cache 总数**: `24 + 10 × 8 = 104` 个条目。  
标注函数按顺序推断，Phase 2 的 extra entries 可能破坏标注规律。  
**建议首次试验用路径 B 的假数据快速验证 monkey-patch 是否生效，看到 24 条即成功。**

### 4.3 辅助函数：自动标注层号和类型

```python
def annotate_smolvla_cache(cache_list, num_vlm_layers=16, self_attn_every_n=2):
    """
    将 flat list 的 attention maps 标注 layer_idx 和 type。

    Args:
        cache_list: get_local.cache['SmolVLMWithExpertModel.eager_attention_forward']
        num_vlm_layers: VLM 层数，默认 16
        self_attn_every_n: 每 N 层使用一次 self_attn，默认 2

    Returns:
        list[dict]: [{"layer_idx": int, "type": str, "probs": ndarray}, ...]
    """
    annotated = []
    remaining = list(cache_list)
    for layer_idx in range(num_vlm_layers):
        if layer_idx % self_attn_every_n == 0:
            annotated.append({
                "layer_idx": layer_idx,
                "type": "self_attn",
                "probs": remaining.pop(0)
            })
        else:
            annotated.append({
                "layer_idx": layer_idx,
                "type": "prefix_self_attn",
                "probs": remaining.pop(0)
            })
            annotated.append({
                "layer_idx": layer_idx,
                "type": "expert_cross_attn",
                "probs": remaining.pop(0)
            })
    return annotated

# 使用:
cache = annotate_smolvla_cache(
    get_local.cache['SmolVLMWithExpertModel.eager_attention_forward']
)

# 按类型筛选:
expert_attns = [e for e in cache if e["type"] == "expert_cross_attn"]
# -> 8 个条目, layer_idx = 1,3,5,7,9,11,13,15
```

### 4.4 推理两阶段的 cache 条目处理

`predict_action_chunk()` 内部调用 `sample_actions()`，分两个阶段:

```python
# Phase 1: prefix encoding（填 KV cache）
# 每层 1 次 self_attn 或 2 次 cross_attn → 24 次 eager_attention_forward

# Phase 2: 迭代去噪（默认 num_steps=10, Euler 方法）
# 每步仅奇数层执行 expert_cross_attn → 每步 8 次，共 10×8=80 次
```

完整推理 cache 总数 = `24 + num_steps × num_cross_attn_layers`。

`annotate_smolvla_cache` 只处理前 24 个 Phase 1 条目。Phase 2 的条目按 8 个一组周期性标注:

```python
phase2_start = num_vlm_layers + num_vlm_layers // self_attn_every_n  # = 24
step_count = num_vlm_layers // self_attn_every_n                     # = 8
for i, entry in enumerate(cache[phase2_start:]):
    entry["step"] = i // step_count
    entry["layer_idx"] = 1 + (i % step_count) * self_attn_every_n    # 1,3,5,...
    entry["type"] = "expert_cross_attn"
```

---

## 5. 可视化

### 5.1 单层单 Head 热力图

```python
def visualize_probs(probs, title="", head_idx=0, batch_idx=0):
    """可视化单个 attention probs 矩阵"""
    att_map = probs[batch_idx, head_idx]  # (Lq, Lk)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(att_map, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Key position →")
    ax.set_ylabel("Query position →")
    ax.set_title(title)
    plt.show()

# 看第3层 expert cross-attention 第0个head:
entry = [e for e in cache if e["layer_idx"] == 3 and e["type"] == "expert_cross_attn"][0]
visualize_probs(entry["probs"], title="Layer 3 Expert Cross-Attn, Head 0")
```

### 5.2 全部 Head 面板

```python
def visualize_all_heads(probs, title_prefix="", batch_idx=0):
    """显示某一层 attention 的所有 head"""
    num_heads = probs.shape[1]
    cols = min(4, num_heads)
    rows = (num_heads - 1) // cols + 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    if rows == 1:
        axes = [axes]
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        if i < num_heads:
            ax.imshow(probs[batch_idx, i], cmap='viridis', aspect='auto')
            ax.set_title(f"{title_prefix} H{i}")
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

entry = [e for e in cache if e["layer_idx"] == 1 and e["type"] == "expert_cross_attn"][0]
visualize_all_heads(entry["probs"], title_prefix="Layer 1 Expert Cross-Attn ")
```

### 5.3 逐层对比（最有分析价值）

```python
def visualize_across_layers(cache, attn_type="expert_cross_attn", head_idx=0, batch_idx=0):
    """横向对比所有层的同类型 attention"""
    entries = [e for e in cache if e["type"] == attn_type]
    n = len(entries)
    cols = 4
    rows = (n - 1) // cols + 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    for i, entry in enumerate(entries):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        im = ax.imshow(entry["probs"][batch_idx, head_idx], cmap='viridis', aspect='auto')
        ax.set_title(f"Layer {entry['layer_idx']}")
        plt.colorbar(im, ax=ax)
    for j in range(n, rows * cols):
        ax = axes[j // cols][j % cols] if rows > 1 else axes[j % cols]
        ax.axis('off')
    plt.suptitle(f"{attn_type} — Head {head_idx} across layers")
    plt.tight_layout()
    plt.show()

visualize_across_layers(cache, "expert_cross_attn", head_idx=0)
```

### 5.4 Action → Image 注意力（核心分析）

```python
def visualize_action_to_image(entry, image_patch_count, head_idx=0, batch_idx=0):
    """
    从 expert_cross_attn 中提取 action token 对 image patches 的注意力。

    SmolVLA prefix 结构: [image_patches × Nimg | lang_tokens × Ntxt | state × 1]
    前 image_patch_count 个 key position 就是图片区域。
    """
    probs = entry["probs"][batch_idx, head_idx]     # (La, Lp)
    # 只取对 image patches 的注意力
    att_to_image = probs[:, :image_patch_count]       # (La, Nimg)
    avg_att = att_to_image.mean(axis=0)                # 平均所有 action token

    # reshape 成 2D（假设 square patches, 如 256 = 16×16）
    side = int(image_patch_count ** 0.5)
    if side * side == image_patch_count:
        heatmap = avg_att.reshape(side, side)
    else:
        heatmap = avg_att  # fallback: 直接显示 1D

    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Attention weight')
    plt.title(f"Layer {entry['layer_idx']}: Action → Image Attention")
    plt.show()

# 使用（需知道 image_patch_count，可从 policy.config 获取）:
entry = [e for e in cache if e["layer_idx"] == 15 and e["type"] == "expert_cross_attn"][0]
visualize_action_to_image(entry, image_patch_count=256)  # 具体数值取决于图像分辨率
```

---

## 6. 注意事项

### 6.1 显存

| 类型 | num_heads | Lq × Lk (典型) | 单个大小 (float32) |
|------|:---------:|----------------|:---:|
| self_attn | 8 | 520 × 520 | ~8.7 MB |
| prefix_self_attn | 8 | 500 × 500 | ~8 MB |
| expert_cross_attn | 6 | 20 × 500 | ~0.24 MB |

全部 24 个 ≈ **120 MB**。`get_local.wrapper` 会自动 `.cpu()`。

### 6.2 训练兼容性

`get_local.wrapper` 会对 `probs` 调用 `.detach().cpu().numpy()`，这会断开计算图。**可视化仅适用于推理模式**。如果后续需要在训练中记录 attention，需要修改 `visualizer.py` 的 wrapper，去掉 `.detach()`。

---

## 7. 执行方案的实施步骤

以下按顺序执行，每步验证通过后再进入下一步。所有可执行代码均在 [smolvla_attention_demo.ipynb](smolvla_attention_demo.ipynb) 中。

### 步骤 1: 验证 Visualizer 基础功能

用项目自带的 ViT demo（`demo.ipynb` 或一行脚本）确认 `get_local` 链路正常 —— 加载 `vit_small_patch16_224`，forward 后 `cache['Attention.forward']` 应包含 12 个 attention map。

### 步骤 2: 确认环境依赖

验证 `torch`、`transformers`、`lerobot` 及其子模块（`SmolVLAPolicy`、`make_pre_post_processors`、`LeRobotDataset`）可正常 import，CUDA 可用。

> 如果 lerobot 未安装: `pip install -e ".[smolvla]"`

### 步骤 3: 下载并加载模型

首次执行 `SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")` 会下载约 5GB 权重。加载后确认 `model.num_vlm_layers` 和 `model.self_attn_every_n_layers` 等结构参数。

### 步骤 4: 路径 B — 假数据验证 monkey-patch

打开 notebook，执行 Step 1 ~ Step 4（路径 B 部分）。关键验证点：
- `model.eager_attention_forward = get_local('probs')(...)` 替换成功
- forward 后 `len(probs_list) >= 24`
- `annotate_smolvla_cache` 正确标注出三类注意力

期望看到 `{'self_attn': 8, 'prefix_self_attn': 8, 'expert_cross_attn': 8}`。如果看到 104 条说明含推理两阶段；0 条则检查 `activate()` 是否在模型加载前执行。

### 步骤 5: 路径 A — 加载真实数据集

将 notebook 中 `DATASET_REPO` 替换为实际数据集（如 `lerobot/aloha_sim_insertion_human`），确认 `dataset[0]` 含 `observation.images.*` 和 `task`，preprocessor 能正常处理。

### 步骤 6: 可视化分析

按顺序执行 notebook Step 5 的各可视化 cell：
1. 单层单 Head → 验证数据可正常 `imshow`
2. 全部 Head 面板 → 观察注意力模式差异
3. 逐层对比 → 🔥 核心面板，观察注意力随深度变化
4. Action → Image → 将 cross-attn 整理成空间热力图
5. 叠加原图 → 直观展示 action 关注图片的哪些区域

### 步骤 7: 清理

notebook Step 6: `get_local.clear()` 清空缓存。如需彻底恢复原函数，重新加载模型即可。

---

## 8. 总结：三步走

```
Step 1: get_local.activate()
Step 2: model.eager_attention_forward = get_local('probs')(model.eager_attention_forward)
Step 3: forward → annotate_smolvla_cache(get_local.cache[...])
```

**改动 lerobot 源码: 0 行。改动 visualizer: 0 行。用户侧代码: ~30 行（含辅助函数）。**

---

## 10. Annotate Cache 两阶段解析

### 10.1 实际调用模式

SmolVLA 推理时的完整 `eager_attention_forward` 调用序列如下：

**Phase 1 — KV Cache 填充** (`fill_kv_cache=True`，`sample_actions` 第818行):
- 调用次数: `num_vlm_layers` 次（16 次）
- 每层走 `forward_attn_layer`，Q/K 均为 prefix tokens
- probs shape: `(B, H, 113, 113)` — 纯 prefix 内部自注意力
- 同时此阶段将 K/V 缓存到 `past_key_values`

**Phase 2 — 迭代去噪** (`fill_kv_cache=False`，`denoise_step` 每个 step 调一次，共 10 步):
- 每步调用 `num_vlm_layers` 次（16 次），10 步共 160 次
- 单步内 layer 0..15 交替执行:
  - 偶数层 (`layer_idx % self_attn_every_n == 0`): `forward_attn_layer` → `self_attn`
    - Q = suffix tokens only → `(B, H, 50, head_dim)`
    - K = cached prefix + suffix → `(B, H, 113+50, head_dim)` = `(B, H, 163, head_dim)`
    - probs: `(B, H, 50, 163)`
  - 奇数层: `forward_cross_attn_layer` → `expert_cross_attn`
    - Q = suffix tokens only → `(B, H, 50, head_dim)`
    - K = cached pure prefix → `(B, H, 113, head_dim)`
    - probs: `(B, H, 50, 113)`

**完整调用总数**: `16 + 10 × 16 = 176`

### 10.2 `annotate_smolvla_cache` 解析逻辑

- 前 `num_vlm_layers` (16) 条: Phase 1, `type="self_attn"`, `step=None`
- 剩余: 按 16 条一组分组为 step 0..9
  - 组内偶数索引: `type="self_attn"`, 奇数索引: `type="expert_cross_attn"`
  - 各组内 `layer_idx` 复位为 0..15

### 10.3 三种 probs shape 的语义

| Phase | Type | Shape | Q 语义 | K 语义 |
|-------|------|-------|--------|--------|
| 1 | self_attn | `(H, 113, 113)` | prefix 全 token | prefix 全 token |
| 2 | self_attn | `(H, 50, 163)` | suffix (action) | cached prefix + suffix |
| 2 | expert_cross_attn | `(H, 50, 113)` | suffix (action) | cached prefix only |

其中 prefix = image(64) + language([redacted] + state(1); suffix = action tokens(50)。

### 10.4 可视化适配要点

- `visualize_probs`: 根据 `phase`+`type` 自动适配 Q/K 分区线；Phase 2 `self_attn` 的 K 轴需在 113 处加额外分割线
- `visualize_across_layers`: 新增 `phase` 和 `step` 筛选参数，避免 Phase 1/2 同 type 混在一起
- `visualize_action_to_image`: image patches 在 K 轴始终位于前 64 个位置，与 phase/type 无关
- `visualize_step_evolution`: 选某一层看 10 步去噪中注意力热力图的变化

---

## 9. 附录：关键源码位置

| 内容 | 文件 | 行号 |
|------|------|:---:|
| `eager_attention_forward` 定义 | `smolvlm_with_expert.py` | 505 |
| Q@K matmul | `smolvlm_with_expert.py` | 535 |
| softmax → `probs` | `smolvlm_with_expert.py` | 541 |
| return att_output | `smolvlm_with_expert.py` | 550 |
| `forward` 主循环 | `smolvlm_with_expert.py` | 404 |
| `get_local.__call__` 实现 | `visualizer/visualizer.py` | 10 |
| ViT demo (参考) | `demo.ipynb` | — |
