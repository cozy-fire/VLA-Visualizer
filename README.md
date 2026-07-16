# VLA-Visualizer

> 本仓库基于 [luo3300612/Visualizer](https://github.com/luo3300612/Visualizer) 修改而来。原项目中的 `visualizer.get_local` 字节码插桩能力仍被保留，本仓库在此基础上增加了面向 VLA 模型的 attention 采集、组织和 Web 可视化能力。

VLA-Visualizer 是一个用于 Vision-Language-Action 模型的本地 attention 可视化工具。项目目标是把不同 VLA 模型的 attention 采集逻辑和通用可视化逻辑解耦：模型相关部分放在各自的 `policies/` adapter 中，数据集读取、attention 筛选、布局、播放和前端展示放在公共模块中。

当前版本**只适配了 SmolVLA**。后续如果需要支持其他模型，可以新增对应 policy adapter，复用现有 Web 展示层和通用 attention 浏览逻辑。

![SmolVLA attention viewer demo](assets/smolvla_attention_viewer.gif)

## 当前已支持能力

当前 SmolVLA adapter 支持对 LeRobot 数据帧执行一次推理，采集 Phase 2 denoising 过程中的 attention cache，并在 Flask Web 页面中浏览。

主要功能包括：

- 选择 SmolVLA checkpoint 和 LeRobot 数据集路径，并持久化历史记录。
- 通过数据帧选择窗口预览数据集图像，再选择要采集的 frame。
- 对选中的数据帧执行一次模型推理，采集内存中的 Phase 2 attention cache。
- 按 `step`、`layer`、`head_idx` 浏览 attention map。
- 支持静态分页浏览。
- 支持三种动态播放模式：
  - `steps`
  - `layers`
  - `heads`
- 支持两种渲染模式：
  - 矩阵模式：展示完整 attention matrix，并标记 token 分界线；
  - 原图叠加模式：提取 image-token attention，并映射回当前数据帧的原图。
- 支持多视角图像观测，图像字段按 `observation.images.*` 收集。

## 代码结构

项目把通用逻辑和模型特化逻辑拆开：

```text
custom_visualizer/
  common/              # 数据集读取、图像收集、attention 筛选、布局推导、序列化
  web/                 # Flask app、API routes、HTML/CSS/JS 前端
  policies/smolvla/    # SmolVLA 模型加载、get_local 插桩、cache 标注
visualizer/            # 上游 Visualizer 的 get_local 字节码插桩工具
```

这种结构的目的不是只服务 SmolVLA，而是让后续模型接入时只需要新增类似：

```text
custom_visualizer/policies/<model_name>/
```

并实现对应 adapter，即可复用现有 Web Viewer。

## 快速启动

在仓库根目录执行：

```powershell
conda activate lerobot
python -m custom_visualizer.policies.smolvla.viewer
```

默认情况下，Flask 服务会监听：

```text
127.0.0.1:7860
```

如果端口被占用，会自动尝试后续可用端口，并打开本机浏览器。

本项目中的 Python 命令默认都应在 `lerobot` Conda 环境中执行。

## 默认路径

当前 SmolVLA adapter 的默认路径为：

```text
Checkpoint:
D:\CodeProject\ModelTrainRepo\Checkpoint\pick_pen_give_human\smolvla\checkpoints\020000\pretrained_model

Dataset:
D:\Huggingface Cache\lerobot\CozyFire\pick_pen_give_human
```

Web 页面会记录 checkpoint 和 dataset 的历史路径。每次成功点击运行采集后，最后一次使用的路径会作为下次启动时的默认值。

## 使用流程

1. 启动 Web Viewer。
2. 在左侧控制面板选择或输入：
   - SmolVLA checkpoint；
   - LeRobot 数据集路径或 Hub repo；
   - 要分析的数据帧。
3. 点击 **确认并运行采集**。
4. 采集完成后，在右侧浏览 attention cache：
   - 不设置筛选条件时展示当前可见范围内的全部 attention map；
   - 设置 `step`、`layer` 或 `head_idx` 后，在全量集合上收窄显示；
   - 切换静态模式或动态播放模式观察不同维度的变化。
5. 根据需要开启“显示原图”模式，将 image-token attention 映射回原始图像。

## Attention cache 语义

当前 SmolVLA adapter 从以下函数中捕获 attention probability：

```text
SmolVLMWithExpertModel.eager_attention_forward
```

采集到的 Phase 2 attention entry 会被标注为：

- `phase`
- `step`
- `layer_idx`
- `type`
- `probs`

当前观察到的 denoising attention shape 通常为：

```text
(B, H, 50, 113)  # expert_cross_attn
(B, H, 50, 163)  # self_attn
```

其中：

- query 轴对应 action tokens；
- key 轴包含 image / language / state 等 prefix tokens；
- `self_attn` 的 key 轴还包含 action tokens。

## 原图叠加模式

开启“显示原图”后，每个 attention map 不再以二维矩阵展示，而是把 image-token 对应的 attention 投影回当前数据帧的原图。

当前流程为：

1. 从指定 head 的 attention 中截取 image-token 列；
2. 沿 action-token 维度求平均，得到每个 image patch 的 attention 分数；
3. 如果存在多视角图像，则按 `observation.images.*` 的稳定排序拆分 image tokens；
4. 根据推导出的 patch grid 将分数 reshape 为空间热力图；
5. 将 patch heatmap 映射到原始图像区域；
6. 在原 attention card 中展示叠加结果。

该模式只影响前端展示，不会修改原始数据集图像。

## 扩展新模型

如果后续需要支持其他 VLA 模型，建议新增一个 policy adapter，而不是修改 Web 层。

一个 adapter 至少需要负责：

- 加载对应模型 checkpoint；
- 构造数据预处理流程；
- 执行指定数据帧的推理；
- 捕获或返回 attention cache；
- 将原始 attention 标注成通用字段：`phase`、`step`、`layer_idx`、`type`、`probs`；
- 返回图像观测、token layout 和 cache summary。

通用 Web Viewer 只依赖 adapter 输出的通用结构，不应直接依赖具体模型内部实现。

## 开发检查

推荐执行：

```powershell
conda activate lerobot
python -m compileall -q custom_visualizer
python -m unittest discover -s tests
node --check custom_visualizer/web/static/app.js
```

如果通过非交互 shell 执行，可以使用：

```powershell
conda run -n lerobot python -m unittest discover -s tests
```

## 关于上游 Visualizer

上游项目提供了 `get_local`，用于在不显式 return 局部变量的情况下，从函数内部收集中间变量。本项目保留该机制，因为当前 SmolVLA 的 attention probability 需要从模型 attention 实现内部捕获。

上游项目和通用示例见：[luo3300612/Visualizer](https://github.com/luo3300612/Visualizer)。

## License

本仓库沿用上游许可证，除非后续另有说明。
