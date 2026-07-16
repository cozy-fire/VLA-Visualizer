# VLA-Visualizer

> 本仓库基于 [luo3300612/Visualizer](https://github.com/luo3300612/Visualizer) 修改而来，保留了上游 `visualizer.get_local` 字节码插桩能力，并在此基础上增加了面向 VLA 模型的 attention 采集、组织和 Web 可视化能力。

VLA-Visualizer 是一个用于 Vision-Language-Action 模型的本地 attention 可视化工具。项目将模型相关的 attention 采集逻辑与通用 Web 可视化逻辑拆开，便于后续接入不同 VLA 模型。

当前版本**只适配了 SmolVLA**。后续支持其他模型时，可以新增对应的 policy adapter，并复用现有的 Web Viewer。

![SmolVLA attention viewer demo](assets/smolvla_attention_viewer.gif)

## 功能概览

- 选择 SmolVLA checkpoint 和 LeRobot 数据集路径。
- 预览并选择要分析的数据帧。
- 采集该数据帧的 Phase 2 denoising attention cache。
- 按 `step`、`layer`、`head_idx` 筛选和浏览 attention map。
- 支持静态分页浏览和 `steps` / `layers` / `heads` 三种动态播放模式。
- 支持矩阵热力图和原图叠加两种展示方式。
- 支持 `observation.images.*` 多视角图像观测。

## 代码结构

```text
custom_visualizer/
  common/              # 数据集读取、图像收集、attention 筛选、布局推导、序列化
  web/                 # Flask app、API routes、HTML/CSS/JS 前端
  policies/smolvla/    # SmolVLA 模型加载、get_local 插桩、cache 标注
visualizer/            # 上游 Visualizer 的 get_local 字节码插桩工具
```

## Quick Start

### 1. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate vla-visualizer
```

如果你需要特定 CUDA 版本的 PyTorch，请根据本机 CUDA/驱动情况参考 PyTorch 官方安装命令调整 `environment.yml` 中的 `pytorch` 相关依赖。

### 2. 启动 SmolVLA Viewer

```bash
python -m custom_visualizer.policies.smolvla.viewer
```

服务默认监听：

```text
127.0.0.1:7860
```

如果端口被占用，会自动尝试后续可用端口，并打开本机浏览器。

### 3. 在页面中采集并浏览

1. 输入或选择 SmolVLA checkpoint。
2. 输入或选择 LeRobot 数据集路径 / Hub repo。
3. 选择数据帧。
4. 点击 **确认并运行采集**。
5. 使用 `step`、`layer`、`head_idx`、动态模式和“显示原图”开关浏览 attention。

## SmolVLA attention 说明

当前 SmolVLA adapter 从以下函数捕获 attention probability：

```text
SmolVLMWithExpertModel.eager_attention_forward
```

采集到的 Phase 2 attention entry 会被统一标注为：

- `phase`
- `step`
- `layer_idx`
- `type`
- `probs`

常见 denoising attention shape：

```text
(B, H, 50, 113)  # expert_cross_attn
(B, H, 50, 163)  # self_attn
```

其中 query 轴对应 action tokens；key 轴包含 image / language / state 等 prefix tokens，`self_attn` 还包含 action tokens。

## 扩展新模型

接入新模型时，建议新增：

```text
custom_visualizer/policies/<model_name>/
```

对应 adapter 需要负责模型加载、数据预处理、推理、attention 采集和 cache 标注；通用 Web Viewer 只依赖标准化后的 cache 结构，不直接依赖具体模型内部实现。

## 开发检查

```bash
python -m compileall -q custom_visualizer
python -m unittest discover -s tests
node --check custom_visualizer/web/static/app.js
```

## 上游项目

上游 Visualizer 提供了 `get_local`，用于在不显式 return 局部变量的情况下，从函数内部收集中间变量。本项目保留该机制，用于捕获模型 attention 实现内部的 attention probability。

上游项目见：[luo3300612/Visualizer](https://github.com/luo3300612/Visualizer)。

## License

本仓库沿用上游许可证，除非后续另有说明。
