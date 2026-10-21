---
name: pfdelta-comparison-plot
description: 从 PFDelta 已完成的评估目录、逐种子 CSV 或已有绘图点 CSV 绘制可迁移的端点训练与中间 rho 增广对比图，适用于 case30、case118、case500 及其他系统；不执行训练、推理或 scaling law 拟合。
---

# PFDelta 训练样本对比图

使用本技能目录中的 `plot_pf_comparison.py`。先读 [README.md](README.md) 中与输入类型相关的部分，并运行 `python plot_pf_comparison.py --help` 确认参数。脚本路径相对于技能目录解析，不依赖当前工作目录或原 PFDelta 仓库。

## 输入与执行

从用户指定的评估结果读取数据，保持源文件只读，将结果写入独立输出目录。支持以下入口：

- `--endpoint-root`、`--mixed-root`：原生运行目录，可重复指定。
- `--csv`：逐种子结果，按测试端点、指标、方法和训练样本组成求算术均值。
- `--points-csv`：已有汇总绘图点，保留原均值及种子名单；不能重新选择种子计算均值。

```bash
python /path/to/pfdelta-comparison-plot/plot_pf_comparison.py \
  --case case500 \
  --endpoint-root /path/to/endpoint-results \
  --mixed-root /path/to/mixed-results \
  --metrics MSE,PBL,rPBL \
  --seeds 11,22,33,44,55 \
  --output-dir /path/to/figures/case500
```

`--case` 是系统名称，不决定样本量、种子或测试数据。没有混合实验时省略 `--mixed-root`。CSV 格式、原生目录结构及历史表兼容方式见 README。

## 数据与图形约定

- 横轴为总训练图记录数：`endpoint_samples + midrho_samples`，不是父工况数；不同端点基数或 `method` 的混合分支分别连接。
- 确认合并结果使用同一固定测试集和相同指标口径；`--test-id` 只能补充缺少的标识，不能覆盖冲突。MSE、PBL、rPBL、MAPE 直接读取已有指标，不相互推导；`MAPE (%)` 不再乘以 100。
- 新数据默认使用全部有效种子。`--seeds` 同时筛选并声明期望集合；缺组默认报错，只有用户意图适用时才用 `--incomplete skip` 或 `available`。不继承旧图事后筛选过的名单，不为平滑曲线删种子。
- 需要复现旧图均值时用原 `--points-csv` 或明确筛选过的原始表，并保留其事后筛选背景。不要将均值伪装为单个种子。
- 默认左右面板为 `rho0`、`rho1`，蓝色为仅端点训练、粉色为混合训练；两轴 log2，不画误差棒。`--uncertainty sd` 仅表示种子间样本标准差，不是置信区间。
- `--sizes` 会筛掉实际点；若需要混合分支共享基线，显式保留相应基线样本数。
- 中文用 `--language zh`，缺少字体时提供 `--font-file /path/to/font.ttf`；不假设目标机器安装了原机器字体。

检查导出的 PNG/PDF/SVG 与 `plot_points.csv`、`per_seed.csv`（逐种子模式）、`metadata.json`。核对样本组成、实际种子、测试集、跳过项和图例后交付。只画已有数据的对比趋势；需要拟合 scaling law 时使用相应技能。
