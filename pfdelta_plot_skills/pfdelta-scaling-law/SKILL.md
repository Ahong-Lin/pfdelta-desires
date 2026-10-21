---
name: pfdelta-scaling-law
description: 从 PFDelta 评估目录、逐种子 CSV 或已有绘图点拟合误差随训练样本数变化的幂律或带平台项幂律，导出可复核的图、拟合参数和数据；适用于任意 case，不执行模型训练或评估。
---

# PFDelta scaling law

运行本技能目录中的 `plot_pf_scaling.py`，先用 `--help` 查看当前参数。使用已有评估值，保持输入只读，输出到独立目录。技能和脚本可迁移到另一台机器；不要依赖原电脑的绝对路径、字体或数据。

## 输入与基本调用

选择原生评估目录 `--endpoint-root` / `--mixed-root`，逐种子表 `--csv`，或已汇总点表 `--points-csv`。目录参数可重复指定；CSV 参数可接受多个文件。逐种子表以 `test`（`rho0` / `rho1`）、`endpoint_samples`、`midrho_samples`、`seed` 标识结果，提供所选指标列，例如 `canos_pf_mse`、`PBL Mean`、`Relative PBL Mean`、`MAPE (%)`。`total_train_samples` 如存在须等于前两类样本之和；用 `method` 区分混合方案，用 `test_id` 标识固定测试集。

```bash
python /path/to/pfdelta-scaling-law/plot_pf_scaling.py \
  --case case500 \
  --endpoint-root /path/to/endpoint-results \
  --metrics MSE \
  --seeds 11,22,33,44,55 \
  --model power --fit-space log \
  --output-dir /path/to/figures/case500_scaling
```

`--case` 只作为系统标识。原生目录读取 `Evaluation/completion.json`、`Evaluation/test_metrics.json` 和 `Model/run_config.json` 等记录；未完成或 smoke 结果不能作为正式结果。沿用两测试端点的数据结构，不从其他指标估算缺失指标。

`--points-csv` 保留既有均值及种子名单，不能重新计算种子筛选结果。历史 scaling 表若省略样本组成，仅在其明确表示端点基线时按仅端点训练解释；混合点必须有可确认的组成，不按总样本数猜测。旧图的事后种子筛选背景须保留，不把旧名单套到新数据。

## 拟合定义与分组

横轴 `N` 是总训练图记录数，默认 `N_ref = 40000`（`--n-ref` 可修改）。每个样本量使用种子算术均值，拟合时各样本量等权。样本数和指标必须为有限正值。

- `--model power`（默认）：`E = A × (N / N_ref)^(-b)`。`b` 可为负，不能强迫不随数据量下降的结果变成下降曲线。至少需要 3 个不同样本量。
- `--model offset`：`E = C + A × (N / N_ref)^(-b)`，`A > 0`、`C ≥ 0`、`b` 限定在正数范围。至少需要 4 个不同样本量；检查边界和辨识性，特别是 `C = 0` 边界。非线性拟合的指数上限默认是 10，可通过 `--max-exponent` 调整；默认对数空间纯幂律不限制指数。
- `--model both`：分别拟合并输出两个模型，不能仅因 offset 参数更多就认定它更可信。
- `--fit-space log`（默认）最小化对数误差；`linear` 最小化原始误差。两者优化目标不同，不混淆 `R²_log` 与 `R²_linear`。

`--series endpoint` 是默认，只拟合端点训练。`mixed` 只拟合混合分支，`all` 同时处理两类。测试端点、指标、混合方法和固定端点基数分别拟合：不能将不同端点基数或端点基线与混合实验拼成一条曲线。`--include-anchor` 可将相应端点基线加入各混合分支的拟合，此选择须在说明中保留。

## 筛选、样式与判断

- `--fit-min` / `--fit-max` 只限制参与拟合的样本数范围，保留范围外的实测点。`--sizes` 则筛选实际输入点；不要将两者混用。
- `--seeds` 筛选并要求各组拥有指定种子；缺组默认 `--incomplete error`。只有符合用户意图时使用 `skip` 或 `available`，并报告实际种子。汇总表不能恢复丢失的逐种子信息。
- 使用相同固定测试集、指标归一化和评估口径。`--test-id` 不能覆盖已有冲突；`MAPE (%)` 不再乘以 100。
- `--layout overlay`（默认）在同一面板显示端点结果，`rho0` 红色、`rho1` 蓝色；`--layout panels` 分成两个测试端点面板。多个混合分支或 `--series all` 优先使用 `panels`。
- `--language zh` 配合可选 `--font-file` 使用中文。`--uncertainty none` 为默认；`sd` 只显示实测点的种子间样本标准差，不是拟合参数或曲线的置信区间。

有限样本量上的幂律拟合是描述性结果，不能单凭较高 R² 证明普适 scaling law。检查拟合质量、有效样本量个数、拟合范围、非衰减指数和 offset 边界；不要隐藏不支持幂律的结果，也不要为了更好看的拟合删种子或改变指标。

## 输出与交付

每个指标输出 PNG、PDF、SVG，并保存 `fit_parameters.json` / `.csv`、`fit_points.csv`、`fit_curves.csv`、`plot_points.csv`、`metadata.json`，逐种子入口另存 `per_seed.csv`。参数表包含 `A`、`b`、`C`、两种 R²、拟合范围和质量信息。

交付前核对图和表，说明模型、拟合空间、样本区间、实际种子、拟合质量以及混合分支是否包含基线。用户如只要求图形，仍保留机器可读拟合结果，便于复核。
