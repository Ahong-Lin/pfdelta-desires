# PFDelta 绘图技能包

本目录可作为 GitHub 仓库内容迁移到其他机器。包含两个技能目录，数据、模型权重和字体由使用者提供；脚本只读取已有评估结果。

| 目录 | 用途 |
| --- | --- |
| [pfdelta-comparison-plot](pfdelta-comparison-plot/SKILL.md) | 对比仅端点训练和增加中间 rho 样本后的实测指标；详细格式见该目录的 [README](pfdelta-comparison-plot/README.md) |
| [pfdelta-scaling-law](pfdelta-scaling-law/SKILL.md) | 拟合幂律及带平台项幂律，导出参数、拟合数据和曲线 |

## 安装

需要 Python 3.9+。在本目录创建环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'matplotlib>=3.6' numpy scipy
```

Windows 使用 `.venv\Scripts\activate` 激活。SciPy 用于带平台项模型及在线性空间拟合纯幂律；默认的对数空间纯幂律拟合只需 NumPy 和 Matplotlib。脚本支持没有桌面的服务器。

将需要的整个技能目录复制到 Codex 技能目录（默认 `~/.codex/skills/`），即可使用 `$pfdelta-comparison-plot` 或 `$pfdelta-scaling-law`。也可以直接运行脚本，无需安装 Codex 技能。

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R pfdelta-comparison-plot "${CODEX_HOME:-$HOME/.codex}/skills/"
cp -R pfdelta-scaling-law "${CODEX_HOME:-$HOME/.codex}/skills/"
```

不要只复制 `SKILL.md`；应连同该目录下的 Python 脚本与说明一起复制。已有同名技能时，先比较内容再自行选择合并或替换。

## 直接运行

在本目录执行，替换示例数据路径即可；`--case` 可填写 case30、case118、case500 或其他系统名称。

```bash
python pfdelta-comparison-plot/plot_pf_comparison.py \
  --case case500 \
  --endpoint-root /path/to/endpoint-results \
  --mixed-root /path/to/mixed-results \
  --metrics MSE --output-dir ./figures/comparison

python pfdelta-scaling-law/plot_pf_scaling.py \
  --case case500 \
  --endpoint-root /path/to/endpoint-results \
  --metrics MSE --model power --fit-space log \
  --output-dir ./figures/scaling
```

两个工具还接受逐种子 `--csv` 和汇总点 `--points-csv`。不假设不同系统使用相同样本量、种子或测试集。默认使用输入中的有效种子，不自动沿用历史图的事后筛选名单。使用中文标签时加 `--language zh`，必要时通过 `--font-file` 指定本机字体。

图形输出为 PNG/PDF/SVG，伴随可复核的数据表和元数据；scaling 工具额外保存拟合参数及曲线。模型和数据处理约定见各自的 `SKILL.md`，所有参数以对应脚本 `--help` 为准。

## 拟合选项

默认 `--model power --fit-space log` 与原 IEEE30/118 scaling 脚本一致：先对每个样本量的种子取算术均值，再用各样本量等权的对数最小二乘拟合 `E=A*(N/40000)^(-b)`。`--model offset` 增加非负常数项 `C`；`--model both` 同时导出两种结果。`--fit-min 40k --fit-max 320k` 可限制拟合区间，范围外的实测点仍显示。

若已经用对比图工具导出了 `per_seed.csv` 或 `plot_points.csv`，可直接传给 scaling 工具的 `--csv` 或 `--points-csv`。默认仅拟合端点训练；混合路径需明确选择 `--series mixed` 或 `all`，并建议使用 `--layout panels`。

## 验证范围

默认幂律已与原 case30/118 的四个指标、两个测试端点共 16 条曲线逐一比较，参数和对数 R² 差异不超过浮点精度。带平台项及线性空间拟合已用已知参数的合成数据验证；独立 agent 已按技能文件完成真实 case118 表的两种模型拟合与图形检查。高样本内 R² 只说明所选范围内的拟合程度，不能据此单独证明误差平台或普适规律。
