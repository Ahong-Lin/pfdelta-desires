# Power Flow 对比图迁移工具

将本目录的 `plot_pf_comparison.py` 和 `README.md` 一起复制到另一台机器，即可为 case30、case118、case500 或其他系统绘制相同风格的对比图。脚本不依赖原仓库的绘图脚本，不执行模型推理，也不需要 PyTorch、Julia 或模型权重。数据仍由你提供；本目录不包含训练数据、模型、服务器数据路径或字体文件。

默认图形与 `IEEE 30/Results/Comparision`、`IEEE 118/Results/Comparision` 的主要表达方式一致：左右两幅图分别是 `rho0`（N-1 线路停运）和 `rho1`（正常拓扑）；蓝色圆点实线连接仅端点训练的结果，粉色菱形虚线表示增加中间 rho 样本的结果。不同端点基数的混合训练分支分别连接，并与相应的端点结果共享起点。横轴是**总训练样本数**，两轴默认采用 log2；默认只画算术均值，不画误差棒。

## 1. 安装与迁移

需要 Python 3.9 及以上、Matplotlib 3.6 及以上。建议使用独立环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'matplotlib>=3.6'
python plot_pf_comparison.py --help
```

例如，从本机将两个文件复制到服务器。请将命令中的服务器地址和目标目录替换为自己的值：

```bash
scp plot_pf_comparison.py README.md user@server:/path/to/plot-tool/
```

Windows 可使用 `py -m venv .venv` 创建环境，再运行 `.venv\Scripts\activate`；其他绘图参数不变。脚本使用无界面绘图后端，可以在没有桌面的服务器上运行。

## 2. 直接读取新模型的评估目录

如果结果采用本项目的运行目录结构，优先使用这一入口：

```text
endpoint-results/
  40k/seed_11/
    Evaluation/completion.json
    Evaluation/test_metrics.json
    Model/run_config.json
  80k/seed_11/...
mixed-results/
  40k+5k/seed_11/
    Evaluation/completion.json
    Evaluation/test_metrics.json
    Model/run_config.json
```

`--endpoint-root` 与 `--mixed-root` 都可以重复指定。`--results-root` 是 `--endpoint-root` 的别名。脚本读取已完成且非 smoke 的评估记录；未完成记录会跳过并记录原因。训练样本组成可来自运行配置中的 `endpoint_samples`、`midrho_samples`，也支持 `40k+5k/seed_11` 这类目录命名。

case118 示例：

```bash
python plot_pf_comparison.py \
  --case case118 \
  --endpoint-root /path/to/case118/endpoint-results \
  --mixed-root /path/to/case118/mixed-results \
  --metrics MSE,PBL,rPBL \
  --seeds 11,22,33,44,55 \
  --test-id T168-v2 \
  --output-dir ./figures/case118
```

case500 使用相同的数据结构和命令，仅替换系统名称及输入目录：

```bash
python plot_pf_comparison.py \
  --case case500 \
  --endpoint-root /path/to/case500/endpoint-results \
  --mixed-root /path/to/case500/mixed-results \
  --metrics MSE,PBL,rPBL \
  --seeds 11,22,33,44,55 \
  --output-dir ./figures/case500
```

`--case` 是图中的系统标识，不会为 case500 套用 case30 或 case118 的样本数、种子名单或数值。没有混合训练结果时，可以省略 `--mixed-root`。

## 3. 使用逐种子 CSV

`--csv` 接受一个或多个 CSV 文件，并将相同测试端点、训练样本组成和指标的种子结果求算术均值。支持已有的原始／筛选后评估表，也可以整理为下面的标准表。

| 列名 | 含义 |
| --- | --- |
| `test` | 必填，`rho0` 或 `rho1`，表示测试端点 |
| `endpoint_samples` | 端点训练样本数；仅端点训练时等于总样本数 |
| `midrho_samples` | 中间 rho 训练样本数；仅端点训练时为 `0` |
| `total_train_samples` | 总样本数，应等于前两列之和 |
| `seed` | 单次训练使用的整数种子 |
| `test_id` | 可选，固定测试数据的版本标识；例如 `T168-v2`。仅拓扑相同不能说明是同一测试集 |
| `method` | 可选，用于区分不同中间样本方案；例如 `rho05`、`multi_rho`。缺省时所有中间样本视为同一方案 |
| `canos_pf_mse` | MSE 指标列 |
| `PBL Mean` | PBL 指标列 |
| `Relative PBL Mean` | rPBL 指标列 |
| `MAPE (%)` | MAPE 指标列，数值已经是百分数 |

只需提供准备绘制的指标列。下面是**仅用于说明格式的虚构数据**，请用实际评估结果替换后使用：

```csv
test,test_id,endpoint_samples,midrho_samples,total_train_samples,seed,canos_pf_mse
rho0,MY_TEST,40000,0,40000,11,0.12
rho1,MY_TEST,40000,0,40000,11,0.10
rho0,MY_TEST,80000,0,80000,11,0.08
rho1,MY_TEST,80000,0,80000,11,0.07
rho0,MY_TEST,40000,5000,45000,11,0.09
rho1,MY_TEST,40000,5000,45000,11,0.08
```

```bash
python plot_pf_comparison.py \
  --case case500 \
  --csv /path/to/endpoint_per_seed.csv /path/to/mixed_per_seed.csv \
  --metrics MSE \
  --output-dir ./figures/case500
```

CSV 中每行应对应一个真实的种子结果。不要将已经求过均值的结果伪装成一个新种子，也不要将多个不同中间 rho 实验混在同一训练样本组成下。已经汇总好的数据应使用下一节的 `--points-csv`。

同一次调用不要重复输入包含相同指标、相同组成、相同种子的表；重复记录会报错。若要画多种混合方案，请填好 `method` 列；同一方案的不同端点基数会自动分开连接。

## 4. 复现原有 Comparision 图中的均值

如果要保留旧图中每个点的数值和种子选择，直接输入旧图导出的绘图点表。脚本支持：

- IEEE 30：`plot_points_PBL.csv`、`plot_points_rPBL.csv`、`plot_points_MSE.csv`、`plot_points_MAPE.csv`。
- IEEE 118：`sample_type_plot_points_PBL.csv`、`sample_type_plot_points_rPBL.csv`、`sample_type_plot_points_MSE.csv`、`sample_type_plot_points_MAPE.csv`。

例如，在原仓库根目录执行以下命令，输入文件保持只读：

```bash
python comparison_plot_migration/plot_pf_comparison.py \
  --case case30 \
  --points-csv 'IEEE 30/Results/Comparision/PBL_rPBL/plot_points_PBL.csv' \
  --metrics PBL \
  --test-id T34 \
  --output-dir ./comparison_plot_migration/reproduced/case30_PBL

python comparison_plot_migration/plot_pf_comparison.py \
  --case case118 \
  --points-csv 'IEEE 118/Results/Comparision/MSE_MAPE/sample_type_plot_points_MSE.csv' \
  --metrics MSE \
  --test-id T168 \
  --output-dir ./comparison_plot_migration/reproduced/case118_MSE
```

IEEE 118 绘图点表使用通用列名 `value`，因此每次必须明确选择**一个指标**，例如 `--metrics MSE`；文件名本身不作为指标定义。汇总点中的共享起点直接复用，不会再被当作独立种子求一次均值。

旧图采用过分组、分指标的事后种子筛选。此工具**不会自动将那些名单应用到新数据**；逐种子模式默认使用输入中的全部有效种子。要保留旧均值，请用 `--points-csv` 或明确提供旧的已筛选逐种子表。默认均值图与原图的统计表达一致，`--uncertainty sd` 才会增加种子间标准差误差棒。字体版本和 Matplotlib 版本可能导致排版细节不同。

## 5. 种子完整性与测试集标识

`--seeds 11,22,33` 同时表示：只使用这些种子，并要求各组拥有这些种子。若某组缺失指定种子，`--incomplete` 控制处理方式：

| 参数 | 行为 |
| --- | --- |
| `--incomplete error` | 默认；报错，提示补齐或明确改变处理方式 |
| `--incomplete skip` | 跳过不完整的组，并记录原因 |
| `--incomplete available` | 使用该组已有的指定种子，记录实际数量 |

未指定 `--seeds` 时，使用输入中已有的种子；不同组的种子集合不一致时会提示，并在输出中保留实际种子名单。汇总点已失去逐种子数值，不能通过种子筛选重新计算其均值。

原生目录入口检查记录的 `test_cache` 标识和 `num_samples`，避免把明显不一致的测试集混在一起。`--test-id` 可为缺少测试集标识的表声明标识；若输入已明确标识其他测试集，则不能用该参数覆盖冲突。路径、名称和样本数一致并不等于完成了测试内容的密码学一致性验证；仅在确定测试数据相同的情况下合并结果。

## 6. 图形参数与中文字体

```bash
python plot_pf_comparison.py \
  --case case118 \
  --csv /path/to/results.csv \
  --metrics MSE,PBL,rPBL \
  --language zh \
  --font-file /path/to/chinese-font.ttf \
  --uncertainty none \
  --sizes 40k,45k,50k,55k,60k,80k,85k,90k,100k,160k \
  --xscale log2 --yscale log2 \
  --dpi 300 \
  --output-dir ./figures/case118_cn
```

- `--metrics`：默认 `all`；也可指定逗号分隔的 `MSE,PBL,rPBL,MAPE`。
- `--sizes`：按总训练样本数筛选端点点和混合点；省略时使用输入中的样本数。
- 如果需要混合曲线从 40k 或 80k 基线出发，`--sizes` 中也保留该基线；被过滤的起点不会凭空补入。
- `--language`：默认 `en`；`zh` 使用中文标签和固定的双面板布局。
- `--font-file`：可重复指定，用于注册本机字体文件。字体文件需自行准备。
- `--uncertainty`：默认 `none`；`sd` 表示种子间样本标准差，不是置信区间。
- `--xscale`、`--yscale`：支持 `log2` 或 `linear`；默认均为 `log2`。
- `--title`：覆盖图标题；`--dpi`：PNG 输出分辨率，默认 `300`。

中文模式优先寻找 SimSun，也会尝试 Noto Serif CJK SC、Noto Sans CJK SC、Source Han Sans SC、Microsoft YaHei 和 PingFang SC。若未找到可用中文字体，脚本会明确报错；可通过 `--font-file` 提供字体，或改用 `--language en`。在另一台机器上复现宋体／Times New Roman 的排版，需要该机器也有相应字体。

## 7. 输出与指标口径

每个指标输出 PDF、SVG、PNG。输出目录还保存规范化的 `per_seed.csv`（逐种子输入时）、实际用于绘图的 `plot_points.csv` 和 `metadata.json`。元数据记录来源文件及哈希、种子集合、跳过的记录、测试集标识、指标说明和绘图参数，便于追溯图中的点。

本工具读取已经计算好的指标，不会根据其他列重新推导 MSE、PBL、rPBL 或 MAPE。跨实验比较前，应确保指标公式、归一化、物理量范围、测试样本和 MAPE 分母的 epsilon 等口径一致。尤其是 `MAPE (%)` 已经使用百分数单位，不应再次乘以 100。原生 `test_metrics.json` 若没有 MAPE，就不能仅凭 MSE 换算 MAPE；请提供已有的 MAPE 评估表，并显式选择指标。

若出现缺组、种子不完整、测试集冲突、重复记录或无效数值，优先核对输入与元数据；不要为了使曲线更平滑而改变种子集合。

## 8. 原脚本与验证记录

本工具抽取并合并了原仓库的三个文件：

- `scripts/plot_ieee30_sample_type_comparison.py`
- `scripts/plot_case118_sample_type_comparison_current.py`
- `scripts/comparison_figure_layout.py`

在本地使用 case30 和 case118 的真实逐种子表，与原 `Comparision` 绘图表逐点核对：两个系统的 MSE、PBL、rPBL、MAPE 共 8 组均值和种子名单完全一致（浮点差值为 0）。原表中重复出现的混合曲线起点只保留一份数据，在绘图时共享，因此规范化表的行数会减少。已经实际导出并检查中文 case30 图、英文 case118 图。

原生训练目录的读取、种子完整性、测试标识和样本数检查，已用合成目录测试；并生成了标明为合成数据的 case500 示例图。这验证的是结果格式兼容性，不代表已经评估了真实 case500 模型。

这份工具不自动拟合 scaling law，只展示实际已有的测试指标随训练样本数的变化；每个点按种子等权平均。中间样本数指新增图记录数，横轴也是图记录数，不是父工况数。
