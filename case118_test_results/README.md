# case118 固定测试集评测结果

本目录保存 liuy 服务器端点训练实验 `case118_endpoint_v2_20261013` 的**已有评测结果**。导出日期为北京时间 2026-10-22，准确快照时间见 `MANIFEST.json`。

共收录 60 个已经完成 300 轮训练及完整测试的模型：

- 训练量：10k、20k、40k、45k、50k、55k、60k、80k、90k、100k、110k、120k。
- 每个训练量均包含种子 11、22、33、44、55，未按成绩筛选种子。
- 训练仅使用 rho=0、rho=1 端点。训练量表示图记录数，配对父工况数为图记录数的一半。
- 快照时 160k、320k 的 10 个任务尚未全部完成，本包不收录它们的中途结果；具体排除列表见清单。

本包只迁移既有结果，没有重新训练或推理，也没有修改服务器结果。模型权重、训练集、验证集、测试缓存均不在包内。下载后可直接汇总和画图；不能只靠本包重新运行模型推理。

## 测试口径

所有模型均在训练结束后加载其验证集 MSE 最佳权重 `best.pt`，然后评测同一份固定的 `shared/test.pt`：

- 16,800 条端点图记录，8,400 个父工况。
- 168 条可停运线路，每条 50 个父工况。
- 每个父工况都有 rho=0 和 rho=1；每个 rho 各 8,400 条记录。
- 测试缓存的原始路径、字节数和 SHA-256 记录在 `MANIFEST.json` 的 `test_identity` 中。测试缓存本体不随结果上传。

`canos_pf_mse` 是逐图 PQ、PV、Slack、Branch 四项 MSE 之和再取平均；`PBL Mean` 是逐母线有功/无功失配向量的模长，先在图内平均再在样本间平均。`PBL Max` 是所选集合内最大的母线失配，不是逐图最大值的平均。其余指标定义保存在每个 `test_metrics.json` 的 `definitions` 中。这里的模型不曾报告 MAPE，因此不能由现有汇总值推算 MAPE。

## 目录和文件

```text
case118_test_results/
  README.md
  MANIFEST.json          # 来源、快照时间、任务列表、源文件大小与 SHA-256
  RUNS.csv               # 60 个模型的索引
  SHA256SUMS             # 全包文件校验值，不包含本文件自身
  10k/seed11/
    Evaluation/
      test_metrics.json  # 整体、分 rho、分线路统计及指标定义
      per_sample.csv     # 逐样本误差，含父工况 ID、线路和 rho
      by_line_rho.csv    # 分线路、分 rho 的统计
      completion.json   # 完成标记、最佳轮次、权重标识及测试结果
      summary.json      # 原评测器保存的汇总，与 test_metrics.json 一致
      val_history.json  # 验证历史
    Model/
      run_config.json
      config.yaml
      provenance.json
  ...
  120k/seed55/
```

原配置中的服务器绝对路径被原样保留以便追溯，绘图脚本不会打开这些路径或加载训练数据。不要因为本机不存在这些旧路径而改写原始配置。

## 在公司服务器下载和校验

只下载结果与绘图工具，避免下载同仓库的其他内容：

```bash
git clone --filter=blob:none --sparse https://github.com/Ahong-Lin/pfdelta-desires.git
cd pfdelta-desires
git sparse-checkout set case118_test_results pfdelta_plot_skills
(cd case118_test_results && sha256sum -c SHA256SUMS)
```

macOS 的校验命令可替换为 `shasum -a 256 -c SHA256SUMS`。若仓库已在本机，先更新仓库再取本目录即可。

## 直接画图，无需 GPU 或 PyTorch

在仓库根目录执行；绘图需要 Python 3.9+：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'matplotlib>=3.6' numpy

python pfdelta_plot_skills/pfdelta-comparison-plot/plot_pf_comparison.py \
  --case case118 \
  --endpoint-root case118_test_results \
  --seeds 11,22,33,44,55 \
  --incomplete error \
  --metrics MSE,PBL,rPBL \
  --output-dir output/case118_comparison

python pfdelta_plot_skills/pfdelta-scaling-law/plot_pf_scaling.py \
  --case case118 \
  --endpoint-root case118_test_results \
  --seeds 11,22,33,44,55 \
  --incomplete error \
  --metrics MSE,PBL,rPBL \
  --model power --fit-space log \
  --output-dir output/case118_scaling
```

两个脚本直接读取本目录保留的原始结构。默认对每个量级的五个种子取算术均值，分别绘制 rho=0 和 rho=1；不会自动移除高误差种子。Scaling 拟合只使用此快照实际包含的 10k–120k 数据，不能把缺失的 160k、320k 当作已经完成。使用带平台项的 `--model offset` 或 `both` 时，还需安装 SciPy。

文件内容完整性由 `SHA256SUMS` 验证；`MANIFEST.json` 同时保留导出时服务器原始 540 个文件的哈希，便于区分源结果和本地新增的说明、索引及校验文件。
