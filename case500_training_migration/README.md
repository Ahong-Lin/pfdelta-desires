# case500 CANOS-PF 训练迁移包

把 liuy 服务器当前 case118 的**模型、损失、训练配置与评测协议**迁移到 case500。此文件夹可以单独上传到另一台 Linux/NVIDIA 服务器，也可作为现有 `pfdelta-desires` 仓库的新子目录提交；不用替换仓库根目录的数据生成文件。

**本包负责训练及其数据缓存转换。** 原始 case500 标签由此前的 `case500_migration` 数据生成项目提供，本包不包含训练数据、权重、服务器密码或运行环境。当前只训练 `ρ=0,1` 的端点模型。

## 统一的实验设置

详见 [完整实验协议](docs/EXPERIMENT_PROTOCOL.md)。主要设置已写入 [configs/experiment.json](configs/experiment.json)：

| 项目 | 设置 |
|---|---|
| 模型 | CANOS-PF，hidden_dim=128，k_steps=20 |
| 训练 | 300 epochs，batch=64，Adam，初始 lr=0.0005 |
| 学习率 | 与 liuy 相同：5 epochs LinearLR warmup，之后每 4 epochs 乘 0.9 |
| 种子 | 11、22、33、44、55 |
| 嵌套规模 | 10k、20k、40k、45k、50k、55k、60k、80k、90k、100k、110k、120k、160k、320k |
| 验证/测试 | 固定集合，均为每条可断线路 50 个父工况；按父工况隔离 |
| 选模/测试 | 每 2 epochs 验证，选择最小验证 MSE，训练结束后在完整测试集报告 MSE/PBL |
| 队列 | 每张明确指定的 GPU 默认 3 个独立任务，tmux 后台运行 |

14 个规模 × 5 个种子 = **70 次正式训练**。k 表示端点记录数量；40k 对应 20,000 个父工况。默认生成协议的 case500 有 502 条可断普通线路，验证与测试分别为 50,200 条端点记录。数量读取生成结果，不硬编码为 case118 的 16,800。

配置中的 `epochs/batch_size/lr/hidden_dim/k_steps` 等顶层字段是实际运行参数；`protocol` 是固定训练规则的说明记录。优化器类型和学习率调度实现位于训练入口，单独改说明字段不会改变算法。

## 文件内容

```text
core/models/                     从 liuy 下载的 CANOS-PF 模型，未改计算逻辑
core/datasets/                   相同的原始样本 → CANOS 图转换代码
core/utils/                      相同的损失与 PBL 计算
configs/experiment.json          正式实验配置
scripts/prepare_case500_data.py   审查冻结划分、构建嵌套 PyG 缓存
scripts/configure_runs.py        根据缓存生成不可覆盖的五种子配置
scripts/endpoint_training_v2/    单任务训练、最佳模型选择和完整测试
scripts/manage_runs.py           每卡 3 任务的可恢复队列
scripts/summarize_results.py      逐种子和跨种子均值/标准差
setup_env.sh / requirements.txt  环境安装
start_tmux.sh                    后台启动
reference/                      liuy 原始脚本和环境版本，供核对
SOURCE_PROVENANCE.json           实际来源、源码 hash 和运行参数
VALIDATION.md                    已完成的检查及尚未实测的部分
```

`reference/` 中的 case118 脚本保留了原有路径，**不作为 case500 的运行入口**。原服务器的训练进程与文件没有被修改。

## 1. 环境安装

在目标 Linux x86_64 服务器使用 Python 3.11 和 NVIDIA GPU。安装器建立本目录 `.venv`，锁定服务器当前的 PyTorch 2.1.2/CUDA 12.1、PyG 2.6.1、NumPy 1.26.3 等核心版本。无需另装 `torch-scatter` / `torch-sparse`。tmux 需由服务器提供。

```bash
cd /你的代码目录/case500_training_migration

# 若默认 python3.11 不存在，指定已有 Python 3.11 的绝对路径。
# export PYTHON_BIN=/path/to/python3.11
bash setup_env.sh
.venv/bin/python scripts/check_environment.py
```

不要把另一台机器的 `.venv` 复制过来。显卡或驱动不支持这套环境时，先在新环境做兼容性检查并记录版本，不能悄悄替换软件版本后仍声称完全复现。

## 2. 接入已生成的数据

`--source-root` 指向**生成结果目录**，里面应有 `raw/`、`manifests/parents.jsonl`、`plan.json` 和 `splits/frozen.json`；它不是数据生成项目的源码目录。

```bash
# 先审查并冻结缓存索引，不读取并转换全部原始图。
.venv/bin/python scripts/prepare_case500_data.py \
  --source-root /data/case500_endpoint_320k \
  --output-root /data/case500_canos_caches \
  --indices-only

# 只构建 10k 训练缓存及完整固定验证/测试缓存，供先行检查。
.venv/bin/python scripts/prepare_case500_data.py \
  --source-root /data/case500_endpoint_320k \
  --output-root /data/case500_canos_caches \
  --scales 10k

# 完成所有正式规模；已有缓存会核验后复用。
.venv/bin/python scripts/prepare_case500_data.py \
  --source-root /data/case500_endpoint_320k \
  --output-root /data/case500_canos_caches \
  --scales all
```

转换程序继承生成包的固定划分，不重新打乱、删题或求解。检查两个端点的固定输入一致、rho/线路身份一致、Newton/PF 求解来源与已保存数值审计一致；拒绝不符合接口假设的数据。随后调用与 case118 相同的图构造器，保持 9 维支路特征和受控停运线路的占位边。

这些检查不等于独立再求一次 PF，也不证明动态稳定性。输入不兼容时程序报错，不静默替换标签或跳过样本。

缓存使用磁盘映射逐图写入，避免先把全部图存成 Python 对象列表。各规模生成独立 `.pt` 文件，较小规模是较大规模的精确前缀；**独立文件会占用额外磁盘**，请按实际小规模缓存大小预留所有规模和构建临时文件的空间。缓存应位于持久存储。

## 3. 生成配置并小试

正式任务配置会记录完整缓存索引的摘要，因此先完成计划使用的缓存，再冻结正式配置；小规模试跑请另用 `Results/pilot`，不要占用正式结果目录。

```bash
.venv/bin/python scripts/configure_runs.py \
  --data-root /data/case500_canos_caches \
  --results-root Results/0-1labels

# 仅在 GPU 0 做两个 optimizer steps 和一个验证 batch。
# 写入单独 smoke_checks 目录，不算完成正式实验，不消耗正式 checkpoint。
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/endpoint_training_v2/train_endpoint_run.py \
  --config Results/0-1labels/_control/configs/10k_seed11.json \
  --smoke-steps 2
```

可以使用 `--scales 10k,20k --seeds 11` 只冻结小组任务，但需要独立结果根目录，不能在原任务清单上覆盖成另一套规模/种子。默认全部 70 个任务；当前不添加 640k 或 1280k。

## 4. 启动队列

把下面 GPU 编号改成新服务器实际分配给你的显卡。默认每卡 3 个任务；若目标卡容纳不了 case500 的三任务并发，可用 `--jobs-per-gpu 1` 或 `2`，无需改变单模型 batch/训练规则。

若服务器已经设置 `CUDA_VISIBLE_DEVICES=2,3`，这里就使用 `--gpus 2,3`，不要重新编号成 `0,1`。队列检查继承的分配范围，拒绝访问该范围外的卡。正式启动前还会逐文件核验冻结索引、缓存和父工况清单的 SHA256；大缓存需要读取时间，日志会报告每份文件的进度。

```bash
# 只预览状态与分配，不训练。
.venv/bin/python scripts/manage_runs.py \
  --manifest Results/0-1labels/_control/runs.json \
  --gpus 0,1,2 --dry-run

# 后台启动；默认使用本目录 .venv/bin/python。
bash start_tmux.sh \
  --manifest Results/0-1labels/_control/runs.json \
  --gpus 0,1,2 --jobs-per-gpu 3

tmux attach -t case500_endpoint
```

离开 tmux 并保留运行：**Ctrl+b，松开，再按 d**。队列结束后保留终端，显示退出码。已有同名 tmux 时不会再次启动队列。

中断后的未完成任务会从 `last.pt` 的完整 epoch 恢复；完成任务核对身份后跳过。失败任务不会无休止自动重试。查看 `Logs/console.log` 和 `progress.json`、处理原因后，再在已停止的队列上显式传 `--retry-failed`。不要修改已经开始的任务配置；改变实验协议应使用新的结果根目录。

## 5. 结果与汇总

```text
Results/0-1labels/40k/seed11/
  Model/best.pt                  验证 MSE 最优权重
  Model/model.pt                 指向 best.pt
  Model/last.pt                  含优化器及随机状态的恢复 checkpoint
  Model/run_config.json          完整配置
  Model/provenance.json          运行环境和源码摘要
  Logs/train.log
  Evaluation/val_history.json
  Evaluation/test_metrics.json
  Evaluation/per_sample.csv
  Evaluation/by_line_rho.csv
  Evaluation/completion.json     完成训练且完整测试通过才标 finished
```

```bash
.venv/bin/python scripts/summarize_results.py --results-root Results/0-1labels
```

生成 `_summary/per_seed.csv` 和 `_summary/by_scale.csv`。汇总只计正式完成的任务，明确列出实际完成种子；较差的种子不会因成绩不好被排除。

## 验证范围

本地完成来源 hash 对照、配对/划分/缓存和启动配置测试，详见 [VALIDATION.md](VALIDATION.md)。尚未在你的新服务器执行完整 case500 GPU 训练，也没有据此测得 case500 的最终成绩、吞吐或每卡三任务显存需求。先小试再正式运行。
