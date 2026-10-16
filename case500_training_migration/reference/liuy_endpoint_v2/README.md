# case118 端点训练：操作说明

本版本只安排 rho=0 和 rho=1 的配对端点，14 个规模、5 个种子，共 70 次正式训练。640k、1280k 和 rho=0.5 均不在队列中。

## 一条命令启动全部任务

登录服务器后：

```bash
cd /data1/liuy/linhj/pfdelta-case118-acpf/pfdelta-case118
bash scripts/endpoint_training_v2/launch_tmux.sh start
```

无需另行激活 conda，脚本使用 `/data1/liuy/miniconda3/envs/pfdelta311/bin/python`。它在 tmux 会话 `case118_endpoint_v2` 内先串行准备和核验缓存，然后启动 9 个固定任务槽：GPU 0、1、2 各 3 个。每个槽一次只运行一个模型，结束后自动接续下一个；队列尾部不足九个任务时并发数自然减少。

第一次启动可能先看到 GPU 空闲，这是缓存准备阶段。全部规模缓存共约 32.6 GiB，每个规模仅存一份供五个种子读取。脚本不会覆盖原始母库，也不会重新求解 PF 标签。

## 查看运行情况和安全断开

```bash
bash scripts/endpoint_training_v2/launch_tmux.sh status
tmux attach -t case118_endpoint_v2
```

离开 tmux：先按 **Ctrl+b**，松开，再按 **d**。之后退出 SSH 或关闭本地终端，服务器训练仍继续。不要用 Ctrl+c 代替 detach。

查看队列日志：

```bash
tail -f ../Results/0-1labels/_endpoint_v2_control/launcher.log
```

查看某次训练：

```bash
tail -f ../Results/0-1labels/40k/seed11/Logs/train.log
```

## 停止和恢复

```bash
bash scripts/endpoint_training_v2/launch_tmux.sh stop
```

这会停止派发新任务，并请求正在运行的模型安全退出。最近完整 epoch 的模型、优化器、学习率调度器和随机数状态保存在 `Model/last.pt`。正在进行但尚未完整保存的 epoch 会在恢复时重跑；不会假报训练完成。测试评测若被中断，会从已训练好的最佳模型重新评测。

待 tmux 队列窗口退出后：

```bash
bash scripts/endpoint_training_v2/launch_tmux.sh restart
```

已经完成且数据/配置校验通过的实验跳过，其余从各自断点恢复。`restart` 不会杀死仍在运行的窗口。重复 `start` 会提示现有会话，避免额外启动九个任务。启动前若检测到 GPU 上已有计算进程，脚本会报出进程并停止，不会结束其他人的任务。

## 实验和数据约定

- 端点记录数：10k、20k、40k、45k、50k、55k、60k、80k、90k、100k、110k、120k、160k、320k。
- 每个父工况为 rho=1 和 rho=0 两条记录，全部规模严格嵌套。
- 数据划分 seed 固定为 20261013；模型种子为 11、22、33、44、55，五个种子共用同一规模数据。
- 每个规模的 168 条线路父工况数最多相差 1。线路内按现有 64 个已知输入簇的母库比例抽样，所有规模均覆盖这 64 簇。这不等于每个可能的联合输入组合均得到覆盖。
- 验证和测试成员保持不变，各 8,400 个父工况、16,800 条端点记录，每条线路各 50 个父工况；不随训练规模变化。
- CANOS-PF：hidden128、k_steps20、batch64、Adam lr=5e-4、300 epochs；前5轮LinearLR warmup，随后StepLR(step_size4,gamma0.9)。验证batch50，每2轮验证，末轮也验证。
- 用验证MSE选择最佳模型，之后评测固定测试集；测试误差不参与选模或筛数据。
- CUDA scatter可能仍有细小数值不确定性，因此记录初始权重摘要、软件版本、源码摘要和全部模型种子。

## 文件位置

数据：`data/case118_endpoint_training_v2/`

```text
train/<规模>/              父ID、源图索引及数量统计
processed/<规模>/train.pt  该规模标签缓存，五个种子共用
shared/                    固定验证和测试缓存链接、父ID
audit.json                 全部规模线路/输入簇/来源批次分布审计
protocol/                  冻结版本、缓存摘要及读取检查
```

模型：`../Results/0-1labels/<规模>/seed<种子>/`

```text
Model/best.pt              最佳验证模型state_dict
Model/model.pt             指向best.pt的兼容链接
Model/last.pt              完整断点，包含优化器和随机数状态
Model/config.yaml          兼容旧工具的模型配置
Model/run_config.json      本次不可变配置
Model/provenance.json      环境和代码版本信息
Evaluation/val_history.json
Evaluation/test_metrics.json
Evaluation/per_sample.csv
Evaluation/by_line_rho.csv
Evaluation/completion.json  正式训练和测试成功后才写finished=true
Logs/train.log
Logs/console.log
progress.json
```

汇总：`../Results/0-1labels/_endpoint_v2_control/` 下的 `run_summary.csv` 和 `seed_summary.json`。后者始终注明已完成种子数，不把未完成的五种子结果当作完整结果。

`smoke_checks/` 是部署检查的独立临时实验，不能作为正式训练成绩；它们没有正式完成标志。

## 仅准备／核验，不训练

```bash
/data1/liuy/miniconda3/envs/pfdelta311/bin/python scripts/endpoint_training_v2/prepare_endpoint_data.py --build-caches
/data1/liuy/miniconda3/envs/pfdelta311/bin/python scripts/endpoint_training_v2/prepare_endpoint_data.py --verify
/data1/liuy/miniconda3/envs/pfdelta311/bin/python scripts/endpoint_training_v2/manage_endpoint_experiments.py plan
```

缓存及断点均使用原子写入；不要手工改已冻结的父工况清单或正在使用的配置。确需改变实验定义时应建立新版本。
