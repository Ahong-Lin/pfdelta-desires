# 本地验证记录（2026-10-14）

本包已完成代码准备、静态审查及下述测试。**尚未在目标 Linux 服务器安装完整 Julia 依赖，也尚未执行真实 case500 的 ACOPF → 两端 NR 全流程小试；没有生成正式 420,400 条记录。** 因此生成成功率、实际吞吐、内存和磁盘占用仍需上机小试确认。

| 检查 | 实际结果 |
|---|---|
| 独立读取 MATPOWER 网络，逐条断开检查连通性并排除变压器 | 500 母线、733 支路、728 在运，502 条候选线路；测试通过 |
| Python 3.10.20，按 requirements.txt 锁定依赖 | 8 项 unittest 全部通过 |
| 父工况配对、输入去重、线路配额、训练集嵌套 | 小型合成清单测试通过 |
| 固定划分复用、篡改检测、smoke 覆盖正式规模、损坏元数据拒绝 | 测试通过 |
| Julia 1.9.4 解析所有 18 个 Julia 文件 | 无语法错误 |
| 6 worker、连续 3 批、队列容量溢出边界、worker 故意异常、失败候选反馈、真实 producer 补采样 | 标准库 stub 测试 16/16 断言通过，退出码 0；最终日志无异常 socket 信息 |
| Bash 脚本语法 | 4 个启动/安装脚本通过 bash -n |
| tmux 启动命令中的空格路径、环境变量转发 | mock tmux 检查通过；没有启动服务器任务 |
| 安装用 PYTHON_BIN 覆盖与运行环境隔离 | 已修复启动脚本绕过项目虚拟环境的问题；新增两项回归测试通过 |

stub 并发测试使用真实采样调度函数、替代的求解结果，**不验证 ACOPF/NR 的数值效果**。`--verify-raw` 核验最终端点保存的身份、方法、残差和固定控制量审计元数据，也不冒充再次独立数值求解。真正的独立功率平衡计算在每条标签生成时执行。

上机验收顺序：

```bash
bash setup_env.sh
bash run.sh plan
bash run.sh smoke /持久磁盘/case500_smoke
bash run.sh verify /持久磁盘/case500_smoke
```

小试应得到 16 个父工况、32 条端点：训练 16 条、验证 8 条、测试 8 条；`state.json.completed=true`，且生成 `splits/frozen.json`。查看 `storage_estimate.json` 后再运行完整规模。

可以单独复现非数值并发测试：

```bash
julia --startup-file=no tests/test_distributed_lifecycle.jl
```

完整配额及方法是本包的**实验协议目标**，不是已经完成的数据集统计。输入聚类分层改善集合之间的可比性，不证明严格同分布或动态稳定性。
