# SDAR on ALFWorld — Qwen2.5-3B, 4×A100 80GB

独立的 **Qwen2.5-3B-Instruct SDAR** 仓库，直接复用 [官方 SDAR](https://github.com/ZJU-REAL/SDAR) 的 GRPO＋门控自蒸馏实现，源码固定为 `29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`。

从原八卡发布 `e8d323f2da95305c735ebfbae482f57b38ac6898` 拆出。本仓库只跑 3b，使用完整 **4×A100 80GB**。只调整 GPU/Ray/FSDP 的四卡资源绑定、默认CPU预算和本尺寸启动入口；**不改上游算法、Teacher、skill、loss、全局采样量、训练步数和评测集**。原八卡仓库保持不动。

## 1. 固定方法与四卡配方

| 项目 | 本仓库 |
| --- | --- |
| 初始化 | Qwen/Qwen2.5-3B-Instruct，revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| 方法 | 官方 SDAR：原生 GRPO＋sigmoid门控自蒸馏 |
| 蒸馏 | gate_beta=5.0，sdar_coef=0.01，skill_all=false |
| Teacher | 当前Actor的无梯度评分视角，额外看到官方静态ALFWorld skill |
| Reference | 固定基模，只用于独立reference KL，不是Teacher |
| GPU | 4个FSDP rank；TP=2，即2个生成副本 |
| 全局采样 | 每update 16任务×8轨迹=128条episode，不随卡数改变 |
| 训练 | 150 updates，LR=1e-6，PPO epoch1，原生token-mean |
| PPO batch | 全局决策行mini-batch256；每卡micro-batch32 |
| 其他原生项 | reference KL=0.01 low_var_kl；entropy=0.001；非法动作惩罚=0.1 |
| 交互 | max steps50、history2、prompt2048、response512 |
| 采样 | 训练temperature1；验证temperature0.4；top_p1、top_k=-1 |
| 验证 | update0及每10步，固定seen140，每任务一条rollout |
| 保存 | 每20步及最终150，原生max_actor_ckpt_to_keep=2 |

Student只看到正常交互历史。Teacher用静态skill对Student已采样的相同token和prefix评分，不重新生成动作，不替换轨迹；验证不给Student skill。没有Route-SFT、RETHINK、Analyzer、局部hint或外接API。原生门控及梯度路径不改：

\[
g_t=\operatorname{sg}\!\left[\sigma\!\left(5(\log\pi_T(y_t)-\log\pi_S(y_t))\right)\right].
\]

四卡下配置的256行全局mini-batch对应每rank64行，micro32，由原生梯度累积补齐；八卡下是每rank32行。保留原生按实际决策行数调整批量的逻辑，不额外改写loss归一化；不同GPU拓扑不承诺逐位数值一致。精确配方见 [experiment.json](configs/experiment.json)，方法与适配边界见 [复现合同](docs/REPRODUCTION.md)。

## 2. 干净服务器安装

需要Linux、Python3.12、兼容CUDA12.8的驱动、四张已获配 **A100 80GB**；默认32个可用逻辑CPU，建议256GiB或更多主机内存。此配方不是A100 40GB配方。提前给模型、原生续训状态和导出留足空间。

```bash
git clone https://github.com/leiye302/sdar-alfworld-qwen25-3b.git
cd sdar-alfworld-qwen25-3b
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil
python src/setup.py --framework
python -m pip install -r configs/requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install -e .deps/sdar -c configs/requirements.txt
python -m pip check

cp .env.example .env
# 按实际分配修改SDAR_GPUS、SDAR_CPUS和ALFWORLD_DATA。
set -a
source .env
set +a
python src/setup.py --download-model 3b
alfworld-download
python src/setup.py --verify
python scripts/check_release.py
python -m pytest tests -q
```

FlashAttention需要匹配的wheel，或CUDA Toolkit/nvcc编译。依赖使用仓库固定约束，不另装未约束上游依赖。全部源码、基模、静态skill和ALFWorld资源来自公开来源；不依赖维护者本机、私有服务器、旧断点或私人数据。大权重/运行日志不随仓库上传。

## 3. 启动本尺寸

```bash
# 不占GPU，只检查四卡配置。
bash scripts/train.sh --dry-run

# 已固定 3b，无需另传 --size。
bash scripts/train.sh

# 结束后合并原生四rank权重。
python src/run.py export --size 3b

# 确认旧控制器与worker已退出，且原生断点完整，再续跑。
bash scripts/train.sh --resume
```

建议在tmux/调度器里运行。兼容入口 `bash scripts/run_both.sh` 在本仓库也只运行 3b，然后导出/生成曲线，不运行另一尺寸。同一组四卡不要同时跑两个实验。不额外加torchrun；由Ray启动四个FSDP rank和两个TP2副本。默认拒绝覆盖已有输出，并使用独立Ray临时目录和控制器锁。

**不能用原八卡完整断点直接续跑本四卡配方。** 续跑必须匹配模型、配方、任务清单和world_size=4，四rank的model/optimizer/extra_state及data.pt齐全。TextWorld环境迭代器不在原生断点内，不承诺中断前后轨迹逐位一致。

## 4. 同一评测集与结果

固定 [seen140_tasks.json](configs/seen140_tasks.json)：valid_seen的140个唯一任务，seed=2026+i，max steps50、history2、response512、temperature0.4。与原对比仓库使用相同任务和生成预算；不是官方128条验证的原样复现。

```bash
bash scripts/eval.sh --model "$PWD/runs/sdar_3b/hf150" \
  --run-dir "$PWD/runs/sdar_3b_final_eval"
python src/report.py
```

训练的update0为基模初始性能，之后每10步同一seen140；结果保存在 `runs/sdar_3b/metrics.jsonl`、`evaluation/update_*/summary.json` 和逐决策token/text记录。报告固定150步，不根据测试集最高成绩挑断点。严格格式指标只做遥测，不修改原生环境执行、奖励或训练门禁。独立评测不启用Teacher或skill。

## 5. 存盘、验证范围与许可

本次四卡拆分继续使用原生SDAR保存/保留逻辑，不复制GRPO仓库的事务式保存改造。max_keep=2是原生保留参数，不承诺异常中断或跨进程续跑后磁盘恰好只有两份；续跑前应检查实际目录及完整性。续训文件、最终HF导出及存盘中间峰值都计入磁盘预算。

12份核心文件及整个小补丁有SHA锁。CPU检查覆盖原生SDAR梯度、Teacher detach、四卡资源/批量、八卡断点拒绝、固定seen140、原生环境并发和shell入口；GitHub Actions从干净Linux检出公开准备上游。**尚未在4×A100上运行正式训练或NCCL/vLLM/FSDP恢复验收，不提供预填成绩。** 首次真机须确认真实环境交互、Teacher/gate/loss/梯度有限、参数更新及完整存盘。

包装代码/框架采用Apache-2.0，模型按自身许可使用（1.5B为Apache-2.0，3B为Qwen Research License）。详见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 2026-10-26 执行设置更正

撤回额外添加的串行环境分组，恢复官方每环境一个 Ray actor、每 actor 0.1 CPU。模型尺寸、获配卡数、训练量、loss、固定 seen140、验证频率和已约定的断点规则不变。未改写上游 rollout、Teacher 或优化器。

已有 checkout 的旧 `.deps/sdar` 仍带旧补丁：在训练停止并保留断点后，将它改名备份，再运行 `python src/setup.py --framework`；或在新的目录重新克隆。不要删除 `runs/`，也不要在运行中的任务里替换依赖。旧配方与新配方的绑定 SHA 不同，禁止绕过恢复校验直接续跑。此更正通过 CPU/干净检出检查不等于已完成 A100 CUDA/NCCL/vLLM 实测。
