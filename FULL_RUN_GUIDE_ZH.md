# mllm-reroute + Cold/Ghost：全量正式实验运行指南

> 本文只给**正式全量运行**，不使用 `--limit`、不使用 `--debug-updates`，也不把 CPU smoke/test 命令当作实验命令。基础环境安装仍可参考根目录 `RUN_GUIDE_COLD_GHOST_ZH.md`；从本文件开始执行的是完整训练、完整 benchmark、完整效率实验、完整消融和结果汇总。

---

## 0. 全量实验到底包含什么

本仓库定义：

- 原项目正式配置：38 份；
- Cold/Ghost 正式配置：24 份；
- Ghost 真正需要训练的 checkpoint family：12 份，因为 compact 与 stagewise 在同一模型/预算/调度下共享权重；
- 正式评测 task config：12 个：`pope`、`gqa`、`mmbench`、`mme`、RefCOCO/+/g 的 8 个 split；
- 主 benchmark：38×12 + 24×12 = **744 个正式子运行**；
- 主效率实验：62 个配置做 prefill FLOPs/KV profile，再做 62 个 CUDA-event runtime benchmark；
- 完整消融：12 个训练 family × 5 种 ablation × 4 个代表任务 = **240 个正式消融子运行**；
- 完整诊断：12 个训练 family 在完整 validation manifest 上运行 freshness/reactivation/residual diagnostics。

这里的“完整”指代码会遍历完整数据集和完整配置矩阵；不是说这些实验结果已经替你在 GPU 上跑完。

---

# 1. 每次登录服务器先执行

假设项目克隆在：

```bash
$HOME/projects/mllm_reroute
```

每次 SSH 登录服务器后：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate reroute_coldghost
cd "$HOME/projects/mllm_reroute"
export CUDA_VISIBLE_DEVICES=0
export CONDA_ENV=reroute_coldghost
export HF_HOME="$HOME/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
unset BBOX_COORD_FORMAT REFCOCO_PROMPT_STYLE
```

这些命令分别恢复 conda、进入本项目环境、进入仓库、只使用一张 GPU，并设置 Hugging Face 缓存和原评测协议需要的环境状态。

本项目的正式训练/评测入口按单 GPU、batch size 1 设计。不要把下面命令替换成 DDP 或 `accelerate launch` 后仍认为是同一协议。

---

# 2. Ghost 独立训练数据格式

Ghost 训练数据**不能来自正式评测集**。准备一个独立 JSONL，例如：

```text
data/ghost_source.jsonl
```

每一行至少包含：

```json
{"sample_id":"train_000001","image_id":"img_000001","image_path":"images/000001.jpg","question":"What is shown in the image?"}
```

必须有四个非空字段：

- `sample_id`：每条样本唯一；
- `image_id`：同一图片的多个问题应使用同一个 image_id；
- `image_path`：相对 JSONL 所在目录或绝对路径；
- `question`：给 LVLM 的问题文本。

不要人工先拆 train/val；正式流程会按 image group 做固定 seed=42 的 deterministic split，同一个 image_id 或完全相同 RGB 像素不会跨 train/val。

下面假设源文件是：

```bash
export SOURCE_JSONL="$PWD/data/ghost_source.jsonl"
```

确认文件存在：

```bash
test -f "$SOURCE_JSONL"
wc -l "$SOURCE_JSONL"
```

---

# 3. 第一步：建立完整评测图片排除索引

正式训练前必须先枚举原项目全部评测任务的图片，以防 Ghost 训练数据泄漏 benchmark。

```bash
python -m cold_ghost.cli index-eval \
  --out data/eval_images.jsonl
```

这一步会通过固定的 lmms-eval task 定义读取正式评测图片，并生成：

```text
data/eval_images.jsonl
data/eval_images.meta.json
```

只有全部 requested tasks 完成后 `.meta.json` 才标记 `complete=true`。中途中断后不要手工伪造 meta 文件。

---

# 4. 第二步：完整训练/验证数据划分与泄漏审计

```bash
python -m cold_ghost.cli split-data \
  --source "$SOURCE_JSONL" \
  --train data/ghost_train.jsonl \
  --val data/ghost_val.jsonl \
  --val-fraction 0.1
```

然后：

```bash
python -m cold_ghost.cli audit-data \
  --train data/ghost_train.jsonl \
  --val data/ghost_val.jsonl \
  --eval-index data/eval_images.jsonl
```

正式训练前这条命令必须成功。它会检查：

- train/val sample_id 不重合；
- image_id 不重合；
- EXIF-normalized RGB SHA-256 不重合；
- train/val 与全部评测图片不重合；
- 与仓库 `bench_data/*.jpg` 不重合。

如果之前已经生成这两个 split，再执行 `split-data` 会拒绝覆盖。要重新划分时先手工备份/删除旧文件，并确保你知道自己为什么重建数据身份。

---

# 5. 查看正式配置矩阵

全部原配置：

```bash
python -m cold_ghost.cli list --group original
```

全部 Ghost 配置：

```bash
python -m cold_ghost.cli list --group ghost
```

只看某个模型：

```bash
python -m cold_ghost.cli list --group ghost --model llava15
python -m cold_ghost.cli list --group ghost --model qwen25vl
```

只看某个预算：

```bash
python -m cold_ghost.cli list --group ghost --tier avg64
python -m cold_ghost.cli list --group ghost --tier avg128
python -m cold_ghost.cli list --group ghost --tier avg192
```

---

# 6. 正式训练：一次训练全部 12 份 Ghost checkpoint

下面是**正式全量训练**，没有 debug updates：

```bash
bash scripts/ghost/train_all.sh \
  --group ghost \
  --model all \
  --tier all \
  --train data/ghost_train.jsonl \
  --val data/ghost_val.jsonl \
  --eval-index data/eval_images.jsonl
```

等价的 Python 命令是：

```bash
python -m cold_ghost.cli sweep \
  --kind train \
  --group ghost \
  --model all \
  --tier all \
  --train data/ghost_train.jsonl \
  --val data/ghost_val.jsonl \
  --eval-index data/eval_images.jsonl
```

它会训练 12 个 family：

```text
2 models × 3 budget tiers × 2 routing schedule families = 12
```

每个 checkpoint 固定执行：

```text
Warm-up : 1000 optimizer updates, lr=1e-4
Rollout : 2000 optimizer updates, lr=3e-5
Accumulation : 8 microbatches/update
Validation : rollout 每 100 updates + 最后一步
Backbone : frozen
Trainable : Ghost modules only
```

默认 checkpoint 路径由配置文件定义，例如：

```text
checkpoints/cold_ghost/llava15/avg64/ours_vs_fastv/best.pt
```

每个 family 目录至少会产生：

```text
warmup.pt
last.pt
best.pt
training.jsonl
training_summary.json
```

`best.pt` 是正式评测默认读取的 checkpoint。

## 6.1 分模型训练

只训练 LLaVA 全部 6 个 family：

```bash
bash scripts/ghost/train_all.sh \
  --model llava15 \
  --tier all \
  --train data/ghost_train.jsonl \
  --val data/ghost_val.jsonl \
  --eval-index data/eval_images.jsonl
```

只训练 Qwen2.5-VL：

```bash
bash scripts/ghost/train_all.sh \
  --model qwen25vl \
  --tier all \
  --train data/ghost_train.jsonl \
  --val data/ghost_val.jsonl \
  --eval-index data/eval_images.jsonl
```

## 6.2 分预算训练

例如只训练 avg64 的 4 个 family：

```bash
bash scripts/ghost/train_all.sh \
  --model all \
  --tier avg64 \
  --train data/ghost_train.jsonl \
  --val data/ghost_val.jsonl \
  --eval-index data/eval_images.jsonl
```

## 6.3 训练中断恢复

`sweep train` 本身不会猜测你要从哪个 checkpoint 恢复。单个 family 恢复时使用：

```bash
python -m cold_ghost.cli train \
  --config experiment/ghost/llava15/avg64/llava15_ghost_ours_vs_fastv_avg64 \
  --train data/ghost_train.jsonl \
  --val data/ghost_val.jsonl \
  --eval-index data/eval_images.jsonl \
  --resume checkpoints/cold_ghost/llava15/avg64/ours_vs_fastv/last.pt
```

必须用同一份 train/val/eval-index；训练代码会检查数据 identity，不允许静默换数据后继续。

---

# 7. 正式主表：38 个原配置 × 12 个任务 = 456 runs

```bash
python -m cold_ghost.cli sweep \
  --kind eval \
  --group original \
  --model all \
  --tier all \
  --tasks all \
  --out experiments/cold_ghost_full/main
```

**不要加 `--limit`。**

这一条完整执行：

```text
POPE
GQA
MMBench
MME
RefCOCO val/testA/testB
RefCOCO+ val/testA/testB
RefCOCOg val/test
```

总计 456 个原 baseline 子运行。

---

# 8. 正式主表：24 个 Ghost 配置 × 12 个任务 = 288 runs

确认 12 个 family 的 `best.pt` 都已存在后：

```bash
python -m cold_ghost.cli sweep \
  --kind eval \
  --group ghost \
  --model all \
  --tier all \
  --tasks all \
  --out experiments/cold_ghost_full/main
```

这会执行 compact + stagewise 两种执行路径；同 family 的两条路径共享已经训练好的 `best.pt`。

总计：

```text
456 original + 288 ghost = 744 formal benchmark runs
```

输出结构形如：

```text
experiments/cold_ghost_full/main/eval/<config-tag>/<task>/
```

每个完成的正式评测目录应包含 `results.json` 和 run manifest 等评测产物。

---

# 9. 如何提前生成 744 条完整命令清单

原 baseline 456 条：

```bash
mkdir -p experiments/cold_ghost_full/commands
python -m cold_ghost.cli sweep \
  --kind eval \
  --group original \
  --model all \
  --tier all \
  --tasks all \
  --out experiments/cold_ghost_full/main \
  --dry-run \
  > experiments/cold_ghost_full/commands/eval_original_456.txt
```

Ghost 288 条：

```bash
python -m cold_ghost.cli sweep \
  --kind eval \
  --group ghost \
  --model all \
  --tier all \
  --tasks all \
  --out experiments/cold_ghost_full/main \
  --dry-run \
  > experiments/cold_ghost_full/commands/eval_ghost_288.txt
```

这里 `--dry-run` **只用于生成命令清单，不是实验运行**。真正正式运行仍用第 7、8 节不带 `--dry-run` 的命令。

---

# 10. 全量 Prefill FLOPs / KV profile：62 configs

```bash
python -m cold_ghost.cli sweep \
  --kind profile \
  --group all \
  --model all \
  --tier all \
  --out experiments/cold_ghost_full/main
```

总配置数：

```text
38 original + 24 ghost = 62
```

默认 profile 是正式一次 measured pass、零 warm-up；这是 executed-op 计数，不是生成速度测试。

输出：

```text
experiments/cold_ghost_full/main/profile/*.json
```

注意新增 profile 的 FLOPs backend 与原 DeepSpeed optional profiler 不是同一计数后端。论文比较时 baseline 与 Ghost 必须使用这一条同一入口重新统计。

---

# 11. 全量 CUDA Events runtime：62 configs

```bash
python -m cold_ghost.cli sweep \
  --kind runtime \
  --group all \
  --model all \
  --tier all \
  --out experiments/cold_ghost_full/main
```

`sweep runtime` 使用正式默认：

```text
warmup = 2
measured passes = 5
decode tokens = 64
```

如果论文需要更多重复次数，必须对所有比较方法统一修改。例如单配置 20 次：

```bash
python -m cold_ghost.cli runtime \
  --config experiment/ghost/llava15/avg64/llava15_ghost_ours_vs_fastv_avg64 \
  --out experiments/cold_ghost_full/runtime20/llava15_ghost_ours_vs_fastv_avg64.json \
  --warmup 5 \
  --passes 20 \
  --decode-tokens 64
```

不要只对 Ghost 增加 repetitions 而 baseline 不变。

---

# 12. 全量 Ghost diagnostics：12 family

运行完整 validation manifest，不传 `--limit`：

```bash
MANIFEST=data/ghost_val.jsonl \
OUT_ROOT=experiments/cold_ghost_full/diagnostics \
bash scripts/ghost/run_full_diagnostics.sh
```

脚本会对 12 个非-stagewise训练 family 分别执行：

```bash
python -m cold_ghost.cli diagnose \
  --config <ghost-training-family> \
  --manifest data/ghost_val.jsonl \
  --out <output.json>
```

用于输出 reactivation freshness、residual span 等分析指标。

---

# 13. 全量消融：240 runs

执行：

```bash
OUT_ROOT=experiments/cold_ghost_full/ablation \
bash scripts/ghost/run_full_ablation.sh
```

脚本固定比较：

```text
full
self_only
context_only
no_fresh
uniform_ghost
```

任务固定为代表性 4 项：

```text
gqa
mmbench
refcoco_testA
refcoco_testB
```

因此：

```text
12 family × 5 ablations × 4 tasks = 240 formal ablation runs
```

这里没有 `--limit`。

---

# 14. 汇总正式结果

主 benchmark/profile/runtime：

```bash
python -m cold_ghost.cli collect \
  --root experiments/cold_ghost_full/main \
  --out experiments/cold_ghost_full/main_summary.csv
```

消融脚本结束后会自动生成：

```text
experiments/cold_ghost_full/ablation/summary.csv
```

`collect` 不会把不兼容的 metrics 或不同 FLOPs backend 强行平均成一个数字；它保留 config、artifact、checkpoint hash 等字段供后续论文表格处理。

---

# 15. 一条命令完整执行整个项目

当且仅当以下条件已经满足：

1. conda 环境安装完成；
2. lmms-eval v0.7.1 已安装并打原项目 patch；
3. GPU/CUDA 可用；
4. `SOURCE_JSONL` 是独立训练数据；
5. 服务器有足够时间、磁盘和模型/数据下载权限；

可以执行：

```bash
export SOURCE_JSONL="$PWD/data/ghost_source.jsonl"
export OUT_ROOT="$PWD/experiments/cold_ghost_full"
PHASE=all bash scripts/ghost/run_full_project.sh
```

它依次运行：

```text
prepare
  -> complete eval-image index
  -> deterministic train/val split
  -> leakage audit
train
  -> 12 formal Ghost checkpoints
baseline evaluation
  -> 456 runs
Ghost evaluation
  -> 288 runs
efficiency
  -> 62 profile + 62 runtime
diagnostics
  -> 12 full-validation analyses
ablation
  -> 240 runs
collect
  -> main_summary.csv
```

这是一条真正的全量 pipeline，不是 smoke test。

---

# 16. 更推荐的服务器运行方式：按 phase 分开跑

因为完整实验非常长，实际科研服务器建议按阶段运行，并使用 `tmux`。

创建会话：

```bash
tmux new -s coldghost
```

进入项目环境后，依次执行。

## Phase A：数据准备

```bash
export SOURCE_JSONL="$PWD/data/ghost_source.jsonl"
PHASE=prepare bash scripts/ghost/run_full_project.sh
```

## Phase B：全部 12 个正式 checkpoint

```bash
PHASE=train bash scripts/ghost/run_full_project.sh
```

## Phase C：456 个原 baseline

```bash
PHASE=eval-baseline bash scripts/ghost/run_full_project.sh
```

## Phase D：288 个 Ghost benchmark

```bash
PHASE=eval-ghost bash scripts/ghost/run_full_project.sh
```

## Phase E：全部效率实验

```bash
PHASE=efficiency bash scripts/ghost/run_full_project.sh
```

## Phase F：全部诊断

```bash
PHASE=diagnostics bash scripts/ghost/run_full_project.sh
```

## Phase G：全部消融

```bash
PHASE=ablation bash scripts/ghost/run_full_project.sh
```

## Phase H：汇总

```bash
PHASE=collect bash scripts/ghost/run_full_project.sh
```

离开 tmux 但保持任务运行：

```text
Ctrl-b d
```

回来：

```bash
tmux attach -t coldghost
```

---

# 17. 分 GPU 手工并行：按模型运行

当前单个 Python 进程仍是单 GPU；如果服务器有多张独立 GPU，可以在不同 shell/tmux 中各跑一个模型，不要把一个模型拆到多 GPU。

GPU 0 跑 LLaVA baseline：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cold_ghost.cli sweep \
  --kind eval --group original --model llava15 --tier all --tasks all \
  --out experiments/cold_ghost_full/main
```

GPU 1 跑 Qwen baseline：

```bash
CUDA_VISIBLE_DEVICES=1 python -m cold_ghost.cli sweep \
  --kind eval --group original --model qwen25vl --tier all --tasks all \
  --out experiments/cold_ghost_full/main
```

Ghost 也同样：

```bash
CUDA_VISIBLE_DEVICES=0 python -m cold_ghost.cli sweep \
  --kind eval --group ghost --model llava15 --tier all --tasks all \
  --out experiments/cold_ghost_full/main
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m cold_ghost.cli sweep \
  --kind eval --group ghost --model qwen25vl --tier all --tasks all \
  --out experiments/cold_ghost_full/main
```

不同进程必须写不同或天然不冲突的 config/task output 路径；不要让两个进程同时训练同一个 checkpoint family。

---

# 18. 单个正式评测命令模板

如果某个 sweep 中某项失败，需要单独重跑，格式如下。

原方法：

```bash
python -m cold_ghost.cli eval \
  --config <experiment/original/...> \
  --task gqa \
  --out experiments/cold_ghost_full/main/eval/manual_original_gqa
```

Ghost：

```bash
python -m cold_ghost.cli eval \
  --config experiment/ghost/llava15/avg64/llava15_ghost_ours_vs_fastv_avg64 \
  --task gqa \
  --out experiments/cold_ghost_full/main/eval/manual_ghost_gqa
```

正式重跑时依然不要加 `--limit`。

---

# 19. 正式任务 selector

全部任务：

```bash
--tasks all
```

只跑普通 VQA/综合任务：

```bash
--tasks vqa
```

对应：

```text
gqa, mmbench, mme
```

只跑 grounding：

```bash
--tasks grounding
```

只跑 POPE：

```bash
--tasks pope
```

论文主表正式复现应使用 `--tasks all`。

---

# 20. 输出目录建议

推荐保持：

```text
experiments/cold_ghost_full/
├── commands/
│   ├── train_12.txt
│   ├── eval_original_456.txt
│   ├── eval_ghost_288.txt
│   ├── profile_62.txt
│   └── runtime_62.txt
├── main/
│   ├── eval/
│   ├── profile/
│   └── runtime/
├── diagnostics/
├── ablation/
├── main_summary.csv
└── ablation/summary.csv
```

`run_full_project.sh` 每次调用都会先刷新命令 manifest，因此你可以随时查看“理论上将运行哪些具体命令”。

---

# 21. 完整运行前必须检查的 checkpoint

训练结束后：

```bash
find checkpoints/cold_ghost -name best.pt -type f | sort
```

正式预期应该有 12 份 family checkpoint。

统计数量：

```bash
find checkpoints/cold_ghost -name best.pt -type f | wc -l
```

应为：

```text
12
```

如果少于 12，不要直接运行 `--group ghost --model all --tier all` 的完整正式评测；先修复缺失训练 family。

---

# 22. 完整评测后检查产物数量

主 benchmark 理论子运行数：744。

可以先统计 `results.json`：

```bash
find experiments/cold_ghost_full/main/eval -name results.json -type f | wc -l
```

如果所有配置/task 都一一产生一个 results 文件，预期为：

```text
744
```

Profile：

```bash
find experiments/cold_ghost_full/main/profile -name '*.json' -type f | wc -l
```

预期 62。

Runtime：

```bash
find experiments/cold_ghost_full/main/runtime -name '*.json' -type f | wc -l
```

预期 62。

消融结果理论上 240 个 `results.json`：

```bash
find experiments/cold_ghost_full/ablation -name results.json -type f | wc -l
```

不要仅凭文件数量判断实验科学上正确；还要查看失败日志、checkpoint hash、task metric 和异常值。

---

# 23. 推荐保存服务器环境与 Git 身份

正式实验开始前：

```bash
mkdir -p experiments/cold_ghost_full/environment
python -m pip freeze > experiments/cold_ghost_full/environment/pip_freeze.txt
conda env export > experiments/cold_ghost_full/environment/conda_env.yml
git rev-parse HEAD > experiments/cold_ghost_full/environment/git_commit.txt
nvidia-smi > experiments/cold_ghost_full/environment/nvidia_smi.txt
```

这样论文结果可以对应到明确代码提交和软件环境。

---

# 24. 最终推荐的实际执行顺序

不要一上来把 `PHASE=all` 丢到服务器然后不看日志。正式复现实验建议：

```text
1. 安装固定环境
2. 准备 independent Ghost source JSONL
3. index-eval
4. split-data
5. audit-data
6. train 12 checkpoint families
7. 检查 12 个 best.pt
8. 跑 456 original benchmark runs
9. 跑 288 Ghost benchmark runs
10. 跑 62 profile
11. 跑 62 runtime
12. 跑 12 diagnostics
13. 跑 240 ablations
14. collect
15. 核对结果数量与异常日志
16. 保存 git commit / conda / pip / nvidia-smi
```

对应的自动化脚本就是：

```bash
scripts/ghost/run_full_project.sh
scripts/ghost/run_full_diagnostics.sh
scripts/ghost/run_full_ablation.sh
```

以上三份脚本均设计为正式全量入口，不包含 `--limit` 或 `--debug-updates`。
