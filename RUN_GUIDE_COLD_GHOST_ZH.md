# mllm-reroute + Cold/Ghost：从零开始的远程服务器运行指南

> 本指南对应本仓库的增量实现，而不是 ALVTS。原始项目的 79 个文件保持字节一致；新增功能使用 `cold_ghost/` 和 `scripts/ghost/`。主方案只有一套确定配置。消融实验用于科研比较，不是主方法的可选实现。
>
> **交付的是源代码、训练/评测脚本和测试，不包含训练好的 Ghost 权重，也不包含独立训练图片。** 必须先准备独立图片—问题清单并完成训练，才能运行 Ghost 正式评测。原始 dense/FastV/PDrop/Reroute 不需要 Ghost 权重，可以先运行。
>
> 已完成的非 GPU 验证及其原始记录见 [docs/VALIDATION.md](docs/VALIDATION.md)。实际 7B 预训练权重、GPU 数值行为、显存峰值、全部数据集评测结果没有在交付时运行；CPU 测试通过不等于这些项目已经验证。

---

## 1. 项目实现了什么

原有 `models/router.py` 的 decision layers、最后序列位置对视觉区的 attention scoring、Full Top-K、keep ratios、`monotonic: false` 和阶段缓存均不改。`avg64/avg128/avg192` 是原实验预算档位，**不是每层固定保留相同数量的 Full token**。

新增实现按以下固定顺序执行：

1. 在原 decision layer 用原路由选 Full。只在剩余 skipped tokens 中预测下一 decision layer 重激活概率，选最多 128 个 Ghost；其余为 Cold。
2. Full 运行原 Transformer。Ghost 使用 Self Evolution、Active Residual Prototype Transport 和 gate 预测 residual；Cold 保持输入状态。
3. Ghost/Cold 分组在阶段内固定，但是 Ghost **每一 decoder layer 更新一次**；更新写入 deferred state，不加入原 Full attention，不写原 KV cache。
4. 下个 decision layer 的原 scorer 读取更新后的状态。因此 Full 选择规则不变，但具体入选的 token 索引可以改变。
5. 最后一次 decision layer 之后不再有重激活，剩余 skipped tokens 全为 Cold。自回归 decode 不新增 Ghost 更新。

固定参数：`K_G=128`、bottleneck=32、prototype 数=8、共享跨度=4 层、skip age 编码上限=8、预测范围=下一次 decision、epsilon=`1e-6`。原视觉编码器、projector 和 decoder 全部冻结；只有新增 Ghost 参数训练。

两条执行路径都实现了：`compact_route` 和 `compact_route_stagewise`。后者在阶段内部逐层更新 deferred buffer，不是只在决策层更新一次。

### 1.1 模型、配置与权重的数量

| 项目 | 本仓库采用的内容 |
|---|---|
| 模型 1 | `llava-hf/llava-1.5-7b-hf`，LLaVA-1.5，32 decoder layers |
| 模型 2 | `Qwen/Qwen2.5-VL-7B-Instruct`，28 decoder layers |
| 原配置 | 38 份：两个 dense，加两个模型 × 三档 × 六种原路由/执行配置 |
| 新 Ghost 配置 | 24 份：两个模型 × 三档 × 两个 Reroute 调度族 × 两条执行路径 |
| 需要训练的 Ghost 权重 | 12 份：两个模型 × 三档 × 两个调度族 |
| 共享范围 | 同一个模型、档位和调度族的 compact/stagewise 共用同一权重 |

Qwen 保留原 `max_pixels=451584`。这是原视觉预算上限，不代表任意图片都会产生恰好 576 个视觉 token。不要为了省显存擅自调小它后仍把结果标成原协议。

### 1.2 原代码不改，怎样接入

`cold_ghost/integration.py` 为当前 Python 进程中的模型实例安装增量包装，直接调用原 scorer/router。评测通过临时替换原评测入口的 patch 函数接入；退出时恢复函数引用。原磁盘文件没有被改写。

原 `test.py` 是上传包内未完成的 GQA 草稿，自带语法错误，并非项目训练/评测入口。保留原件；可运行修正版在 `cold_ghost/examples/gqa_fixed.py`。`verify` 会明确报告这个已知问题，不会把它隐藏成“全仓库没有语法错误”。

---

## 2. 从 Windows 连接远程 Linux 服务器

以下安装路线固定为 **Linux x86_64 + NVIDIA GPU + Python 3.10 + CUDA 12.8 PyTorch wheel + conda**。不适用于 Windows 原生训练、ARM 服务器、CPU offload、跨 GPU 拆分 decoder 或视频/多图输入。

在 Windows PowerShell 执行：

```powershell
ssh YOUR_USER@YOUR_SERVER
```

把 `YOUR_USER` 换成服务器用户名，把 `YOUR_SERVER` 换成地址；这行登录服务器。后续 bash 命令在登录后的 Linux 终端执行，而不是在本地 PowerShell 执行。

登录后检查：

```bash
uname -m
nvidia-smi
free -h
df -h "$HOME"
```

`uname -m` 检查架构，下面安装器要求输出 `x86_64`；`nvidia-smi` 查看实际显卡和驱动；`free -h` 查看主存；`df -h` 查看主目录磁盘。GPU 驱动须能运行 CUDA 12.8 对应 wheel，Qwen 的默认计算 dtype 为 BF16，还要支持 BF16。显存需求与图片、问题长度、训练阶段有关，本交付没有实测一个可以保证所有任务通过的最低显存数。

本实现训练使用同一个冻结骨干先做 dense teacher 再做 student，并不是同时加载两份 7B 权重；但训练会保留 student 反向图，仍可能比只做推理占用更多显存。主机端也要有空间保存当前样本的 teacher 状态。

---

## 3. 安装 conda，并创建项目环境

### 3.1 首次安装 Miniforge

Miniforge 提供 conda；这里固定使用官方 `26.7.2-0` 安装器。已安装可用 conda 的机器不重复安装，只需让当前 shell 能执行 `conda`，然后进入第 3.2 节。

```bash
mkdir -p "$HOME/installers"
cd "$HOME/installers"
export MF_VERSION=26.7.2-0
export MF_FILE="Miniforge3-${MF_VERSION}-Linux-x86_64.sh"
curl -fL -o "$MF_FILE" "https://github.com/conda-forge/miniforge/releases/download/${MF_VERSION}/${MF_FILE}"
curl -fL -o "${MF_FILE}.sha256" "https://github.com/conda-forge/miniforge/releases/download/${MF_VERSION}/${MF_FILE}.sha256"
sha256sum -c "${MF_FILE}.sha256"
bash "$MF_FILE" -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda init bash
conda --version
```

逐行作用：创建下载目录；进入目录；固定版本号；构造安装器文件名；下载安装器；下载官方 SHA256；核对下载内容，**不是 OK 就停止**；安装到个人主目录且不需要 sudo；让当前 shell 立即识别 conda；配置后续 bash 登录；显示 conda 版本。`curl -fL` 会跟随跳转并在 HTTP 错误时失败，不会把错误页面当安装器。

### 3.2 获取仓库并创建环境

先用 base 环境安装 Git，以避免新服务器没有 Git：

```bash
conda install -n base -c conda-forge git -y
mkdir -p "$HOME/projects"
cd "$HOME/projects"
git clone https://github.com/rita2126top-alt/mllm_reroute.git
cd mllm_reroute
conda env create -f environment.cold_ghost.yml
conda activate reroute_coldghost
python --version
which python
```

第一行安装下载代码所需 Git；第二至五行创建工作目录、下载仓库并进入项目；第六行按仓库环境文件创建独立环境；第七行激活；最后两行确认 Python 3.10 以及 Python 路径位于 `reroute_coldghost` 环境。

环境文件安装 Python、pip、Git、patch 等基础工具。PyTorch 等 Python 依赖在下一节安装。不要对原 `requirements.txt` 做手工改版本。

### 3.3 安装原项目依赖和 lmms-eval

所有命令在仓库根目录运行：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-cold-ghost.txt
mkdir -p external
git clone --branch v0.7.1 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git external/lmms-eval
python -m pip install -c requirements.txt -e external/lmms-eval
bash lmms_eval_patches/apply.sh
python scripts/ghost/check_dependencies.py
```

第一行升级当前环境的 pip；第二行安装原发布依赖及 pytest，包含 `torch==2.11.0+cu128`、`torchvision==0.26.0+cu128`、`transformers==5.4.0`；第三、四行把固定版本评测框架放在 `external/`；第五行可编辑安装，并用原版本清单约束依赖，避免评测框架静默升级 Transformers；第六行应用原项目已有的三份 RefCOCO utilities 补丁；最后检查依赖冲突。该检查不静默忽略错误；唯一特例是原 Decord 0.6.0 的 Linux wheel 平台标记告警，必须实际生成并解码两帧 CPU 视频验证通过才继续，告警原文与 WHEEL 元数据都会输出。其他 pip check 错误均终止。

`external/lmms-eval` 是第三方 checkout，不属于那 79 个冻结的原始文件。原来的补丁脚本会修改这个外部 checkout，这是原评测协议要求，并非改写项目原代码。

**不能把安装失败当成功继续执行。** 完整依赖环境的 CPU 验证记录见 `docs/VALIDATION.md`。安装源不可访问或版本冲突时先修复环境，不能随意换成最新版后直接比较论文结果。

本主流程不安装 DeepSpeed：新增 FLOPs 入口使用 PyTorch 自带计数器，时延仍复用原 CUDA Events 协议。旧版 DeepSpeed 可选脚本的限制在第 15 节单独说明。

### 3.4 每次重新登录要执行的命令

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate reroute_coldghost
cd "$HOME/projects/mllm_reroute"
export CUDA_VISIBLE_DEVICES=0
export CONDA_ENV=reroute_coldghost
unset BBOX_COORD_FORMAT REFCOCO_PROMPT_STYLE
```

前两行恢复 conda；第三行回到仓库；第四行只暴露服务器的物理 GPU 0，程序内使用 `cuda:0`；第五行供原 shell 脚本调用正确 conda 环境；最后清除上次实验遗留的 bbox/prompt 覆盖值，使评测按照原模型默认协议设置。

主流程一次只使用一张可容纳模型的 GPU。不要用 `accelerate launch`/DDP 启动当前训练入口，也不要用 `device_map=auto` 将 decoder 分到多张卡后期待本增量模块自动支持。

---

## 4. 设置缓存、目录和检查环境

```bash
export HF_HOME="$HOME/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
mkdir -p "$HF_HOME" data checkpoints experiments/cold_ghost
python -m cold_ghost.cli doctor --require-hf --require-lmms
python -m cold_ghost.cli doctor --require-hf --require-lmms --gpu
```

`HF_HOME` 控制模型和数据缓存根目录，可以在首次下载前改到服务器分配的大容量磁盘；不要将私密 token 写进脚本或仓库。第二行避免 tokenizer 并发告警；第三行关闭不需要的远程实验记录；第四行创建工作目录。第一次 doctor 只检查代码和依赖；第二次由你在服务器执行，额外检查 CUDA 并运行一个极小矩阵运算。交付测试没有执行这个 `--gpu` 检查。

doctor 必须返回 `ok: true`。Qwen 运行前还应确认输出的 `bf16_supported` 为 true。`torch.cuda.is_available()` 失败不是 Ghost 算法结果，而是服务器环境尚未满足要求。

```bash
python -m cold_ghost.cli verify
CUDA_VISIBLE_DEVICES="" bash scripts/ghost/test_cpu.sh
python -m cold_ghost.cli --help
```

第一行核对原文件 SHA256 并检查 Python/bash 语法；第二行在安装好的环境执行所有 CPU 单元和小模型测试，临时隐藏 GPU，不会下载 7B 权重；第三行显示全部 CLI 功能。已知旧 `test.py` 错误会单独列出；新增运行入口不应出现语法错误。

保存实际环境以便复查：

```bash
python -m pip freeze > experiments/cold_ghost/installed_packages.txt
conda env export > experiments/cold_ghost/conda_environment_resolved.yml
git rev-parse HEAD > experiments/cold_ghost/source_commit.txt
```

依次记录真实 pip 依赖、conda 环境和源码提交。不要把未验证的环境文件或实验成绩写成服务器已经运行成功。

---

## 5. 下载两个原始模型

程序会自动下载缺失模型。先明确下载可以把网络问题和算法问题分开：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('llava-hf/llava-1.5-7b-hf'))
print(snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct'))
PY
```

这段命令在同一个 Python 进程中按原配置模型 ID 下载文件到 HF 缓存，并打印本地 snapshot 路径。不会训练模型，也不会把模型权重上传 GitHub。下载需要足够磁盘和网络；某个资源要求认证时，先使用当前安装版本的 `hf auth login` 在服务器交互式登录，不把 token 放进命令历史里的共享脚本。

不要在文件未缓存齐全前设置 `HF_HUB_OFFLINE=1`。模型 ID 相同也可能解析到不同 revision，正式实验应保留上面打印的 snapshot 路径和实际下载信息。

---

## 6. 先跑原始 baseline，确认评测环境正常

先定义后续会复用的配置路径：

```bash
export BASE_LLAVA=experiment/baseline/llava15
export BASE_QWEN=experiment/baseline/qwen25vl
export CG_LLAVA=experiment/ghost/llava15/avg64/llava15_ghost_ours_vs_fastv_avg64
export CG_QWEN=experiment/ghost/qwen25vl/avg64/qwen25vl_ghost_ours_vs_fastv_avg64
export RR_LLAVA=experiment/llava15/avg64/llava15_ours_vs_fastv_avg64
```

它们只是减少重复输入的 shell 变量，对应 `configs/` 下的配置路径，不加 `.yaml`。`BASE` 为未压缩原模型；`RR` 为原 Reroute；`CG` 为固定 Cold/Ghost 方案。

```bash
python -m cold_ghost.cli eval --config "$BASE_LLAVA" --task pope --limit 1 --out experiments/cold_ghost/smoke_baseline_llava
python -m cold_ghost.cli eval --config "$BASE_QWEN" --task pope --limit 1 --out experiments/cold_ghost/smoke_baseline_qwen
```

两行分别加载两个原模型，通过原 lmms-eval 执行 POPE 的极小样本检查；`--limit 1` 只为检查数据下载、模型加载和结果输出，**不是正式评测分数**；`--out` 把结果隔离保存，防止覆盖不同实验。

原数据集由固定版本 lmms-eval 的原 task 定义加载，不要求手动改 task YAML。网络、授权、上游数据不可用时会报出真实失败；交付没有代为下载全部评测集。

重复同一实验时使用新的 `--out`，不要直接覆盖已有 `results.json`。出现 `failure.json` 时先读其错误和终端 traceback，不把失败运行计入结果。

---

## 7. 准备独立 Ghost 训练数据：不能使用评测图片替代

### 7.1 必须提供的输入

确定方案只指定了独立图片—问题清单，**没有指定或附带可直接下载的训练数据集**。因此本仓库不虚构一个数据来源。你需要准备自己有权使用的图片与问题；不需要人工答案，监督来自冻结 dense teacher 和路由轨迹。

约定目录示例：

```text
data/
  independent_images/
    image_000001.jpg
    image_000002.jpg
    ...
  questions.csv
  source.jsonl
```

`questions.csv` 必须有表头：

```csv
sample_id,image_id,image_path,question
q000001,independent_000001,independent_images/image_000001.jpg,What is in the image?
q000002,independent_000002,independent_images/image_000002.jpg,Describe the main object.
```

以上两行仅展示格式，不是随仓库提供的训练样本。`image_path` 相对最终 JSONL 所在目录解析；同一图片的多个问题应使用同一个 `image_id`、不同 `sample_id`。至少需要两个独立图片组才能分出训练和验证；极少图片只能检查管线，不能据此声称科研性能。

从你实际准备的 CSV 转成清单：

```bash
python - <<'PY'
import csv, json
from pathlib import Path
source = Path('data/questions.csv')
target = Path('data/source.jsonl')
required = ['sample_id', 'image_id', 'image_path', 'question']
if target.exists():
    raise FileExistsError(target)
with source.open(encoding='utf-8-sig', newline='') as src, target.open('x', encoding='utf-8') as dst:
    reader = csv.DictReader(src)
    if not set(required).issubset(reader.fieldnames or []):
        raise ValueError(f'CSV must contain {required}')
    count = 0
    for row in reader:
        item = {k: row[k] for k in required}
        if any(not str(value).strip() for value in item.values()):
            raise ValueError(f'Empty field in row {count + 2}')
        dst.write(json.dumps(item, ensure_ascii=False) + '\n')
        count += 1
    if count == 0:
        raise ValueError('No samples')
print(f'Wrote {count} records to {target}')
PY
```

此段逐行读取 CSV，检查字段，写入 UTF-8 JSONL；不会生成假图片、自动复制 benchmark 或产生训练答案。实际图片存在性与内容校验由后续命令完成。

### 7.2 按图片分训练/验证

```bash
python -m cold_ghost.cli split-data --source data/source.jsonl --train data/ghost_train.jsonl --val data/ghost_val.jsonl --val-fraction 0.1
```

它先解析图片路径并计算 EXIF 归一化 RGB 内容哈希，再按 image_id 或完全相同图片内容分组，以 seed=42 确定性划分。`0.1` 是验证图片组比例，不是简单按问题行数取 10%。同一图的不同问题不会跨两个集合。已存在清单会拒绝覆盖。

### 7.3 为全部原评测集建立排除索引

```bash
python -m cold_ghost.cli index-eval --out data/eval_images.jsonl
```

该命令通过原 lmms-eval task 定义枚举 POPE、GQA、MMBench、MME 和八个 RefCOCO/+/g split 的评测图片，只提取图片内容和可用图片 ID；不使用测试答案训练。此步骤可能触发评测数据下载和 CPU 图片解码，但不会加载 7B 模型或使用 GPU。

只有所有任务成功结束，才会生成完整索引及 `data/eval_images.meta.json`。中断时的 `.partial` 不是合法索引，不能手工把它标为 complete 来绕过数据隔离。

```bash
python -m cold_ghost.cli audit-data --train data/ghost_train.jsonl --val data/ghost_val.jsonl --eval-index data/eval_images.jsonl
```

它检查训练/验证互斥、与全部评测图片 ID/内容互斥，并额外排除原 `bench_data/` 三张效率测量图片。检查通过应有 `image_overlap: false`。内容哈希检查识别完全相同的规范化像素，**不是近重复/裁剪变体检测**；数据来源本身仍需遵守独立性和使用权限。

清单或图片缺失、评测索引不完整、存在重叠都会在训练前报错，不会静默降级为使用测试集。没有独立数据时可以继续运行原始方法和 CPU 测试，但不能跳过这一步伪造可正式评测的 Ghost 权重。

---

## 8. 训练第一个 Ghost checkpoint

### 8.1 正式训练命令

```bash
python scripts/train_ghost.py --config "$CG_LLAVA" --train data/ghost_train.jsonl --val data/ghost_val.jsonl --eval-index data/eval_images.jsonl
```

本入口与 `python -m cold_ghost.cli train` 等价。配置决定骨干、原路由调度和 Ghost 参数。默认输出为：

```text
checkpoints/cold_ghost/llava15/avg64/ours_vs_fastv/
  training.jsonl
  warmup.pt
  last.pt
  best.pt
  training_summary.json
```

训练流程固定如下，不需要手动运行第二阶段：

| 设置 | 数值与含义 |
|---|---|
| Warm-up | 1000 个 optimizer updates，学习率 `1e-4` |
| Rollout | 2000 个 optimizer updates，学习率 `3e-5` |
| microbatch / 梯度累积 | 1 / 8，每 8 个样本完成一次 optimizer update |
| 数据顺序 | seed=42，遍历结束继续下一轮，直到达到 update 数 |
| optimizer | AdamW，betas `(0.9,0.999)`，epsilon `1e-8` |
| weight decay | 矩阵 `0.01`，bias 和 prototype queries 不衰减 |
| clipping | 梯度范数上限 1.0 |
| 损失 | Residual、Direction、Reactivation、Freshness，权重 `1,0.1,0.1,1` |
| 验证 | 每 100 次 rollout updates，以及最后一次 update |
| 权重选择 | 独立 Ghost 验证集最小总损失；并列选更早 update |

Warm-up 在非最后阶段的全部 skipped candidates 上训练 predictor，但不把预测状态写回原 identity 路由；rollout 真正执行 Ghost 状态更新和下次 Full 重选，Freshness Loss 可以沿状态轨迹反传。优化器在阶段转换时重置，Ghost 参数保留。

新增参数以 FP32 保存和优化，投影运算按骨干 dtype 执行。LLaVA FP16 使用 loss scaling；如果出现由 loss scale 引起的非有限梯度，训练降低 scale 并重算同一组累积样本，不把失败步骤算成已完成 update。真正的非有限前向损失仍会立即报错，不通过跳样本掩盖。

骨干保持 eval，参数冻结，但不能把整个 student forward 包进 `torch.no_grad()`，否则 Freshness Loss 无法训练早期 Ghost。训练 `use_cache=False`，关闭有状态路由不支持的 gradient checkpointing。

### 8.2 查看训练进度和结果

在另一个已登录的终端执行：

```bash
tail -n 3 checkpoints/cold_ghost/llava15/avg64/ours_vs_fastv/training.jsonl
cat checkpoints/cold_ghost/llava15/avg64/ours_vs_fastv/training_summary.json
```

第一行查看最近记录的阶段、update、四项损失和梯度范数；第二行在训练完成后查看选出的 checkpoint 和数据审计摘要。summary 尚未生成不等于训练出错，可先看终端及日志。

`best.pt` 是用于评测的 adapter；`last.pt` 是带 optimizer/RNG 的断点；`warmup.pt` 是第一阶段完成快照。**不是把所有骨干参数重新保存一份**。

### 8.3 中断后恢复

```bash
python scripts/train_ghost.py --config "$CG_LLAVA" --train data/ghost_train.jsonl --val data/ghost_val.jsonl --eval-index data/eval_images.jsonl --resume checkpoints/cold_ghost/llava15/avg64/ours_vs_fastv/last.pt
```

除增加 `--resume` 外，配置和数据保持原样。会检查 checkpoint 中的模型、维度、路由调度、Ghost 配置、数据哈希，并恢复 optimizer 和随机状态。每 100 updates/阶段结束保存断点，所以意外中断后可能需要重算最近未保存的 updates。

不能用 `best.pt` 代替带 optimizer 的恢复断点，也不能把另一档位、另一模型、另一调度族的权重混用。

### 8.4 只检查服务器训练链路的 debug 命令

```bash
python scripts/train_ghost.py --config "$CG_LLAVA" --train data/ghost_train.jsonl --val data/ghost_val.jsonl --eval-index data/eval_images.jsonl --debug-updates 2 --out checkpoints/debug_llava
```

这行在你自己的服务器做各两次 update 的工程检查，单独写入 debug 目录。debug 权重明确带标记，并被正式 Ghost 评测/推理加载器拒绝。它不是替代上述 1000+2000 updates 的主实验；完成 debug 后按第 8.1 节正式训练。

---

## 9. 用训练好的 Ghost 推理和评测

### 9.1 一张图片生成答案

```bash
python -m cold_ghost.cli infer --config "$CG_LLAVA" --image data/independent_images/image_000001.jpg --question "What is in the image?" --max-new-tokens 64 --out experiments/cold_ghost/example_answer.json
```

将图片路径换成实际文件；`--question` 是输入问题；`--max-new-tokens` 限制新增答案长度；默认 greedy、`use_cache=True`。从配置路径读取 `best.pt`，不会用随机 Ghost 参数替代缺失权重。控制台和 JSON 都会给出答案。

### 9.2 POPE 冒烟与正式评测分开

```bash
python -m cold_ghost.cli eval --config "$CG_LLAVA" --task pope --limit 1 --out experiments/cold_ghost/smoke_ghost_llava
python -m cold_ghost.cli eval --config "$CG_LLAVA" --task pope --out experiments/cold_ghost/eval_ghost_llava_pope
```

第一行只检验链路；第二行没有 `--limit`，才使用原 task 的完整评测 split。相同 run 的配置、权重指纹、结果分开保存。比较原 Reroute：

```bash
python -m cold_ghost.cli eval --config "$RR_LLAVA" --task pope --out experiments/cold_ghost/eval_original_reroute_llava_pope
```

这行使用原 identity skip，不需要 Ghost 权重。比较时保持相同模型、调度族、档位、任务和数据版本。

### 9.3 Grounding、VQA 及原主表任务

```bash
python -m cold_ghost.cli eval --config "$CG_LLAVA" --task refcoco_testA --out experiments/cold_ghost/eval_ghost_llava_refcoco_testA
python -m cold_ghost.cli eval --config "$CG_LLAVA" --task gqa --out experiments/cold_ghost/eval_ghost_llava_gqa
python -m cold_ghost.cli eval --config "$CG_LLAVA" --task paper_main --out experiments/cold_ghost/eval_ghost_llava_paper_main
```

三行依次执行一个 grounding split、GQA、以及原多任务主表配置。`paper_main` 只包含 POPE + 八个 grounding split，不包含 GQA/MMBench/MME；要覆盖所有功能使用第 11 节的 `--tasks all`。

LLaVA 继续用 `bbox=normalized,prompt=full`；Qwen 继续用 `bbox=pixel,prompt=nuwa`。不要为了让答案看起来正常手动混用两个模型的 RefCOCO prompt 或坐标解释。

### 9.4 Stagewise 使用同一权重

```bash
export CG_LLAVA_STAGEWISE=experiment/ghost/llava15/avg64/llava15_ghost_ours_vs_fastv_stagewise_avg64
python -m cold_ghost.cli eval --config "$CG_LLAVA_STAGEWISE" --task pope --out experiments/cold_ghost/eval_ghost_llava_stagewise_pope
```

第一行选择相应 stagewise 配置；第二行仍加载 `ours_vs_fastv/best.pt`。不要再训练一套 stagewise 参数来混淆算法与调度实现的差异。

自定义 checkpoint 位置时明确传参：

```bash
python -m cold_ghost.cli eval --config "$CG_LLAVA" --checkpoint checkpoints/cold_ghost/llava15/avg64/ours_vs_fastv/best.pt --task pope --out experiments/cold_ghost/eval_explicit_checkpoint
```

`--checkpoint` 只改变权重文件位置，不解除对模型/路由/Ghost 配置一致性的检查。

---

## 10. 训练全部 12 份权重

先看计划，不启动训练：

```bash
bash scripts/ghost/train_all.sh --group ghost --model all --tier all --dry-run
```

此命令只打印 12 条训练命令。stagewise 的重复权重训练被去除。

正式执行：

```bash
bash scripts/ghost/train_all.sh --group ghost --model all --tier all --train data/ghost_train.jsonl --val data/ghost_val.jsonl --eval-index data/eval_images.jsonl
```

会串行训练两个模型、三个档位、两个调度族，自动使用各自的输出目录。任何子进程失败会停止，而不是跳过后显示全部完成。默认没有多卡并行调度。

只训练 Qwen 的某个档位：

```bash
bash scripts/ghost/train_all.sh --group ghost --model qwen25vl --tier avg128 --dry-run
```

这行展示该组合的两个调度族命令。去掉 `--dry-run` 才执行，其他参数含义相同。训练中断后使用具体配置的第 8.3 节恢复命令；不要直接重跑总 sweep 企图覆盖已经存在的权重目录。

---

## 11. 覆盖所有原始实验和全部 Ghost 评测

### 11.1 列出实际配置

```bash
python -m cold_ghost.cli list --group original
python -m cold_ghost.cli list --group ghost
python -m cold_ghost.cli list --group all
```

依次列出 38、24、62 份配置。它读取真实配置文件，不生成假配置名称。

### 11.2 任务选择器

| `--tasks` | 包含任务 |
|---|---|
| `pope` | POPE |
| `vqa` | GQA、MMBench 英文 dev、MME |
| `refcoco` | RefCOCO val/testA/testB |
| `grounding` | RefCOCO、RefCOCO+、RefCOCOg 的八个 split |
| `ablation` | GQA、MMBench、RefCOCO testA/testB |
| `paper_main` | 一次加载中执行原 POPE + 八个 grounding split |
| `all` | POPE、三个 VQA-style 配置、八个 grounding 配置 |
| 逗号分隔 | 例如 `pope,gqa,refcoco_testA` |

单次 `eval` 用 `--task`，sweep 用复数 `--tasks`；不要写成 `eval --task all`。

### 11.3 先验证枚举，再正式跑

```bash
bash scripts/ghost/run_paper_table.sh --group all --model all --tier all --tasks all --dry-run > experiments/cold_ghost/evaluation_plan.txt
```

这行只把计划写入文件，不加载 GPU 模型。`62 configs × 12 task configs = 744` 个子运行，包含原 baseline/原方法与 Ghost 两条执行路径。它们不是 744 个不同数据集。

先跑一个范围内的所有任务冒烟：

```bash
bash scripts/ghost/run_paper_table.sh --group ghost --model llava15 --tier avg64 --tasks all --limit 1 --out experiments/cold_ghost/smoke_suite
```

这里有四份 Ghost 配置 × 十二任务，全部是小样本工程检查。需要该模型/档位的两个调度族权重都已训练。

完整 Ghost 评测：

```bash
bash scripts/ghost/run_paper_table.sh --group ghost --model all --tier all --tasks all --out experiments/cold_ghost/ghost_full
```

完整原方法评测：

```bash
bash scripts/ghost/run_paper_table.sh --group original --model all --tier all --tasks all --out experiments/cold_ghost/original_full
```

分别执行 288 与 456 个子运行，总计 744。必须去掉 `--limit`，并使用和 smoke 不同目录。默认同一 GPU 串行运行；程序报错会停止，已完成的独立目录保留。

---

## 12. Prefill FLOPs、KV cache、显存统计

```bash
python -m cold_ghost.cli profile --config "$RR_LLAVA" --passes 1 --warmup 0 --out experiments/cold_ghost/profile/original_reroute_llava.json
python -m cold_ghost.cli profile --config "$CG_LLAVA" --passes 1 --warmup 0 --out experiments/cold_ghost/profile/ghost_llava.json
```

两行用同样的原 `bench_data/manifest.json` 三样本 cohort、原预处理与 prefill 协议，分别测原 Reroute 和 Ghost。`passes=1,warmup=0` 沿用原 shell profiler 默认值，不是精确 latency 测量的设置。

统计包含：supported executed-op FLOPs、每层 KV sequence length、原 K+V token 计数、KV tensor bytes、CUDA peak allocated memory，以及 Ghost 参数/动态状态相关字段。**Ghost 的实际 forward 计算在计数范围内**，不会通过扣掉 Ghost 开销维持名义 iso-FLOPs。

重要的计数边界：新增入口使用 `torch.utils.flop_counter.FlopCounterMode`，multiply-add 按 2 FLOPs，统计其支持的矩阵/卷积/attention 算子。它不是逐条硬件指令计数，也不穷尽所有归一化、非线性和索引开销。它与旧 DeepSpeed 计数器不能不加说明地混在一张表；JSON 明确标记 `comparable_to_legacy_deepspeed_without_recount: false`。

**比较表中原方法和 Ghost 都用上面这个新入口重测，原方法算法与 token schedule 不变。** 单独保留旧 DeepSpeed 字段时要标注不同计数后端，不能用旧数减新数声称节省。

批量重测所有配置：

```bash
bash scripts/ghost/run_profile.sh --group all --model all --tier all --dry-run
bash scripts/ghost/run_profile.sh --group all --model all --tier all --out experiments/cold_ghost/efficiency
```

第一行只展示 62 条计划；第二行正式执行，结果在 `efficiency/profile/`。sweep 固定采用上述主协议；单独调整 passes/warmup 只在直接 `profile` 命令中使用，并在结果中保留设置。

---

## 13. Prefill / decode 时延测量

```bash
python -m cold_ghost.cli runtime --config "$RR_LLAVA" --passes 5 --warmup 2 --decode-tokens 64 --out experiments/cold_ghost/runtime/original_reroute_llava.json
python -m cold_ghost.cli runtime --config "$CG_LLAVA" --passes 5 --warmup 2 --decode-tokens 64 --out experiments/cold_ghost/runtime/ghost_llava.json
python -m cold_ghost.cli runtime --config "$CG_LLAVA_STAGEWISE" --passes 5 --warmup 2 --decode-tokens 64 --out experiments/cold_ghost/runtime/ghost_llava_stagewise.json
```

三行分别计时原 Reroute、Ghost compact、Ghost stagewise。新 runtime 入口直接调用原 `profiler/bench_runtime.py` 的 CUDA Events 计时循环，仅替换加载模型/安装增量模块的接入点；仍是三样本 cohort、两次 warm-up、五次测量、64 个 decode tokens。

不能从 CPU 冒烟测试时长推断 GPU 加速比，也不能只看理论 FLOPs 声称 Ghost 一定更快。需要测出新增计算带来的准确率、latency 和内存权衡。

```bash
bash scripts/ghost/run_runtime_bench.sh --group all --model all --tier all --out experiments/cold_ghost/efficiency
```

这行对全部 62 配置按原默认协议计时，结果在 `efficiency/runtime/`。正式测时尽量避免同一张卡同时运行其他工作，并记录显卡、驱动、软件环境；不要把不同硬件测出的 absolute latency 直接作方法对比。

---

## 14. Staleness 诊断、消融与结果汇总

### 14.1 诊断 skipped-state 问题

```bash
python -m cold_ghost.cli diagnose --config "$CG_LLAVA" --manifest data/ghost_val.jsonl --limit 32 --out experiments/cold_ghost/diagnostics/llava_avg64.json
```

它在独立 Ghost 验证数据上比较原 identity 路径与 Ghost 路径，记录重激活 token 的 age/误差、Residual 低秩结构等实际诊断信息。`limit=32` 限制样本数，不是原准确率 benchmark。输出用于检验假设，代码不会预设 Ghost 必须改善误差，也不会生成虚构的实验图表或成绩。

### 14.2 固定主方法之外的科研消融

主方法一直是 `ablation=full`。提供 `self_only`、`context_only`、`no_fresh`、`uniform_ghost` 供相同协议下的消融；必须各自重新训练，不是把 full checkpoint 的组件临时关掉后当作同等训练的对照。

```bash
for ARM in self_only context_only no_fresh uniform_ghost; do
  python scripts/train_ghost.py --config "$CG_LLAVA" --ablation "$ARM" --train data/ghost_train.jsonl --val data/ghost_val.jsonl --eval-index data/eval_images.jsonl || break
  python -m cold_ghost.cli eval --config "$CG_LLAVA" --ablation "$ARM" --task pope --out "experiments/cold_ghost/ablations/${ARM}/pope" || break
done
```

每轮先训练再评测当前消融，任何命令失败后停止循环。默认权重隔离在 `checkpoints/cold_ghost_ablations/<ARM>/...`，身份校验禁止混用 full 与消融 checkpoint。`uniform_ghost` 是固定的均匀位置选取对照，不是改变主方法的 react Top-K。

### 14.3 汇总真实结果

```bash
python -m cold_ghost.cli collect --root experiments/cold_ghost --out experiments/cold_ghost/summary.csv
```

它递归收集已产生的 `results.json` 和 profile/runtime 汇总，记录配置、原始指标、checkpoint 指纹、数据限制和 FLOPs 后端。CSV 不自动把不同指标算成一个无意义平均数，也不把 smoke/正式分数或不同 FLOPs 后端混算。

检查 CSV 的 `limit`、`kind`、`flops_backend` 和 `artifact` 列；`artifact` 可以追溯原始 JSON。正式表只纳入完整评测的行，并明确样本数量和测量硬件。

---

## 15. 原项目命令仍然存在，怎样使用

新扩展不替换原 README、不覆盖原 scripts。原入口运行的仍是原方法，不会因为目录里多了 `cold_ghost` 就自动启用 Ghost。

```bash
export CONDA_ENV=reroute_coldghost
bash scripts/run_setting.sh baseline/llava15 --tasks pope --limit 1 --gpu 0 --log-dir experiments/legacy_logs
bash scripts/run_setting.sh llava15/avg64/llava15_ours_vs_fastv_avg64 --tasks grounding --gpu 0 --log-dir experiments/legacy_logs
bash scripts/run_paper_table.sh all --tier all --tasks paper_main --gpu 0
bash scripts/run_runtime_bench.sh all --tier all --n-passes 5 --n-warmup-passes 2 --n-decode-tokens 64 --gpu 0
```

第一行让原脚本使用当前新环境；第二行原 dense POPE 冒烟；第三行原 Reroute grounding；第四行原 38 配置的原主表任务；第五行原 runtime sweep。这里的路径从 `configs/experiment/` 往下写，不像新 CLI 带 `experiment/` 前缀。

原 `run_setting.sh` 在一个任务失败后会打印 FAIL 并继续其他任务，这是原文件自身的行为，已保留。需要 fail-fast 和统一 manifest 的实验，使用本指南的新入口执行原配置。

原 DeepSpeed profiler 的历史入口为：

```bash
bash scripts/run_profile.sh all --tier all --n-passes 1 --n-warmup-passes 0 --gpu 0
```

这条是**原历史接口说明，不是基础环境安装完即可保证执行的命令**。它需要原 requirements 中注释为 OPTIONAL 的 `deepspeed==0.14.5` 及其 CUDA 编译条件；交付未验证这套旧可选依赖与 PyTorch 2.11/CUDA 的兼容性。不要直接执行原注释里的其他 CUDA index/旧 nvcc 建议然后认为配置已验证。当前完整效率功能使用第 12、13 节的新 profiler/runtime，原算法对照和 Ghost 都能经同一入口测量。

---

## 16. 新入口与文件定位速查

| 文件/命令 | 作用 |
|---|---|
| `cold_ghost/modules.py` | block-shared Self、Prototype、Context、Gate、Reactivation 参数 |
| `cold_ghost/integration.py` | 原 router/scorer 接入、Full/Ghost/Cold 状态和两条 compact 路径 |
| `cold_ghost/losses.py` | 四项监督、真实下一 decision 标签、重激活 Freshness |
| `cold_ghost/training.py` | dense teacher、warm-up、rollout、AMP 累积与恢复 |
| `cold_ghost/checkpoints.py` | checkpoint 保存、验证和配置身份保护 |
| `cold_ghost/data.py` | 图片分组、原评测集排除索引和数据审计 |
| `cold_ghost/evaluation.py` | 复用原 lmms-eval 和结果写出 |
| `cold_ghost/profiling.py` | 新计数器/原 runtime 协议接入 |
| `cold_ghost/diagnostics.py` | 原 skip 与 Ghost 的 representation 诊断 |
| `cold_ghost/cli.py` | 所有命令统一入口 |
| `models/ghost.py` | 提供新增方法的导入入口 |
| `scripts/train_ghost.py` | 训练入口 |
| `scripts/ghost/run_eval.py` | 新评测 Python 入口，参数与 CLI eval 相同 |
| `scripts/ghost/run_setting.sh` | 一次评测，使用 `--config/--task/--out` |
| `scripts/ghost/run_paper_table.sh` | 多配置/多任务 eval sweep |
| `scripts/ghost/run_profile.sh` | profile sweep |
| `scripts/ghost/run_runtime_bench.sh` | runtime sweep |
| `scripts/ghost/train_all.sh` | 去除执行路径重复的 12 权重训练 sweep |
| `scripts/ghost/test_cpu.sh` | 原件哈希、语法、CPU 测试、GQA 修正版检查 |
| `.github/workflows/` | GitHub CPU 自动验证记录和提交工作流 |

示例：

```bash
python scripts/ghost/run_eval.py --config "$CG_LLAVA" --task pope --limit 1 --out experiments/cold_ghost/python_entry_smoke
bash scripts/ghost/run_setting.sh --config "$CG_LLAVA" --task pope --limit 1 --out experiments/cold_ghost/shell_entry_smoke
```

分别展示 Python/shell 薄入口；不是额外的算法。通常只选择一种入口运行某次实验，避免重复测量。

---

## 17. 常见报错与准确处理方式

**`Missing trained adapter` / `Train the Ghost adapter before evaluation`**：先完成对应模型、档位和调度族训练。不要创建空 `best.pt`、加载 debug 权重、取消校验或改用另一档位权重。

**`Missing independent-data manifest` / evaluation index partial**：独立训练图片和完整排除索引尚未准备好，回到第 7 节。缺数据不是用测试集替代的理由。

**`Image leakage`**：清单之间或与评测图片重叠；更正数据来源并重新划分/审计。不要只改 image_id 来绕过，内容哈希也会检查。

**`CUDA unavailable` / `no kernel image` / driver 错误**：先检查安装的是否是 CUDA wheel、显卡是否被正确分配、驱动是否兼容。不要把 CPU 版安装到正式 GPU 环境后继续跑。当前 CI 中的 CPU wheel 仅用于测试，不是第 3 节的服务器安装配置。

**`BF16 not supported`**：当前 Qwen 正式配置需要相应硬件支持。换到支持 BF16 的服务器运行，不能无记录地改 dtype 后仍声称原协议。

**`CUDA out of memory`**：先检查是否同卡有其他进程，是否只暴露一张卡，模型/图片预算是否符合原配置。当前实现不支持 decoder 跨卡切分、CPU offload 和 gradient checkpointing；需要足够显存的设备。随意缩图片、改 keep ratio 或 batch 会改变实验条件，不能作为不加说明的修复。

**`AMP overflow retry`**：是 loss scaling 的重算日志；同一累积窗口成功后才增加 update 计数。若最终仍抛出非有限损失/梯度错误，应保留日志定位问题，不把其当已完成训练。

**`Original source content changed`**：运行 `git diff` 找到对原 79 文件的修改。先备份个人修改，再有选择恢复；不要盲目执行破坏性 `git reset --hard`。第三方 `external/lmms-eval` 的原补丁不属于这个哈希范围。

**`transformers` 导入异常**：确认环境激活正确，`python scripts/ghost/check_dependencies.py` 无冲突且 `transformers==5.4.0`。不要用系统 Python 执行 conda 环境里安装的项目。

**Task 找不到 / RefCOCO 结果异常**：确认 editable checkout 是 v0.7.1，重新运行原 `apply.sh` 并执行 doctor；清理错误的 `BBOX_COORD_FORMAT`、`REFCOCO_PROMPT_STYLE` 环境变量。不同模型的 prompt/坐标解释不应混用。

**`Results already exist`**：使用新输出目录。训练断点用 `--resume`，不是覆盖已有评测结果。

**`pytest` 显示 skipped**：本地缺少 Transformers/Hydra 时部分接口测试会跳过；这不是相应测试通过。安装完整依赖后重跑，在正式 GitHub 记录中查看实际 passed/skipped 数。

**长时间离开终端**：按服务器已有的作业调度/会话管理规范执行；不要把 SSH 窗口关闭等同于作业一定会继续。这里的程序没有远程代跑服务，也没有未完成实验的后台承诺。

---

## 18. 按顺序跑通的完成标志

从零运行的顺序固定为：环境与 CPU 检查 → 下载模型 → 原 baseline 小样本评测 → 独立数据清单与完整隔离审计 → 第一份正式 Ghost 训练 → Ghost 小样本评测 → 完整任务评测 → 其余 12 权重/24配置 → 同后端 profile 与原 runtime → 诊断/消融与汇总。

对应完成标志：doctor `ok=true`；原件校验通过；CPU tests 通过；原 baseline 产生真实 `results.json`；audit `image_overlap=false`；训练 `training_summary.json` 完成且有经过验证的 `best.pt`；Ghost eval manifest `status=completed`；正式结果无 `--limit`；效率 JSON 标明计数后端和真实硬件。

**没有运行过的阶段不要提前标成完成。** 此交付没有给出模型准确率、加速比或论文级效果保证；这些须由训练后的实际测量决定。

## 19. 外部安装依据与代码依据

安装器版本来自 conda-forge Miniforge 官方发布 `26.7.2-0`；PyTorch 2.11.0 / torchvision 0.26.0 的 cu128 与 CPU wheel 对应关系来自 PyTorch 官方 previous-versions 文档。评测框架固定 `EvolvingLMMs-Lab/lmms-eval` v0.7.1。模型、任务、预处理、预算调度依据本仓库未修改的 `configs/`、`scripts/` 和 `profiler/`；Cold/Ghost 算法依据本次确定方案。

- [Miniforge 官方固定版本](https://github.com/conda-forge/miniforge/releases/tag/26.7.2-0)
- [PyTorch 官方历史版本安装命令](https://pytorch.org/get-started/previous-versions/#v2110)
- [lmms-eval v0.7.1 原始依赖文件](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/v0.7.1/pyproject.toml)
- [完整确定方案](docs/DESIGN_COLD_GHOST_ZH.md)
- [实际验证结果及范围](docs/VALIDATION.md)
