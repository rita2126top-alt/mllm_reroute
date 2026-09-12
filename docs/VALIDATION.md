# Cold/Ghost 交付验证报告

## 1. 验证范围

本报告区分“实现完成”“CPU 实际运行通过”和“尚未运行的 GPU/正式实验”。没有把 7B 模型或正式 benchmark 的效果写成已验证结果。

原始 ZIP 的 79 个文件按 SHA-256 逐一核对，均保持不变。原始源码导入提交：`7789923889980fdfd7391bda8526aec8730c7d2c`。首个完整扩展提交：`cf106f42a6e2b9069fc6df07f757f980a08d5c8c`。

## 2. 已完成的测试

### 2.1 本地 CPU

本地 Python 3.13 / PyTorch 2.10.0+cpu：33 个测试通过，2 个测试模块因未安装 Transformers/Hydra 而跳过。新增 Python 文件通过 AST 语法检查，新增 shell 脚本通过 `bash -n`，原始 79 个文件通过哈希检查。

测试包含：Ghost 初始化 identity 等价、Full/Ghost/Cold 集合互斥、Cold 状态逐位不变、阶段内 Ghost 逐层更新、末阶段停用、decode 停用、重激活时读取已更新 deferred state、双 compact 路径一致、梯度与四项损失、独立数据隔离、配置矩阵、检查点身份、两阶段训练/保存/恢复、FP16 溢出重试与禁止漏计更新。

### 2.2 首次真实 Transformers 架构 CPU 测试

GitHub Actions：

https://github.com/rita2126top-alt/mllm_reroute/actions/runs/34679103429

结果：**39 passed，0 failed，0 errors，0 skipped**。执行环境为 Python 3.10.21、PyTorch 2.11.0+cpu、torchvision 0.26.0+cpu、Transformers 5.4.0、Accelerate 1.13.0、Hydra 1.3.2。

这是实际实例化的随机小型 LLaVA 和 Qwen2.5-VL 架构，不是对 Hugging Face 接口做纯 mock，也不是下载并评测 7B 预训练权重。验证了多模态 prefill、KV cache、greedy generation、冻结骨干上的 Ghost 梯度传播及两种执行路径。原始 JUnit/环境记录保存在 `docs/validation/initial_native_cpu/`。

## 3. 已识别的原始文件问题

上传 ZIP 的根目录 `test.py` 原本即有语法错误（第 7 行），并含未完成的 GQA 示例。按“原文件不变”的要求保留，没有隐藏这个错误。因此不能宣称“原始整个仓库零语法错误”。

它不被原项目评测入口或新增训练/推理入口导入。独立修正版是 `cold_ghost/examples/gqa_fixed.py`。`cold_ghost.cli verify` 明确列出原始问题，同时检查新代码；`pytest.ini` 将测试范围限定为真实新增测试目录。

## 4. 完整依赖检查中的 Decord 问题

原始 requirements 中的 Decord 0.6.0 在新 pip 的 `pip check` 中报告平台 WHEEL 元数据不兼容。新增的 `scripts/ghost/check_dependencies.py` 不修改包、WHEEL 元数据或原始依赖文件，也不屏蔽一般依赖冲突。

只有 Linux x86_64 上出现**唯一这一条**告警时，它才实际生成 64×64 的双帧 AVI，并用 Decord CPU 解码验证形状和帧顺序。任何额外版本冲突、导入失败或解码失败都会退出报错。不能把这个例外描述为“原生 pip check 无告警”。16×16 的极小视频触发过旧 Decord 的行对齐问题，测试素材已明确调整为 64×64。

## 5. 未执行和不保证的事项

没有连接用户的服务器；没有运行 CUDA kernel、下载 7B 权重进行生成或训练、完成 1000+2000 updates、下载全部评测数据、运行完整 744 项评测、测量真实 GPU 峰值显存/延迟，也没有生成正式训练完成的 Ghost checkpoint。

服务器的 NVIDIA driver、CUDA wheel 兼容性、剩余显存、网络权限、数据许可和模型效果必须由服务器上的实际运行确认。已提供对应命令和失败检查，但 CPU 通过不能替代 GPU 集成测试，也不能保证所有硬件上都无错误。

本版本的确定支持范围为：Linux x86_64、单 GPU、batch size 1、单图像 LLaVA-1.5-7B/Qwen2.5-VL-7B、原两种 compact 路径。不新增 DDP、CPU offload、多图像/视频训练或 gradient checkpointing 支持。原始可选 DeepSpeed profiler 文件保留，但未验证其旧版本组合；新增 profiler 在同一计数口径下覆盖 baseline 与 Ghost，并披露未计数算子，不把它伪装为原 DeepSpeed 表的相同口径。

## 6. 重新运行

在指南的依赖安装完成后，于仓库根目录执行：

```bash
python scripts/ghost/check_dependencies.py
python -m cold_ghost.cli doctor --require-hf --require-lmms
python -m cold_ghost.cli verify
bash scripts/ghost/test_cpu.sh
```

首次使用从根目录的 `RUN_GUIDE_COLD_GHOST_ZH.md` 开始；算法依据是 `docs/DESIGN_COLD_GHOST_ZH.md`。
