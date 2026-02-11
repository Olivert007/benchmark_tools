# Profile Tools — Quick Start

ROCm GPU Profiling 工具集，封装 `rocprofv3` 的 ATT / PMC 追踪流程和 Triton IR/ASM 导出。

## 前置条件

- ROCm 环境已安装（`rocprofv3`、`amd-smi` 在 PATH 中）
- Python 环境已安装 `torch` 和 `triton`

## 文件说明

| 文件 | 用途 |
|------|------|
| `rocprofv3_att_trace.sh` | ATT 追踪：逐指令 Hitcount / Latency / Stall 分析 |
| `rocprofv3_pmc_trace.sh` | PMC 追踪：硬件计数器采集（Cache、LDS、Occupancy 等） |
| `triton_ir_asm_dump.sh` | 导出 Triton 编译器中间表示（MLIR / LLVM IR / AMDGCN） |
| `conv3d_att_trace.yaml` | Conv3D Kernel 的 ATT 配置 |
| `conv3d_pmc_trace.yaml` | Conv3D Kernel 的 PMC 配置（10 pass，50+ 计数器） |

---

## 1. ATT 追踪

```bash
# 最简用法（三个必选参数：-f 脚本、-c 配置、-o 输出目录）
./rocprofv3_att_trace.sh \
    -f ./your_script.py \
    -c ./conv3d_att_trace.yaml \
    -o ./output_att
```

**常用选项：**

```bash
# 指定 GPU + Triton 日志
./rocprofv3_att_trace.sh \
    -f ./your_script.py \
    -c ./conv3d_att_trace.yaml \
    -o ./output_att \
    -d 0 -v 1

# 启用 PC Sampling（Beta）
./rocprofv3_att_trace.sh \
    -f ./your_script.py \
    -c ./conv3d_att_trace.yaml \
    -o ./output_att \
    -p -m host_trap -u time -i 1

# 向 Python 脚本传参（用 -- 分隔）
./rocprofv3_att_trace.sh \
    -f ./your_script.py \
    -c ./conv3d_att_trace.yaml \
    -o ./output_att \
    -- --no-torch-baseline --repeat-runs 30
```

**输出目录结构：**

```text
output_att/
└── your_script_triton_v3.3.1+git..._20260211_143052/
    ├── build_info.txt            # 环境快照（Triton 版本、LLVM commit 等）
    ├── rocprof_execution_log.txt # 执行日志
    └── *.json / *.csv            # ATT 追踪数据
```

> 完整参数说明：`./rocprofv3_att_trace.sh -h`

---

## 2. PMC 追踪

```bash
# 最简用法
./rocprofv3_pmc_trace.sh \
    -f ./your_script.py \
    -c ./conv3d_pmc_trace.yaml \
    -o ./output_pmc
```

**常用选项：**

```bash
# 指定 GPU（ROCR_VISIBLE_DEVICES 索引）
./rocprofv3_pmc_trace.sh \
    -f ./your_script.py \
    -c ./conv3d_pmc_trace.yaml \
    -o ./output_pmc \
    -g 0

# 清除旧输出后重新追踪
./rocprofv3_pmc_trace.sh \
    -f ./your_script.py \
    -c ./conv3d_pmc_trace.yaml \
    -o ./output_pmc \
    --clear

# 向 Python 脚本传参
./rocprofv3_pmc_trace.sh \
    -f ./your_script.py \
    -c ./conv3d_pmc_trace.yaml \
    -o ./output_pmc \
    -- --dtype bf16 --batch-size 512
```

**注意：** 脚本执行前会自动将 GPU 设为 `stable_std`（锁定频率），结束后自动恢复 `auto`。如需修改 GPU 索引映射，编辑脚本顶部的 `AMD_SMI_INDICES` 和 `ROCR_VISIBLE_DEVICES_INDICES` 数组。

> 完整参数说明：`./rocprofv3_pmc_trace.sh -h`

---

## 3. Triton IR/ASM 导出

```bash
# 最简用法
./triton_ir_asm_dump.sh \
    -f ./your_script.py \
    -o ./output_dump
```

**常用选项：**

```bash
# 指定 GPU + 最大日志级别
./triton_ir_asm_dump.sh \
    -f ./your_script.py \
    -o ./output_dump \
    -d 0 -v 4

# 自定义子目录名
TRITON_DUMP_DIR_NAME=my_dump ./triton_ir_asm_dump.sh \
    -f ./your_script.py \
    -o ./output_dump
```

**输出内容：** 编译缓存目录下包含 MLIR（`*.ttir`、`*.ttgir`）、LLVM IR（`*.llir`）、AMDGCN 汇编（`*.amdgcn`）等 Triton 编译全链路产物。

> 完整参数说明：`./triton_ir_asm_dump.sh -h`

---

## 自定义 YAML 配置

如需为其他 Kernel 创建配置，参考已有文件：

- **ATT**：修改 `conv3d_att_trace.yaml` 中的 `kernel_include_regex` 匹配你的 Kernel 名称
- **PMC**：修改 `conv3d_pmc_trace.yaml` 中的 `kernel_include_regex`，按需增删 `pmc` 计数器（每个 pass 最多 8 个硬件计数器）
