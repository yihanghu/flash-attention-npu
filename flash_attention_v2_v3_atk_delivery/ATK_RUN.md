# FA v2/v3/v4 ATK 测试运行说明

本目录 `flash_attention_v2_v3_atk_delivery` 存放 flash_attention 系列算子的 ATK 测试工程，
采用「双标杆」方式验证 NPU 算子精度：CPU 作为真值标杆（`--bm_device cpu`），NPU 作为被测算子。

## 1. 环境准备

```bash
# 加载 CANN 环境（torch_npu / NPU 依赖）
source /data/hyh/env/ascend-toolkit/set_env.sh
```

ATK 可执行文件默认使用 `atk`（可用 `which atk` 确认，例如 `/usr/local/bin/atk`），
亦可用 `ATK_BIN` 环境变量覆盖：

```bash
export ATK_BIN=/usr/local/python3.11.15/bin/atk
```

## 2. 用例生成

对每个接口目录执行，生成随机用例 JSON：

```bash
cd /data/hyh/flash_attention_v2_v3_atk_delivery/interfaces/<interface>
atk case -f <interface>.yaml -p . -s 0
```

生成产物：`result/<interface>/json/all_<interface>.json`（以及 excel/csv）。

其中 `<interface>` 例如 `flash_attn_func_v4`、`flash_attn_varlen_func_v4`。

## 3. 精度测试执行

### 3.1 单用例调试（方式 A）

```bash
cd /data/hyh/flash_attention_v2_v3_atk_delivery/interfaces/<interface>
atk task -c result/<interface>/json/all_<interface>.json -n nodes_accuracy.yaml -p . --task accuracy --bm_device cpu --single_process -s 0 -e 1
```

参数说明：
- `-n nodes_accuracy.yaml`：节点配置 = NPU(device 0) + CPU。
- `--bm_device cpu`：以 CPU 为真值标杆，NPU 为被测算子。
- `-s 0 -e 1`：只运行第 0 号用例，用于快速验证。

### 3.2 全量执行

用接口目录脚本（含 `--save_data`、`--log warning` 及结果归档）：

```bash
cd /data/hyh/flash_attention_v2_v3_atk_delivery/interfaces/<interface>
bash run_accuracy.sh
```

或批量运行所有接口：

```bash
bash /data/hyh/flash_attention_v2_v3_atk_delivery/accuracy/run_accuracy.sh
```

## 4. 结果查看

每次运行在 `interfaces/<interface>/atk_output/<时间戳>/report/` 生成 Excel 报告：

- `summary`：各后端用例通过率与「精度是否达标」。
- `statistic`：逐用例输入/输出，以及 `npu_0_精度通过`、`cpu_0_精度通过` 等列。

通过标志：`summary` 表中 `精度是否达标 = Pass`。

## 5. 可选：GPU 三标杆

```bash
FA_ATK_GPU_BM=1 bash run_accuracy.sh
```

使用 `nodes_accuracy_gpu.yaml`（额外增加 GPU 节点，需提前在该文件中配置 GPU 主机 IP/端口，
且关闭单进程模式）。