# Spatial Mapping Analyzer

当前只有一条执行路径：读入 DAG → 固定 mapping → 验证 → 真实 TT-Sim → CPU 校验 → 保存结果。
Analyzer 每个 op 分配一个 region、一个 core；暂不搜索最优解，也不做 fusion。

## 先跑两个例子

从仓库根目录初始化并编译 TT-Sim（需要支持 C++20 的 g++）：

```bash
git submodule update --init --recursive
cd third_party/ttsim
python make.py src/_out/release_wh/libttsim.so
cd ../../spatial-mapping-analyzer
python -m pip install -r requirements.txt

python run_analyzer.py --program examples/gemm_relu_gemm.yaml
python run_analyzer.py --program examples/scalar_diamond.yaml \
  --inputs examples/scalar_diamond.inputs.json --parallel
```

- Tensor：32×32 int32 的 `Y = ReLU(A @ B) @ C`，默认 seed=0 生成输入。
- Scalar：`p=a+b; left=p+c; right=ReLU(p); y=left+right`。示例输入得到 `p=-2, left=3, right=0, y=3`。
- `--parallel` 允许无依赖的 region 同时开始，Join 等待两条分支完成；不加该选项就顺序执行。
- `--iterations N` 重复完整流程并传递历史反馈；当前 Analyzer 忽略反馈，每次使用相同 mapping 和输入。

默认架构是 `examples/wormhole.yaml`。结果放到新的 `results/run-*` 目录。
可用 `--arch`、`--seed`、`--output`、`--tt-sim-library` 和 `--tt-sim-timeout` 覆盖默认值。
`--inputs` 是名称到扁平 int32 数组的 JSON；矩阵按行排列，scalar 的 shape 为 `[]`，数据数组仍含一个值。

## 按这个顺序 review

| 文件 | 要看什么 |
| --- | --- |
| `run_analyzer.py` | 入口：读配置，调用循环 |
| `pipeline.py` | 主流程：提案、验证、执行、反馈、选 best、保存 |
| `analyzer.py` | 当前 mapping 规则；以后替换 `propose_mapping(arch, program, feedback)` |
| `specs.py`、`mapping_ir.py`、`report.py` | 输入、mapping、报告的数据结构 |
| `validator.py` | op 覆盖、依赖顺序、shape、core 数是否合法 |
| `tt_sim.py` | 启动模拟器子进程，把结果与 CPU reference 对比 |

底层执行只涉及三个辅助文件：`workload.py` 准备输入、reference 和每个 op 的执行描述；
`kernels.py` 生成 RV32IM 指令；`simulator.py` 通过 PCI/BAR 接口装载程序、推进模拟器、取回结果。
进程隔离保留，因为 TT-Sim 遇到非法指令会直接退出进程。上游子模块未修改。

## 看哪些输出

每个 trial 有 `arch.yaml`、`program.yaml`、`mapping.yaml`、`validation.json`、`report.json`。
`inputs.json` 和 `reference.json` 记录输入及 CPU 预期值；`correctness.json` 给出校验结果。
`runner_result.json` 中的 `trace` 和 `waves` 记录 core 分配、分支重叠与完成顺序。
另外保存可重放的执行描述、指令二进制、调用命令及日志。

总目录有 `history.jsonl`、`summary.json`、`best_mapping.yaml`。非法 mapping 不执行；执行失败的 trial
不参与 best 选择。已有结果目录不会覆盖。没有可用结果时 CLI 返回 2。

## 当前范围与检查

实际执行的是 **int32 BRISC kernel**，支持维度不超过 32 的 scalar/vector/matrix、最多八个单 op/core region。
加法和乘法按 int32 溢出规则处理。数据由 host 转发；尚未接入 Tensix FPU/SFPU 或设备端 NoC 传输。
CPU reference 检查所有中间结果，预期值不会写入模拟设备。

API steps 只用于执行记录，不能当硬件 latency。性能指标保持 unsupported，排序分数为固定 synthetic 1，
并列时选择第一个成功 trial。缺少库或不支持的程序会明确失败，没有 mock 回退。

```bash
python -m unittest discover -s tests -v
python export_schemas.py  # 四份 schema 写入 results/schemas，模型定义是唯一来源
```

测试集中在 `tests/test_e2e.py`。本地未编译库时会跳过真实执行测试；CI 强制编译并运行两个例子，上传结果和 schema。
旧 `--backend` / `--execution-policy` 选项已删除；改用默认真实执行和 `--parallel`。
