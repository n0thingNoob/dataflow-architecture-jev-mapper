# Spatial Mapping Analyzer

当前有两种 analyzer：
- 默认 passthrough：每个 op 一个 region、一个 core。
- `--search`：枚举当前 TT-Sim backend 真正能执行的不同 mapping，用来收集后续性能/训练数据。

两种模式都走同一条真实执行路径：DAG → mapping → 验证 → TT-Sim → CPU 校验 → report/history。

## 快速运行

从仓库根目录初始化并编译 TT-Sim：

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

搜索多个可执行 mapping：

```bash
python run_analyzer.py \
  --program examples/scalar_diamond.yaml \
  --inputs examples/scalar_diamond.inputs.json \
  --search --candidate-limit 12 --iterations 12
```

candidate search 由 backend 显式声明哪些 Mapping IR 自由度会改变真实执行，并用 backend execution signature 去掉 lowering 后等价的候选。当前：
- BRISC 搜索合法 topological order、execution policy 和 logical-core placement。
- Tensix placement probe 只搜索 placement。
- 两核 Tensix chain 搜索所有不同的 producer→consumer ordered core pairs；不会把未下沉到 TT-Metal 的 execution policy 当成不同候选。

暂不生成 fusion 或 multi-core region。BRISC 路径现在只做 correctness：不产生 ranking objective，也不输出 `best_mapping.yaml`。TT-Sim 的 API steps 仅作调度诊断，不作为硬件 cycles。

## 代码阅读顺序

| 文件 | 责任 |
| --- | --- |
| `run_analyzer.py` | CLI，选择 passthrough/search analyzer |
| `pipeline.py` | proposal → validation → execution → feedback/history |
| `analyzer.py` | analyzer 策略；未来 learned/Jev analyzer 也放这里 |
| `candidate_generator.py` | 纯函数、确定性的 executable candidate 枚举 |
| `measurement_collection.py` | 随机化 repeated measurement 与稳健聚合 |
| `backend_contract.py` | backend capabilities、candidate signature 和 observed execution contract |
| `specs.py` | architecture/program 数据结构 |
| `mapping_ir.py` | Mapping IR，包括 region 和 logical-core placement |
| `validator.py` | program/mapping/placement 合法性 |
| `tt_sim.py` | TT-Sim backend 与 report |
| `workload.py` | 将 Mapping IR lowering 到当前 BRISC harness |
| `kernels.py`、`simulator.py` | 底层 BRISC 指令和隔离的 TT-Sim runtime |

## 当前执行范围

Tensor 示例是 32×32 int32 的 `Y = ReLU(A @ B) @ C`。
Scalar diamond 是：

```text
p = a + b
left = p + c
right = ReLU(p)
y = left + right
```

默认 BRISC 路径仍是 int32 kernel，数据由 host 转发。可选 TT-Metal 路径已覆盖单核 BF16 add，以及两个 Tensix core 之间通过设备端 NoC 传递中间 tile 的两级 add chain。TT-Sim runtime 仍只做 correctness；在真实 Wormhole 上使用 `--tensix-runtime device` 时，单核 BF16 add 和两核 chain 都会从 TT-Metal device profiler 读取 `DEVICE KERNEL DURATION [ns]`，并作为 `source=measured` 的 ranking objective。当前仍未支持通用 fusion 或多核 region。

每个 trial 保存：
- `arch.yaml`
- `program.yaml`
- `mapping.yaml`
- `validation.json`
- `report.json`
- simulator/debug artifacts

整个 run 保存 `history.jsonl` 和 `summary.json`；每个 run 带稳定的 `run_id`，每个 trial 持久化 backend `requested_execution_signature`。backend report 另外保存运行时实际观察到的 `observed_execution`，并以 `report.objective` 作为 objective 的唯一持久化来源，不再在 trial 顶层镜像重复字段。这些 identity 不可混用。只有 backend 提供可比较 objective 时才生成 `best_mapping.yaml`。

## 测试

```bash
python -m unittest discover -s tests -v
python export_schemas.py
```

CI 会编译固定版本 TT-Sim，运行 unit tests、passthrough E2E 和 search E2E。

真实 Wormhole device profiler 的 kernel duration 现在可以作为 measured performance label；TT-Sim、host wall-clock 和 synthetic 数值都不会被当成真实训练标签。

可用 `export_dataset.py` 把一个或多个 analyzer run 导出成训练 JSONL：

```bash
python export_dataset.py \
  --run results/device-run-a \
  --run results/device-run-b \
  --output results/dataset/measured-mappings.jsonl
```

exporter 只接受 `status=ok`、`correctness=passed`、`objective.source=measured`，并同时具备 requested execution signature、observed execution identity 和 measurement context 的 trial。dataset 将 `observation_id`、`content_hash`、`program_group_id`、`execution_group_id` 分开：重复测量保留为独立 observation，同一 observation 内容变化会被视为冲突，后续交叉验证按 program/execution group 防止泄漏。对应 JSON Schema 由 `export_schemas.py` 输出为 `dataset_record.schema.json`。完整 identity/CV 约束见 `ARCHITECTURE.md`。

物理设备采样使用 `collect_measurements.py`，支持 `--backend single-add` 和 `--backend two-add-chain`：每个 effective candidate 重复执行并按固定 seed 随机打散顺序，raw trial 全部保留，同时输出 median/MAD/min/max 聚合。collection 模式不会按单次测量生成 `best_mapping.yaml`。TT-Metal device measurement 还会把 profiler API 的 duration 与独立生成的 `cpp_device_perf_report.csv` 交叉核对；不一致的 observation 会被拒绝。

当前 measured workload 已覆盖单核 placement 与两级 direct-NoC chain placement；下一阶段先固定 split 并建立 heuristic baseline，再决定 learned scorer。


## Heuristic baseline

先不接 learned model。当前 baseline 只对 backend 已经证明可执行的候选 mapping 做确定性打分：

```text
Program + Architecture
        ↓
candidate_generator
        ↓
HeuristicScorer
        ↓
measured-latency evaluation
```

当前 `HeuristicScorer` 是 topology-aware deterministic cost model。正式 ranking 只使用三个 placement-sensitive 项：tensor-size weighted critical-path hops、multicast-aware network byte-hops、以及 deterministic XY-route peak directed-link load。per-core endpoint traffic 仅保留为 diagnostic，不参与 ranking，因为在当前 injective one-op-per-core 搜索空间里它经常是 placement-invariant。若 architecture 同时提供 `bandwidth_bytes_per_cycle` 与 `link_latency_cycles`，还会产生一个 cycle-like critical-path diagnostic；该值同样不参与 ranking。当前 topology model 只接受显式 logical mesh / `wormhole-noc` profile，并把 `wormhole-noc` 当作逻辑 mesh proxy，不对其他 network topology 静默套用 XY routing。对真正对称、结构等价的 placement，heuristic 会保持并列，而不是加入任意 core-ID 偏置。

生成固定、无 program/execution leakage 的 split manifest：

```bash
python dataset_split.py \
  --dataset results/dataset/measured-mappings.jsonl \
  --output results/dataset/split.json
```

在 held-out split 上对 heuristic 与真实 measured latency 做比较：

```bash
python evaluate_scorer.py \
  --dataset results/dataset/measured-mappings.jsonl \
  --split-manifest results/dataset/split.json \
  --split test
```

evaluation 会在相同 architecture、objective 和完整 measurement context 内比较候选；repeated observations 先按 `execution_group_id` 取 median。对于 heuristic 同分候选，不再只报告候选枚举顺序选中的一个值，而是同时记录 deterministic selection、top-score tie 数量、tie 集合中的 best/worst measured latency，以及对应 regret range。split 使用 deterministic balanced program-group assignment：数据量允许时 train/validation/test 都非空，同时继续禁止同一 `execution_group_id` 跨 split。learned scorer 放到后续独立 PR。

## TT-Sim profiler experiment

在不使用真实 Wormhole 的情况下，可以用 pinned TT-Metal + TT-Sim + TT-Metal device-profiler instrumentation 做 heuristic sanity experiment。该路径使用 `--tensix-runtime ttsim-profile`，profile objective 会明确标记为：

```text
name   = ttsim_profile_kernel_duration
source = estimated
unit   = ns
```

这些数值只用于 simulator-side ranking/regret 实验，**不会**被 `export_dataset.py` 接受为真实训练标签，也不能解释成 physical Wormhole latency。

CI 中会对两级 Tensix chain 的全部 12 个 ordered producer→consumer placements 各运行 3 次，共 36 次 profiler execution，并生成：

```text
results/ci-tensix-profile/
  summary.json
  heuristic_vs_ttsim_profile.json
  trial_*/
```

`evaluate_ttsim_profile.py` 会按 mapping 聚合 profiler median/MAD，再报告 heuristic 的 deterministic selection、oracle、top-score tie 数量及 tie regret range。

## 可选 Tensix placement probe

BRISC correctness 路径之外，仓库现在提供一个可选的 TT-Metal/Tensix probe，用来验证：

```text
Mapping logical core ID -> TT-Metal logical worker CoreCoord -> observed worker CoreCoord -> Tensix compute
```

它支持单个 32x32 BF16 add placement probe，以及两个不同 Tensix core 上的两级 BF16 add chain；chain 的中间 tile 直接通过 NoC 传递，不返回 host。TT-Sim 模式只验证 correctness/placement；真实 device 模式下，单核 add 与两核 chain 都会读取 TT-Metal device profiler 的 kernel duration 并提供 measured objective。search 会按 backend execution signature 去重，避免重复运行 lowering 后相同的 placement。构建和运行方法见
`tensix_probe/README.md`。TT-Metal 作为外部依赖使用，不作为本仓库 submodule。
