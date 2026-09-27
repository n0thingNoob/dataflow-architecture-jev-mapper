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

当前 candidate generator 只使用 backend 已经能忠实执行的自由度：
- 合法 topological order
- serial / dependency-parallel execution policy
- 显式 logical-core placement

暂不生成 fusion 或 multi-core region，因为当前 BRISC backend 还不能忠实执行这些 mapping。BRISC 路径现在只做 correctness：不产生 ranking objective，也不输出 `best_mapping.yaml`。TT-Sim 的 API steps 仅作调度诊断，不作为硬件 cycles。

## 代码阅读顺序

| 文件 | 责任 |
| --- | --- |
| `run_analyzer.py` | CLI，选择 passthrough/search analyzer |
| `pipeline.py` | proposal → validation → execution → feedback/history |
| `analyzer.py` | analyzer 策略；未来 learned/Jev analyzer 也放这里 |
| `candidate_generator.py` | 纯函数、确定性的 executable candidate 枚举 |
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

实际计算仍是 int32 BRISC kernel；数据由 host 转发。尚未接入 Tensix FPU/SFPU、设备端 NoC、fusion 或多核 region。

每个 trial 保存：
- `arch.yaml`
- `program.yaml`
- `mapping.yaml`
- `validation.json`
- `report.json`
- simulator/debug artifacts

整个 run 保存 `history.jsonl` 和 `summary.json`；只有 backend 提供可比较 objective 时才生成 `best_mapping.yaml`。

## 测试

```bash
python -m unittest discover -s tests -v
python export_schemas.py
```

CI 会编译固定版本 TT-Sim，运行 unit tests、passthrough E2E 和 search E2E。

下一阶段需要解决的是可信 performance label；在此之前不接 learned/Jev model，也不把 synthetic objective 当训练标签。


## 可选 Tensix placement probe

BRISC correctness 路径之外，仓库现在提供一个可选的 TT-Metal/Tensix probe，用来验证：

```text
Mapping IR placement -> TT-Metal CoreCoord -> TT-Sim Tensix compute
```

它目前只支持一个 32x32 BF16 add，仍然不提供 timing objective。构建和运行方法见
`tensix_probe/README.md`。TT-Metal 作为外部依赖使用，不作为本仓库 submodule。
