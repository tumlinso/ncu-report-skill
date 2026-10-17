# Volta (GV100 / Tesla V100 / SM70)

Use this reference when the report was produced on Volta. Keep the shared
Collect → Extract → Diagnose → Recommend workflow, but select a profiler that
still supports GV100: Nsight Compute 2025.2.x is the final supported release
series. Select it explicitly when several `ncu` executables are installed
(`NCU=/path/to/ncu`); later NCU releases do not support Volta collection.

For a standalone harness, compile for the target explicitly, for example
`nvcc -O2 -std=c++17 -lineinfo -arch=sm_70 harness.cu -o harness` with a
CUDA toolkit that supports the target. Record the compiler, target, profiler
version, clock policy, and actual SM clock in the report.

## Evidence available on SM70

Static launch, occupancy, instruction, memory, rule-engine, and source-counter
data can still be useful. Metric spellings and support depend on the selected
2025.2 installation and report sections. Query that profiler (`ncu --query-metrics`,
`--list-sections`) and use the report's actual metric names. The Blackwell list
in [`08-metric-names.md`](08-metric-names.md) is not an SM70 metric contract.
The Python helpers preserve collected raw metrics and treat absent values as
unavailable; never substitute zero for a metric that was not collected or is
not implemented on the GPU.

PM-sampling time series used by the Blackwell workflow are unavailable on
Volta. A missing timeline is an unsupported analysis dimension, not a failed
collection and not evidence that utilization was zero or flat. Static averages
cannot establish a tail shape. Older report APIs may also lack newer optional
methods such as `timed_warp_samples()`; use the analyses present in that API.

The following names were queryable for `gv100` with Nsight Compute 2025.2.1.
They establish that the profiler knows the metric for this chip; they do not
prove it was collected in a particular report. Check `action.metric_names()`
or the report's section results and request an appropriate section or metric
when needed. Rollup suffixes such as `.avg`, `.sum`, or `.pct` appear in the
report and vary by metric.

| Diagnostic question | GV100 metric examples in NCU 2025.2.1 | Useful sections |
|---|---|---|
| Active and eligible warps; issue | `sm__warps_active`, `smsp__warps_eligible`, `sm__issue_active`, `smsp__issue_active` | `Occupancy`, `SchedulerStats`, `SpeedOfLight` |
| Issue stalls | `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active`, `...short_scoreboard...`, `...barrier...`, `...math_pipe_throttle...` | `SchedulerStats`, `SourceCounters` |
| Spills / local traffic | `smsp__sass_inst_executed_op_local_ld`, `smsp__sass_inst_executed_op_local_st`, `l1tex__t_requests_pipe_lsu_mem_local_op_ld`, `..._st` | `InstructionStats`, `MemoryWorkloadAnalysis`, `SourceCounters` |
| Shared-memory bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` and its `_op_ld` / `_op_st` variants | `MemoryWorkloadAnalysis`, `SourceCounters` |
| Global traffic and caches | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld`, `..._st`, `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate`, `lts__t_sector_hit_rate` | `MemoryWorkloadAnalysis`, `SpeedOfLight` |
| DRAM traffic and rate | `dram__bytes`, `dram__bytes_read`, `dram__bytes_write`, `gpu__dram_throughput` | `MemoryWorkloadAnalysis`, `SpeedOfLight` |
| FP and Tensor Core instructions | `smsp__inst_executed_pipe_fma`, `smsp__inst_executed_pipe_fp16`, `smsp__inst_executed_pipe_tensor_op_hmma`, `sm__ops_path_tensor_src_fp16_dst_fp32`, `sm__pipe_tensor_op_hmma_cycles_active` | `ComputeWorkloadAnalysis`, `InstructionStats`, `SpeedOfLight` |

Ellipses in metric names above abbreviate shared prefixes; expand names using
the selected profiler before requesting them. Not every listed metric is part
of the default `full` set. The availability of an occupancy-limit or launch
resource field also depends on the section and report; use the Occupancy
section's displayed limits rather than inventing missing register or shared
memory values.

SourceCounters and line attribution are separate evidence from PM timelines.
When the profiler collected per-PC samples and the binary has line information,
inspect the supported source/stall fields. Absence of samples may mean the
section was not collected or that this report has no source counters; absence
of source lines may mean missing `-lineinfo` or source mapping. The source-line
workflow has been validated end to end on V100 with Nsight Compute 2025.2.1
using the [small dependent-load fixture](../tests/README.md): all 44 sampled PCs
mapped to source, and the helper's counts agreed with NCU's source export.
This qualifies that fixture and toolchain, not every kernel or inline layout.
A sampled PC identifies
the consumer instruction at which a warp could not issue; inspect adjacent
instructions and source dependencies before assigning cause.

## Diagnose the measured bottleneck

Start with valid counters, the selected kernel and launch geometry. Check the
grid against the device's SM count, waves, theoretical and achieved occupancy,
and active versus eligible warps. Occupancy is a capacity measure, not a score:
low occupancy matters when it leaves too few independent warps to hide the
observed latency. High occupancy cannot compensate for a small grid, little
instruction-level parallelism, or a serialized dependency chain.

For scheduler limits, inspect issue activity, eligible warps per cycle, and
the collected warp-issue stall categories. A high stall share is a clue, not a
root cause: attribute it to the sampled consumer and verify the instruction,
data dependency, and surrounding code. Distinguish dependency waits from
barrier/synchronization, execution-pipeline saturation, and no-work conditions.

Check registers per thread, occupancy limits attributed to registers, local
load/store traffic, and spill evidence together. High register count alone does
not prove a problem. Spilling is more concerning when local-memory traffic and
memory stalls are present; test a smaller tile or shorter live ranges and
compare unprofiled throughput.

For shared memory, examine allocation per block and occupancy limits, actual
versus ideal shared-memory wavefronts or conflict counters when available, and
the source access pattern. Capacity pressure can reduce residency; bank
conflicts serialize accesses. Do not infer either from shared-memory use alone.

For global memory, combine requested/transferred sectors, L1 and L2 behavior,
DRAM bytes or throughput, and load/store instruction evidence. High DRAM
throughput near the measured bandwidth ceiling supports a bandwidth-bound
diagnosis. Low DRAM throughput with long dependency stalls suggests latency or
insufficient concurrency, not DRAM saturation. Cache hit rates are meaningful
only with their traffic counts and the relevant cache path; an isolated hit
percentage can mislead. Reduce unnecessary bytes and improve coalescing before
adding staging or more complicated pipelines.

For arithmetic, compare instruction mix and active compute-pipeline evidence
with the operation performed. Volta has first-generation Tensor Cores and
WMMA. Eligible matrix operations can use FP16 inputs with FP32 accumulation;
validate shape, layout, numerical tolerance, and actual Tensor Core counters.
Do not infer Tensor Core use from datatype alone. Scalar FP32 work remains a
valid path when the operation or accuracy requirements do not fit an eligible
matrix formulation.

SM70 does not support BF16 Tensor Core arithmetic, TF32, `cp.async`, TMA,
WGMMA, or Blackwell `tcgen05` instructions. Do not recommend them for a Volta
kernel. WMMA is supported on Volta; do not describe it as Ampere-only.

## Report conclusions at the right confidence

For each conclusion, state its evidence class:

- **Measured bottleneck:** supported by available metrics or samples, with
  names and values and any necessary context.
- **Working hypothesis:** consistent with the evidence but needs a targeted
  source inspection, counter, or controlled benchmark to confirm.
- **Unavailable analysis:** the hardware, report, profiler API, or source
  information cannot provide the needed measurement. Name that limitation;
  do not fill it with a guessed value.

Use unprofiled benchmark runs for speed comparisons. Nsight Compute replay
counts and durations characterize collection, not application throughput.
Change one lever tied to the diagnosed limiter and remeasure using the same
workload and clock policy.
