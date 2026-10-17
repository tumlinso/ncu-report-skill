# Final Report Template

The report is the deliverable. Everything else (`.ncu-rep`, Python artifacts, CSVs) is evidence. Structure matters: a busy reader should see the top findings in 30 seconds and be able to drill into details if they want.

Save as `$PROFILE_RUN_DIR/REPORT.md`.

---

## Template

```markdown
# `<kernel_name>` Profiling Report

**Kernel:** `<exact kernel name or template instantiation>`
**Target GPU:** `<actual GPU name, SM count, compute capability>` (`nvidia-smi --query-gpu=name,compute_cap --format=csv`)
**Profiler used to collect:** `<version from report metadata or collection provenance; unknown if unavailable>`
**Reader API:** `ncu_report <exposed version or unknown>`; module location: `<path or unknown>`
**Clock:** `<selected supported policy or unknown>`, SM at `<measured value or unavailable>`
**Compile flags:** `<actual flags and target>` (or the prebuilt library profiled, and whether it has line information)
**Profile date:** YYYY-MM-DD
**Run directory:** `<profile_root>/<run_name>/`

---

## 0. Profiling setup

> How exactly did we get these numbers? Required for reproducibility.

- Harness: `harness/*` — what it is (standalone harness / a driver calling the real library / the original binary). Why.
- Workloads: which real tensors / shapes were used. Cite the workload UUID or shape tuple.
- Dispatch paths covered: list each `(SF / template params, grid, block)` combination profiled.
- Metric-name caveats: any metric names that differ from stock NCU docs (see `08-metric-names.md`).
- Evidence availability: list measurements present and analyses unavailable on this GPU, report, or API. The current `$NCU --version` does not identify which profiler produced an older report.

Minimal runnable command listing:

    # Select the profiler and a clock policy supported by its --help output.
    export NCU="${NCU:-ncu}"
    # Optional: CLOCK_CONTROL=base for 2025.2, or boost on a newer profiler if supported.
    CLOCK_ARGS=()
    if [[ -n "${CLOCK_CONTROL:-}" ]]; then CLOCK_ARGS=(--clock-control "$CLOCK_CONTROL"); fi

    # Set TARGET_ARCH to the actual target (for example, sm_70 for Volta).
    nvcc -O2 -std=c++17 -lineinfo -arch="$TARGET_ARCH" harness.cu -o "$PROFILE_RUN_DIR/harness/harness"

    # Request available standard sections; PM timelines depend on GPU support.
    "$NCU" --set full --import-source yes "${CLOCK_ARGS[@]}" \
        -k "regex:<kernel_regex>" -c 1 \
        -o "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" \
        "$PROFILE_RUN_DIR/harness/harness" [args]

### Artifacts

    <run_name>/
    ├── REPORT.md                       ← this file
    ├── harness/...                     ← standalone harness
    ├── reports/                        ← raw .ncu-rep files (re-openable with ncu-ui)
    └── analysis/                       ← scripts + extracted metrics

---

## 1. Headline numbers

> A single table that tells the whole story at a glance.

| Metric | `<tag1>` | `<tag2>` | Source |
|---|---:|---:|---|
| **Duration** | X µs | Y µs | `gpu__time_duration.sum` |
| SM throughput (% peak) | X% | Y% | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| Memory throughput (% peak) | X% | Y% | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM throughput (% peak, reads + writes) | X% | Y% | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| L1 hit rate | X% | Y% | `l1tex__t_sector_hit_rate.pct` |
| L2 hit rate | X% | Y% | `lts__t_sector_hit_rate.pct` |
| Tensor Core usage | X or unavailable | Y or unavailable | cite a collected, target-supported metric |
| Reg / thread | X | Y | `launch__registers_per_thread` |
| Theoretical / Achieved occupancy | X% / Y% or unavailable | ... | cite reported fields |
| Waves / SM | X or unavailable | Y or unavailable | cite reported field |

These rows are candidate fields. Use only metrics present in each report and
supported by its architecture; leave missing values unavailable rather than
recording zero.

**One-line read:** <"The kernel runs at X% of peak SM throughput — it's latency-bound on Y, not DRAM-BW-bound."> — this is the punchline.

---

## 2. Per-dimension analysis

> Walk through the six analysis dimensions, cite metrics, and state findings. Distinguish a measured bottleneck, a hypothesis needing evidence, and an unavailable analysis.

### 2.1 SM occupancy & launch geometry
<grid size, block size, waves/SM, occupancy, register/shared-mem limits where reported; otherwise mark unavailable>

### 2.2 Thread-block balance (tail effect)
<available per-SM active-cycle evidence, supported PM timeline shape if present, input distribution; mark unsupported or uncollected evidence unavailable>

### 2.3 Instruction-level stall analysis
<stall breakdown %, top source-line hotspots (cite file:line + samples + stall type)>

### 2.4 Tensor Core utilization
<measured value or unavailable; a missing counter is not 0%>

### 2.5 SM utilization timeline
<supported timeline shape with plot reference, or state unavailable for hardware/profiler/report>

### 2.6 Memory access pattern
<sectors/request, L1/L2 hits, DRAM throughput, store efficiency, register spill>

### 2.7 Additional findings
<items from NCU rule engine not otherwise mentioned — each with the rule's `Est. Speedup: X%`, and whether the counters bear it out>

---

## 3. Summary diagnosis

| Factor | `<tag1>` | `<tag2>` | Impact |
|---|---|---|---|
| <factor 1> | <status> | <status> | <ranked impact> |
| <factor 2> | ... | ... | ... |

---

## 4. Optimization directions (ranked by impact)

> Each priority: name the change, cite evidence, estimate magnitude, flag effort.

### Priority 1 — <one-line name>

<what to do, concretely, with line numbers / function names from the existing kernel>

**Evidence:**
- <metric + value>
- <NCU rule + est. speedup>

**Expected impact:** <X% end-to-end, Y% on the hot path>, <which workloads benefit>

**Effort:** <low/medium/high + rough description of the code change>

### Priority 2 — ...

<same structure>

### Priority 3 — ...

(Stop at 3-5. More dilutes the signal.)

---

## 5. Confidence & caveats

- What I'm sure about: <list>
- What I'm uncertain about: <list + what would resolve the uncertainty>
- Anything the profile couldn't answer that the user should know: <list>

---

## 6. Reproduction

    # from the repo root
    export PROFILE_RUN_DIR=<profile_root>/<run_name>
    <one-block runnable script that builds the harness, runs ncu, and parses>
```

---

## Style rules

- **Cite specific metric values for every claim.** "SM throughput X.X%" (with the actual number from your report) > "SM throughput is low".
- **Name files and line numbers.** "Line L of `harness.cu`" (pasting the actual file/line) > a high-level description like "the main memory load".
- **Use NCU's own estimates, checked.** Rule-engine `Est. Speedup: X%` numbers are a starting point — check each against the counters it rests on (small-grid, bank-conflict and L2-compression estimates can be far off), and say when you overrule one.
- **Rank by magnitude.** Fix the 50% problem before the 5% problem.
- **Keep the top-line summary dense.** A reader should be able to get the #1 finding in 10 seconds of reading.
- **Link to artifacts.** Don't paste huge tables into the prose — link to `analysis/compare_<tag1>_vs_<tag2>.txt` etc.

## Anti-patterns

- ❌ Generic advice without evidence ("you might consider using shared memory").
- ❌ More than 5 "priorities" — you're probably padding.
- ❌ Re-running the same profile with different tags and copy-pasting the same analysis — consolidate.
- ❌ Reporting from the CLI table directly. Extract, interpret, write — don't dump.
- ❌ Omitting the setup section. Without it, nobody can reproduce or trust the numbers.
