---
name: ncu-report-skill
description: Use before deciding how to optimize a CUDA kernel (hand-written CUDA, Triton, CUTLASS/CuTe), when a kernel is slower than expected, and whenever you are about to run Nsight Compute (ncu) or read its output — instead of guessing at the bottleneck, writing your own profiling script, or reading ncu's console tables by eye. One ncu collection, then helpers that extract available metrics, rule results, source stalls and supported timelines, followed by evidence-based diagnoses. Also when a user asks to profile a kernel, read an ncu report or write an optimization plan ("profile 一下", "为什么慢", "ncu 报告"). Blackwell guidance targets ncu 2026.3; Volta is supported with ncu 2025.2.
---

# Skill: CUDA Kernel Profiling (Nsight Compute)

**When to use:** before choosing what to optimize in a GPU kernel (CUDA C++, Triton, CUTLASS/CuTe), when a kernel is slower than its budget or baseline, after a change whose effect you do not understand, and every time you are about to run `ncu` or read a report — so the bottleneck comes from measurements, not guesses. Also when someone asks to profile a kernel, find its bottlenecks, read an ncu report or write an optimization plan ("profile X", "为什么这个 kernel 慢", "ncu report 说...", "下一步怎么优化", "帮我看一下这份 ncu 报告").

**Profiler support:** Blackwell examples target Nsight Compute 2026.3.1. Volta (GV100 / Tesla V100 / `sm_70`) requires Nsight Compute 2025.2.x; later releases removed GV100 support. Select the profiler executable explicitly with `NCU` (default `ncu`) and check `"$NCU" --version`, `"$NCU" --help`, `"$NCU" --list-sets`, and `"$NCU" --list-sections`. Flags, sections, metric names, report APIs, and hardware features vary by release. Architecture-specific guidance is identified below; use [`reference/10-volta.md`](reference/10-volta.md) for SM70 diagnosis. The Blackwell companion remains scoped to Blackwell.

---

## Golden rule

**Profile → Diagnose → Plan, in that order. Never guess.**

Most under-performing CUDA kernels are under-performing for exactly one reason that ncu can tell you in 10 seconds. Don't invent hypotheses before you have the report. Don't start coding a fix before you've matched the observed pattern to a known diagnosis. Don't write a wall of suggestions — rank them by evidence and expected impact.

---

## Quickstart (what to do when someone says "profile this kernel")

0. **Create a new run directory first** — **one directory per run**, never reuse an existing one — under the directory your project keeps profiling artifacts in (its README, AGENTS.md or contributing docs may name one; if none does, create one at the repo root and say where in the report). Each run holds its own `harness/`, `reports/`, `analysis/` and `REPORT.md`; the docs call it `$PROFILE_RUN_DIR`. See [`reference/00-directory-layout.md`](reference/00-directory-layout.md).

1. **Decide what you're profiling.** What inputs? Which dispatch path? What question do you want answered? If the kernel takes variable-sized inputs (variable seq lengths, variable batch sizes), you must pick specific representative shapes from the user's workload — don't profile with arbitrary inputs.

2. **Get a profile target.** If the kernel already carries line info (many prebuilt libraries do; [`reference/02-harness-guide.md`](reference/02-harness-guide.md) shows how to check), profile the real code: a few-line driver (Python or C++) that calls it once on a representative input is enough. Build a standalone harness with `-lineinfo` when the build has no line info or the kernel is buried in a slow program; the template is [`helpers/harness_template.cu`](helpers/harness_template.cu). Keep it in `$PROFILE_RUN_DIR/harness/`.

3. **Collect one report** with `--set full`: it requests standard sections, including SourceCounters and PM sampling where the profiler and GPU support them. Volta does not provide the PM timelines used by the Blackwell workflow; this does not prevent other analyses. `--import-source yes` embeds source when available. Give `-o` the `.ncu-rep` extension (without one, newer releases may choose `.ncu-repz`; the helpers read both):
   ```bash
   export NCU="${NCU:-ncu}"  # shared by collection and Python binding discovery
   # Optional explicit clock policy: base on 2025.2, or boost on newer
   # releases when listed by --help. If unset, record the profiler default.
   CLOCK_ARGS=()
   if [[ -n "${CLOCK_CONTROL:-}" ]]; then CLOCK_ARGS=(--clock-control "$CLOCK_CONTROL"); fi
   "$NCU" --set full --import-source yes "${CLOCK_ARGS[@]}" \
       -k "regex:KERNEL" -c 1 -o "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" <command> [args]
   ```
   See [`reference/03-collection.md`](reference/03-collection.md) for picking the launch, skipping warm-up and faster targeted runs.

4. **Parse with the helpers** — not by eye-balling the CLI — into `$PROFILE_RUN_DIR/analysis/`: `analyze_reports.py` (key metrics and NCU's rule results), `extract_stall_hotspots.py` (stalls per source line; `--binary` for inlined CUTLASS/CuTe code), `plot_timeline.py` (PM timelines). See [`helpers/README.md`](helpers/README.md) and [`reference/04-python-api.md`](reference/04-python-api.md).

5. **Work through the six analysis dimensions.** See [`reference/05-analysis-dimensions.md`](reference/05-analysis-dimensions.md). Every one matters, but on any given kernel only 1–2 will dominate.

6. **Match patterns to the diagnosis playbook.** See [`reference/06-diagnosis-playbook.md`](reference/06-diagnosis-playbook.md). It maps NCU signal → likely cause → concrete fix, with example counts for "how big is this".

7. **Write the report** at `$PROFILE_RUN_DIR/REPORT.md` with evidence-backed recommendations, ranked by expected impact. See [`reference/07-report-template.md`](reference/07-report-template.md).

---

## Where the GPU is

Only `ncu` (and the program it launches) needs a GPU. Building the harness for an explicit architecture, `ncu --import`, and every Python helper only read and write files, so they can run on a machine or in a container without one. No helper picks a GPU or locks a clock: run `ncu` on the GPU your environment gives you, the way it runs any GPU command. If GPU commands run in another container or on another machine, keep the run directory where both see it (inside the project, say) and pass paths that exist on both sides.

Select clock control from `"$NCU" --help`: Nsight Compute 2025.2 supports `base`, `none`, and `reset`; modern releases may offer `boost`. To preserve the profiler's default behavior, leave `CLOCK_CONTROL` unset; when you need a fixed, comparable policy, set it to a value supported by that version (for example, `base` for 2025.2 or `boost` on a modern release that lists it). Record the policy and measured clock. See [`reference/03-collection.md`](reference/03-collection.md#gpu-clocks).

---

## File index

### Reference docs (read these when you need details)

| File | Purpose |
|---|---|
| [`reference/00-directory-layout.md`](reference/00-directory-layout.md) | **Read first.** Directory / naming conventions — one run = one subdirectory, no cross-contamination |
| [`reference/01-workflow.md`](reference/01-workflow.md) | End-to-end checklist from "user request" to "final report" |
| [`reference/02-harness-guide.md`](reference/02-harness-guide.md) | Profiling the real code vs. a standalone harness; checking for line info; drivers for Python / prebuilt libraries |
| [`reference/03-collection.md`](reference/03-collection.md) | ncu command recipes: full, targeted, source-only, survey, clocks |
| [`reference/04-python-api.md`](reference/04-python-api.md) | `ncu_report` Python API patterns with copy-pasteable code |
| [`reference/05-analysis-dimensions.md`](reference/05-analysis-dimensions.md) | Six analysis dimensions: occupancy, balance, stalls, tensor core, timeline, memory |
| [`reference/06-diagnosis-playbook.md`](reference/06-diagnosis-playbook.md) | Pattern → diagnosis → fix. Merges Blackwell programming principles with NCU signals |
| [`reference/07-report-template.md`](reference/07-report-template.md) | How to structure the final report |
| [`reference/08-metric-names.md`](reference/08-metric-names.md) | Metric names on Blackwell with ncu 2026.3, and the ones that differ from older docs |
| [`reference/09-common-issues.md`](reference/09-common-issues.md) | Permissions, PM sampling, inlined code, TVM-FFI / PyTorch gotchas |
| [`reference/10-volta.md`](reference/10-volta.md) | Portable SM70 collection limits, supported evidence, and Volta-specific diagnosis |

### Helpers (reusable code)

| File | Purpose |
|---|---|
| [`helpers/harness_template.cu`](helpers/harness_template.cu) | Standalone harness template — paste your kernel, fill in input allocation, done |
| [`helpers/safetensors_loader.h`](helpers/safetensors_loader.h) | Header-only safetensors reader (no external deps) for loading real workload tensors |
| [`helpers/analyze_reports.py`](helpers/analyze_reports.py) | Key metrics, NCU rule results (by estimated speedup), side-by-side comparisons |
| [`helpers/extract_stall_hotspots.py`](helpers/extract_stall_hotspots.py) | Per-line stall aggregation; `--binary` attributes inlined code to the kernel's own lines |
| [`helpers/plot_timeline.py`](helpers/plot_timeline.py) | ASCII PM-sampling timeline plotter — makes tail effect visible |
| [`helpers/list_flashinfer_workloads.py`](helpers/list_flashinfer_workloads.py) | Browse a flashinfer-trace dataset — shape histograms, filter by axis, resolve safetensors paths for specific UUIDs |
| [`helpers/ncu_utils.py`](helpers/ncu_utils.py) | Shared Python helpers: finds `ncu_report`, report loading (every kernel in a report), safe metric access, rules, PM series, inline chains |

---

## Critical lessons (don't skip)

1. **Metric names differ between GPUs and ncu releases.** [`reference/08-metric-names.md`](reference/08-metric-names.md) lists Blackwell names. For Volta, use [`reference/10-volta.md`](reference/10-volta.md), inspect the selected profiler's `--query-metrics`, and enumerate collected values with `action.metric_names()`. A missing metric is unavailable evidence, not a zero measurement.

2. **Always compile with `-lineinfo`** (or profile code that has it). Without it, ncu's source view is blank and you cannot do per-line stall analysis. ncu attributes each PC to its innermost inlined frame — a library header for CUTLASS/CuTe code — so pass the profiled binary to `extract_stall_hotspots.py --binary` to rank the kernel's own lines.

3. **PM sampling timelines depend on hardware and profiler support.** On supported GPUs, `helpers/plot_timeline.py` can show time-varying utilization; Volta does not support these PM-sampling timelines. Missing timelines do not block occupancy, instruction, memory, rule, or source-counter analysis. Do not claim a measured tail shape without a supported timeline or other direct evidence.

4. **Load-imbalance on variable-length inputs is often the #1 bottleneck.** If the user's workload has sequences of varying length, per-SM active-cycle variance will often dwarf every other effect. Always check the input distribution.

5. **NCU's rule engine (`--page details`, or `rules_<tag>.txt` from `analyze_reports.py`) tells you where to look.** Each rule comes with `Est. Speedup: X%`. Read them first, then check each against the counters before ranking it: estimates for small grids, shared-memory bank conflicts or L2 compression can be far off.

6. **Don't delegate understanding.** Run the profiles yourself, open the reports, cite specific metric values. Never write "the profile shows it's memory-bound" — instead, name the two or three metric values that back your conclusion (e.g., "`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` well under 10%, and `long_scoreboard` stalls dominate the pcsamp histogram, so the kernel is **latency-bound on L1**, not DRAM-bandwidth-bound"). Fill in the actual numbers from your report. Specificity is the deliverable.

---

## Related skills

- [`blackwell-cuda-programming.md`](blackwell-cuda-programming.md) — Blackwell-specific programming principles and checklists, preserved as a companion reference. Use it when proposing *new* kernel designs; use this skill when diagnosing *existing* kernels.
