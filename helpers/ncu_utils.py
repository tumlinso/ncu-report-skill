"""Shared helpers for parsing reports from multiple Nsight Compute generations.

Usage:
    from ncu_utils import load_report, select_actions, safe, dump_all_metrics

`ncu_report` ships with Nsight Compute under `<install>/extras/python`. Standard
Python import resolution is tried first. Otherwise the selected profiler
(`NCU_BIN`, `NCU`, or `ncu` on PATH), CUDA roots, and NVIDIA HPC SDK roots are
used for bounded discovery. `NCU_REPORT_PYTHON_PATH` can name the bindings
directory explicitly. `python3 ncu_utils.py` prints the selected locations.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path


# --- Locate ncu_report -------------------------------------------------------

def _version(path: Path) -> tuple[int, ...]:
    """Numeric path components, used only to order sibling profiler installs."""
    return tuple(int(x) for x in re.findall(r"\d+", str(path)))


def selected_ncu_executable(environ=None) -> str | None:
    """Resolve the selected profiler executable, honoring NCU_BIN before NCU."""
    env = os.environ if environ is None else environ
    selected = env.get("NCU_BIN") or env.get("NCU") or "ncu"
    path = Path(selected).expanduser()
    if path.is_file():
        return str(path.resolve())
    return shutil.which(selected)


def _has_ncu_report(directory: Path) -> bool:
    try:
        return ((directory / "ncu_report.py").is_file()
                or (directory / "ncu_report" / "__init__.py").is_file())
    except OSError:
        return False


def _candidate_ncu_python_dirs(ncu_path=None, environ=None) -> list[Path]:
    """Return likely binding directories, preferring the selected profiler install.

    Searches executable ancestors, CUDA/HPC environment roots, and bounded common
    installation layouts. It does not recursively walk arbitrary filesystems.
    """
    env = os.environ if environ is None else environ
    explicit_ncu = env.get("NCU_BIN") or env.get("NCU")
    executable = ncu_path if ncu_path is not None else selected_ncu_executable(env)
    # An explicitly selected but unresolved profiler must not lead to bindings
    # being borrowed from a different installation found in a generic root.
    if ncu_path is None and explicit_ncu and executable is None:
        return []
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path):
        path = path.expanduser()
        try:
            path = path.resolve()
        except OSError:
            pass
        if path not in seen and _has_ncu_report(path):
            seen.add(path)
            candidates.append(path)

    def add_root(root: Path):
        add(root / "extras" / "python")
        add(root / "Nsight_Compute" / "extras" / "python")
        add(root / "nsight-compute" / "extras" / "python")
        patterns = (
            "*", "nsight-compute-*", "*/nsight-compute-*",
            "profilers/*/Nsight_Compute", "*/profilers/*/Nsight_Compute",
            "*/*/profilers/*/Nsight_Compute",
        )
        installs = []
        for pattern in patterns:
            try:
                installs.extend(root.glob(pattern))
            except OSError:
                continue
        for install in sorted(set(installs), key=_version, reverse=True):
            add(install / "extras" / "python")

    if executable:
        exe = Path(executable).expanduser()
        try:
            exe = exe.resolve()
        except OSError:
            pass
        # Standalone `ncu`, Toolkit `bin/ncu`, and HPC `compilers/bin/ncu`.
        try:
            temporary_root = Path(tempfile.gettempdir()).resolve()
        except OSError:
            temporary_root = None
        for ancestor in list(exe.parents)[:4]:
            if ancestor != Path("/") and ancestor != temporary_root:
                add_root(ancestor)

    for name in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "NVHPC", "NVCOMPILERS"):
        value = env.get(name)
        if value:
            add_root(Path(value))

    # Common package-manager and Toolkit prefixes, using bounded globs only.
    add_root(Path("/opt/nvidia/nsight-compute"))
    add_root(Path("/usr/local"))
    try:
        roots = sorted(Path("/usr/local").glob("cuda*"), key=_version, reverse=True)
    except OSError:
        roots = []
    for root in roots:
        add_root(root)
    return candidates


def ncu_python_dir() -> Path | None:
    """Return the discovered Nsight Compute Python binding directory, if any."""
    candidates = _candidate_ncu_python_dirs()
    return candidates[0] if candidates else None


def _import_from_directory(directory: Path):
    path = str(directory.resolve())
    sys.path.insert(0, path)
    try:
        return importlib.import_module("ncu_report")
    except ImportError as exc:
        raise ImportError(
            f"Unable to import ncu_report from {path}: {exc}. Check that the Nsight "
            "Compute Python bindings match this Python interpreter."
        ) from exc


def _load_ncu_report():
    """Import ncu_report via normal Python resolution, override, then discovery."""
    try:
        return importlib.import_module("ncu_report")
    except ModuleNotFoundError as exc:
        if exc.name != "ncu_report":
            raise ImportError(
                f"Python found ncu_report but could not load one of its dependencies: {exc}. "
                "Check that PYTHONPATH points to bindings compatible with this interpreter."
            ) from exc
    except ImportError as exc:
        raise ImportError(
            f"Python found ncu_report but could not import it: {exc}. Check that PYTHONPATH "
            "points to bindings compatible with this interpreter."
        ) from exc

    override = os.environ.get("NCU_REPORT_PYTHON_PATH")
    if override is not None:
        path = Path(override).expanduser()
        if not path.is_dir() or not _has_ncu_report(path):
            raise ImportError(
                "NCU_REPORT_PYTHON_PATH is set but does not contain ncu_report: "
                f"{path}. Set it to the Nsight Compute extras/python directory."
            ) from None
        return _import_from_directory(path)

    found = ncu_python_dir()
    if found is None:
        selected = (os.environ.get("NCU_BIN") or os.environ.get("NCU")
                    or selected_ncu_executable() or "not found on PATH")
        raise ImportError(
            "ncu_report is not importable and no Nsight Compute bindings were found "
            f"for the selected profiler ({selected}). Set NCU_REPORT_PYTHON_PATH or "
            "add <Nsight Compute install>/extras/python to PYTHONPATH. Select the "
            "profiler with NCU_BIN or NCU."
        ) from None
    return _import_from_directory(found)


ncu_report = _load_ncu_report()


# --- Loading -----------------------------------------------------------------

def actions(report):
    """Every action (= one profiled kernel launch) of a loaded report, in order."""
    out = []
    for r in range(report.num_ranges()):
        rng = report.range_by_idx(r)
        out.extend(rng.action_by_idx(a) for a in range(rng.num_actions()))
    return out


def load_report(path):
    """Load a report (.ncu-rep, or 2026.3's .ncu-repz) and return (report, first action).

    Keep the report alive while using the action.
    """
    r = ncu_report.load_report(str(path))
    return r, actions(r)[0]


def load_action(path):
    """Shortcut when you don't need the report object separately."""
    _, action = load_report(path)
    return action


def select_actions(path, kernel=None):
    """(report, [actions]) of a report, keeping those whose function or demangled name matches `kernel`."""
    r = ncu_report.load_report(str(path))
    acts = actions(r)
    if kernel:
        pat = re.compile(kernel)
        acts = [a for a in acts
                if pat.search(kernel_name(a, "function")) or pat.search(kernel_name(a))]
    return r, acts


def kernel_name(action, form="demangled"):
    """The kernel's name: "demangled" (with template arguments), "function" or "mangled"."""
    base = {"demangled": action.NameBase_DEMANGLED, "function": action.NameBase_FUNCTION,
            "mangled": action.NameBase_MANGLED}[form]
    return action.name(base)


def action_tags(tag, acts):
    """One output tag per action: `tag` alone for one action, else `tag.<i>_<function>`.

    The function name is cut to 60 characters of [A-Za-z0-9_.-], so file names stay short.
    """
    if len(acts) == 1:
        return [tag]
    return [f"{tag}.{i}_{re.sub(r'[^A-Za-z0-9_.-]', '_', kernel_name(a, 'function'))[:60]}"
            for i, a in enumerate(acts)]


# --- Safe metric access ------------------------------------------------------

def safe(action, name, default=None):
    """Return metric value, or `default` if the metric is missing or errors."""
    try:
        return action[name].value()
    except Exception:
        return default


def safe_many(action, names, default=None):
    """Bulk-fetch multiple metrics. Returns a dict name -> value-or-default."""
    return {n: safe(action, n, default) for n in names}


def metric_or_none(action, *candidates):
    """Try each candidate name, return first that works. Useful for
    GPU-gen-specific names: some metric names differ on sm_100 vs sm_90."""
    for n in candidates:
        v = safe(action, n, None)
        if v is not None:
            return v
    return None


# --- Value-kind robust accessor (for per-instance data) ---------------------

def metric_value_at(m, i):
    """Read the i-th instance value regardless of value kind."""
    k = m.kind()
    if k == m.ValueKind_UINT64:
        return m.as_uint64(i)
    if k in (m.ValueKind_DOUBLE, m.ValueKind_FLOAT):
        return m.as_double(i)
    if k == m.ValueKind_STRING:
        return m.as_string(i)
    # Fallbacks
    try:
        return m.as_uint64(i)
    except Exception:
        try:
            return m.as_double(i)
        except Exception:
            return None


def per_instance_values(action, metric_name):
    """Return a list of per-instance values, or None if the metric has none."""
    try:
        m = action[metric_name]
    except Exception:
        return None
    try:
        n = m.num_instances()
    except Exception:
        return None
    if n == 0:
        return None
    return [metric_value_at(m, i) for i in range(n)]


# --- Archive all metrics -----------------------------------------------------

def dump_all_metrics(action, outfile):
    """Dump every metric name + value to a JSON file for later analysis.

    Returns the number of entries written.
    """
    out = []
    for n in sorted(action.metric_names()):
        try:
            m = action[n]
            rec = {"name": n}
            try:
                rec["value"] = m.value()
            except Exception as e:
                rec["error"] = str(e)
            try:
                rec["unit"] = m.unit()
            except Exception:
                pass
            out.append(rec)
        except Exception as e:
            out.append({"name": n, "error": str(e)})
    Path(outfile).write_text(json.dumps(out, indent=1, default=str))
    return len(out)


# --- PC → source line mapping ------------------------------------------------

def per_pc_values(action, metric_name):
    """For a source-level metric (with correlation_ids = PCs), return list of (pc, value)."""
    try:
        m = action[metric_name]
    except Exception:
        return []
    try:
        n = m.num_instances()
    except Exception:
        return []
    if n == 0 or not m.has_correlation_ids():
        return []
    cor = m.correlation_ids()
    out = []
    for i in range(n):
        try:
            pc = cor.as_uint64(i)
        except Exception:
            try:
                pc = int(cor.as_double(i))
            except Exception:
                pc = None
        try:
            v = metric_value_at(m, i)
        except Exception:
            # An unreadable sample is missing evidence, not a measured zero.
            continue
        if v is None:
            continue
        out.append((pc, v))
    return out


def pc_to_source_line(action, pc):
    """Return (file, line) for a given PC, or ('?', 0) if unavailable.

    Requires -lineinfo at compile time.
    """
    try:
        si = action.source_info(pc)
        if si is None:
            return "?", 0
        return si.file_name(), si.line()
    except Exception:
        return "?", 0


# --- Curated metric sets -----------------------------------------------------
#
# Curated names span several architectures. A metric that is not collected by a
# report stays `None`; the full archive preserves every metric the report has.
# For a fuller list and rationale see ../reference/08-metric-names.md.

KEY_METRICS = [
    # Launch geometry
    "launch__grid_size",
    "launch__block_size",
    "launch__grid_dim_x",
    "launch__grid_dim_y",
    "launch__grid_dim_z",
    "launch__block_dim_x",
    "launch__waves_per_multiprocessor",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__thread_count",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "device__attribute_multiprocessor_count",
    "device__attribute_max_warps_per_multiprocessor",
    # Timing and the clock the kernel ran at (ncu's --clock-control)
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg.per_second",
    "smsp__cycles_active.avg",
    # SOL
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    # Occupancy
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "sm__warps_active.max.per_cycle_active",
    "sm__warps_active.min.per_cycle_active",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_active.avg.peak_sustained",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warps_eligible.max.per_cycle_active",
    # IPC
    "sm__inst_executed.avg.per_cycle_active",
    "smsp__issue_active.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__issue_inst0.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.avg",
    # Compute pipes
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    # FP16/FP64 pipe views exposed by the 2025.2 ComputeWorkloadAnalysis section.
    "sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active",
    "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active",
    # Tensor core
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    # Versioned HMMA views remain separate because their definitions can differ.
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_op_hmma_cycles_active_v2.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor_op_hmma_v2.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor_subpipe_hmma.avg.pct_of_peak_sustained_active",
    # Shared-memory SourceCounters views, when that source-level data is exposed.
    "derived__memory_l1_wavefronts_shared_excessive",
    "derived__memory_l1_conflicts_shared_nway",
    "memory_l1_wavefronts_shared",
    "memory_l1_wavefronts_shared_ideal",
    # SharedMemoryConflicts section keeps load and store counters distinct.
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg",
    "sm__ops_path_tensor_op_utchmma_src_bf16_dst_fp32_sparsity_off.avg.pct_of_peak_sustained_elapsed",
    "sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__sass_inst_executed_op_utcmma.sum",
    # DRAM
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",   # reads + writes
    "dram__bytes_read.sum",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum",
    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    # Caches
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct",
    # Memory instruction counts
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__sass_inst_executed_op_shared.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
    # Sectors / requests (for coalescing analysis)
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio",
    # Stall reasons — aggregate ratios
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_misc_per_issue_active.ratio",
    # Stall reasons — per-PC (SourceCounters section, part of --set full)
    "smsp__pcsamp_sample_count",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_wait",
    "smsp__pcsamp_warps_issue_stalled_barrier",
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
    "smsp__pcsamp_warps_issue_stalled_mio_throttle",
    "smsp__pcsamp_warps_issue_stalled_lg_throttle",
    "smsp__pcsamp_warps_issue_stalled_not_selected",
    "smsp__pcsamp_warps_issue_stalled_dispatch_stall",
    "smsp__pcsamp_warps_issue_stalled_drain",
    "smsp__pcsamp_warps_issue_stalled_no_instructions",
    "smsp__pcsamp_warps_issue_stalled_selected",
    "smsp__pcsamp_warps_issue_stalled_branch_resolving",
    "smsp__pcsamp_warps_issue_stalled_membar",
    "smsp__pcsamp_warps_issue_stalled_sleeping",
    "smsp__pcsamp_warps_issue_stalled_misc",
    "smsp__pcsamp_warps_issue_stalled_tex_throttle",
]
B200_KEY_METRICS = KEY_METRICS  # the name older scripts import


# --- NCU rule results --------------------------------------------------------

_SEVERITY = {0: "INF", 1: "OK", 2: "OPT", 3: "WRN", 4: "ERR"}  # ncu_report.MsgType_*
_SPEEDUP_TYPE = {1: "local", 2: "global"}                      # ncu_report.SpeedupType_*


def _link(m):
    kind, _, rest = m.group(1, 2, 3)
    if kind == "url":  # @url:<text>:<address>@
        text, _, address = rest.partition(":")
        return f"{text} ({address})" if address else text
    return rest.rsplit(":", 1)[-1]  # @section:<id>:<title>@ and the like: the title


def _plain(message):
    """Rule text with ncu's "@<kind>:...@" link markup turned into plain text."""
    return re.sub(r"@(\w+)(:)([^@]*)@", _link, message or "")


def rule_speedups(action):
    """NCU rule-engine results as dicts, highest estimated speedup first.

    Keys: speedup (percent, None when the rule gives none), speedup_type ("local": of
    the part the rule is about, "global": of the whole kernel), severity
    (OPT/WRN/INF/OK/ERR), section, rule, title, message.
    """
    return rule_speedups_result(action)[0]


def rule_speedups_result(action):
    """Return ``(results, status, error)`` for optional report rule processing.

    Status is ``available``, ``no_results``, ``unsupported`` or ``error``. The
    report API does not distinguish an untriggered rule from omitted rule data;
    ``no_results`` therefore makes no claim about why the list is empty.
    """
    method = getattr(action, "rule_results", None)
    if not callable(method):
        return [], "unsupported", None
    try:
        report_results = method()
    except Exception as exc:
        return [], "error", str(exc)
    if report_results is None:
        return [], "no_results", None
    out = []
    try:
        for rr in report_results:
            est = rr.speedup_estimation() if rr.has_speedup_estimation() else {}
            msg = rr.rule_message() if rr.has_rule_message() else {}
            if not msg.get("message") and not est:
                continue
            out.append({
                "speedup": est.get("speedup"),
                "speedup_type": _SPEEDUP_TYPE.get(est.get("type")),
                "severity": _SEVERITY.get(msg.get("type"), "?"),
                "section": rr.section_identifier(),
                "rule": rr.rule_identifier(),
                "title": msg.get("title", ""),
                "message": _plain(msg.get("message", "")),
            })
    except Exception as exc:
        return [], "error", str(exc)
    out.sort(key=lambda r: -(r["speedup"] or 0.0))
    return out, ("available" if out else "no_results"), None


def action_architecture(action) -> str:
    """Return a stable ``sm_<major><minor>`` label, or ``unknown``."""
    major = safe(action, "device__attribute_compute_capability_major")
    minor = safe(action, "device__attribute_compute_capability_minor")
    try:
        return f"sm_{int(major)}{int(minor)}"
    except (TypeError, ValueError):
        return "unknown"


def timed_warp_sample_info(action):
    """Return ``(status, count, error)`` for the optional timed warp-sample API."""
    status, samples, error = _timed_warp_samples(action)
    return status, (len(samples) if samples is not None else None), error


def _timed_warp_samples(action):
    """Read optional timed warp samples and keep unsupported distinct from empty."""
    method = getattr(action, "timed_warp_samples", None)
    if not callable(method):
        return "unsupported", None, None
    try:
        samples = method()
    except Exception as exc:
        return "error", None, str(exc)
    if not samples:
        return "no_data", [], None
    return "available", samples, None


def report_metadata(report, action, rule_summary=None):
    """Summarize report identity and which evidence is present for an action."""
    try:
        metric_names = set(action.metric_names())
    except Exception:
        metric_names = set()
    try:
        reader_path = str(Path(ncu_report.__file__).resolve())
    except (AttributeError, OSError):
        reader_path = "ncu_report (Python import)"
    get_version = getattr(report, "get_version", None) if report is not None else None
    try:
        reader_version = str(get_version()) if callable(get_version) else None
    except Exception:
        reader_version = None
    source_metrics = sorted(n for n in metric_names if n.startswith("smsp__pcsamp_"))
    warp_status, warp_count, warp_error = timed_warp_sample_info(action)
    if rule_summary is None:
        rule_results, rule_status, rule_error = rule_speedups_result(action)
    else:
        rule_results, rule_status, rule_error = rule_summary
    return {
        "architecture": action_architecture(action),
        # The reader API does not document a report-producer version field.
        # Keep this unknown instead of relabeling the reader/library version.
        "producer_profiler_version": None,
        "producer_profiler_version_status": "unknown; not exposed as report metadata",
        "reader_api": {"module": "ncu_report", "path": reader_path,
                        "version": reader_version},
        "measurements": {
            "report_metric_count": len(metric_names),
            "curated_metric_count": len(KEY_METRICS),
            "curated_metrics_present": sum(name in metric_names for name in KEY_METRICS),
            "curated_metrics_missing": sum(name not in metric_names for name in KEY_METRICS),
            "source_stall_metrics_present": source_metrics,
            "pm_sampling_series_count": len(pm_series_names(action)),
            "timed_warp_samples": {"status": warp_status, "count": warp_count,
                                   "error": warp_error},
            "rule_results": {"status": rule_status, "count": len(rule_results),
                             "error": rule_error},
        },
    }


# --- PM sampling time series -----------------------------------------------------

def pm_series_names(action):
    """Names of the PM-sampling time series in a report, sorted.

    The PmSampling section names them `<UNIT>.<group>.<metric>` (e.g.
    `TPC.TriageCompute.sm__inst_executed_realtime.avg.per_cycle_elapsed`); reports of ncu
    before 2026.3 may also have `pmsampling:smsp__warps_issue_stalled_*` (the
    PmSampling_WarpStates section, which 2026.3 removed).
    """
    pat = re.compile(r"^(pmsampling:|[A-Z][A-Z0-9_]*\.\w+\.)")
    out = []
    for n in action.metric_names():
        if pat.match(n):
            try:
                if action[n].num_instances() > 1:
                    out.append(n)
            except Exception:
                pass
    return sorted(out)


def pm_series_result(action, name):
    """Return ``(series, status, error)`` without requiring timeline APIs.

    Legacy report APIs may provide useful unaligned PM samples. If optional
    timeline alignment is missing or fails, the raw series is returned.
    """
    try:
        names = set(action.metric_names())
    except Exception as exc:
        return None, "error", str(exc)
    if name not in names:
        return None, "missing", None
    try:
        metric = action[name]
    except Exception as exc:
        return None, "error", str(exc)

    status, error = "raw", None
    timelines_method = getattr(action, "timelines", None)
    if callable(timelines_method):
        try:
            timelines = timelines_method()
        except Exception as exc:
            timelines, status, error = None, "alignment_error", str(exc)
        align = getattr(timelines, "aligned_metric", None) if timelines is not None else None
        if callable(align):
            try:
                aligned = align(metric)
                if aligned is not None:
                    metric, status = aligned, "aligned"
            except Exception as exc:
                status, error = "alignment_error", str(exc)

    try:
        count = metric.num_instances()
        if count < 2:
            return None, "no_data", error
        values = [metric_value_at(metric, i) for i in range(count)]
        timestamps = None
        if metric.has_correlation_ids():
            correlations = metric.correlation_ids()
            timestamps = [correlations.as_uint64(i) for i in range(count)]
        return (timestamps, values), status, error
    except Exception as exc:
        return None, "error", str(exc)


def pm_series(action, name):
    """(timestamps_ns, values) of one PM-sampling series, or None."""
    return pm_series_result(action, name)[0]


def stall_reason_names():
    """ncu_report's StallReason codes -> names (long_scoreboard, wait, ...)."""
    return {getattr(ncu_report, k): k[len("StallReason_"):].lower()
            for k in dir(ncu_report) if k.startswith("StallReason_")}


def warp_state_timeline_result(action, buckets):
    """Stall reasons over the kernel's run, from its timed warp samples (PmSampling).

    Return ``(timeline, status, error)`` so missing API, absent samples and errors
    remain distinct for callers that want to explain unavailable analysis.
    """
    status, samples, error = _timed_warp_samples(action)
    if status != "available":
        return None, status, error
    if buckets <= 0:
        return None, "error", "bucket count must be positive"
    try:
        t0 = min(sample["timestamp"] for sample in samples)
        span = max(sample["timestamp"] for sample in samples) - t0 or 1
        names = stall_reason_names()
        counts, totals = {}, [0] * buckets
        for sample in samples:
            bucket = min(buckets - 1, (sample["timestamp"] - t0) * buckets // span)
            totals[bucket] += 1
            reason = names.get(sample["stall_reason"], str(sample["stall_reason"]))
            counts.setdefault(reason, [0] * buckets)[bucket] += 1
        timeline = (span, {reason: [count / total if total else 0.0
                                   for count, total in zip(values, totals)]
                           for reason, values in counts.items()})
        return timeline, status, None
    except Exception as exc:
        return None, "error", str(exc)


def warp_state_timeline(action, buckets):
    """Return a timed stall timeline, or None when unsupported or unavailable."""
    return warp_state_timeline_result(action, buckets)[0]


# --- Inlined source lines (for CUTLASS/CuTe-style kernels) ---------------------

_FILE_RE = re.compile(r'//## File "([^"]+)", line (\d+)')
_INST_RE = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;?\s*$")


def _opcode(sass):
    words = sass.split()
    if words and words[0].startswith("@"):
        words = words[1:]
    return words[0].split(".")[0] if words else ""


class BinaryError(Exception):
    """The binary given for inline chains cannot be read (missing, or no device code)."""


_extracted = {}    # (binary, arch) -> cubin paths
_disassembly = {}  # cubin path -> nvdisasm -gi text


def _run(cmd, **kw):
    import subprocess
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise BinaryError(f"{' '.join(cmd[:2])} {cmd[-1]}: " + (r.stderr or r.stdout).strip())
    return r.stdout


def _cubins(binary, arch):
    """The cubins of `binary` for sm_<arch>: that architecture's if any, else its family's, else all."""
    import atexit
    import shutil as _shutil
    import tempfile

    key = (binary, arch)
    if key not in _extracted:
        if binary.suffix == ".cubin":
            _extracted[key] = [binary]
            return _extracted[key]
        names = re.findall(r"ELF file\s+\d+:\s+(\S+)", _run(["cuobjdump", "-lelf", str(binary)]))
        if not names:
            raise BinaryError(f"{binary} holds no cubin (cuobjdump -lelf lists none)")
        chosen = ([n for n in names if f".sm_{arch}" in n]
                  or [n for n in names if f".sm_{arch[:-1]}" in n] or names)
        tmp = tempfile.mkdtemp(prefix="ncu_utils_cubins_")
        atexit.register(_shutil.rmtree, tmp, True)
        _run(["cuobjdump", "-xelf", ",".join(chosen), str(binary)], cwd=tmp)
        _extracted[key] = sorted(Path(tmp).glob("*.cubin"))
    return _extracted[key]


def inline_chains(action, binary):
    """Map each PC of the kernel to its inline chain, from the binary that was profiled.

    `action.source_info(pc)` gives one frame, mostly the innermost inlined one (a library header
    line for CUTLASS/CuTe code); `nvdisasm -gi` gives every frame. `binary` is a cubin
    or anything `cuobjdump -xelf` can extract one from (executable, .so, fatbin); only the
    cubins of the report's GPU architecture that define the kernel are disassembled, once.

    Returns (chains, checked, mismatched): chains maps pc -> [(file, line), ...] from
    the innermost frame to the outermost, which is the line in the kernel function
    itself; checked/mismatched count the sampled PCs whose SASS was compared with the
    report's (a PC outside the kernel's own code, such as a non-inlined device call,
    counts as mismatched). Needs cuobjdump and nvdisasm on PATH; raises BinaryError
    when the binary cannot be read.
    """
    binary = Path(binary).resolve()
    if not binary.is_file():
        raise BinaryError(f"{binary} does not exist")
    mangled = kernel_name(action, "mangled")
    arch = (f"{int(action['device__attribute_compute_capability_major'].value())}"
            f"{int(action['device__attribute_compute_capability_minor'].value())}")
    # launch__function_pcs: one instance per function of the launch, the kernel first
    base = int(metric_value_at(action["launch__function_pcs"], 0))
    sampled = [pc for pc, _ in per_pc_values(action, "smsp__pcsamp_sample_count")]
    best = ({}, 0, 0)
    for cubin in _cubins(binary, arch):
        if mangled.encode() not in cubin.read_bytes():
            continue
        if cubin not in _disassembly:
            _disassembly[cubin] = _run(["nvdisasm", "-gi", str(cubin)])
        insts, chain, in_func, block = {}, [], False, False
        for line in _disassembly[cubin].splitlines():
            if line.startswith("\t.section\t"):  # any section ends the kernel's code
                in_func = (line.startswith("\t.section\t.text.")
                           and line.split(".text.", 1)[1].split(",")[0] == mangled)
                chain, block = [], False
                continue
            if not in_func:
                continue
            m = _FILE_RE.search(line)
            if m:
                if not block:
                    chain = []
                chain.append((m.group(1), int(m.group(2))))
                block = True
                continue
            block = False
            mi = _INST_RE.match(line)
            if mi:
                insts[base + int(mi.group(1), 16)] = (mi.group(2), list(chain))
        if not insts:
            continue
        checked = [pc for pc in sampled if pc in insts]
        mismatched = sum(_opcode(insts[pc][0]) != _opcode(action.sass_by_pc(pc))
                         for pc in checked) + len(sampled) - len(checked)
        if not best[0] or mismatched < best[2]:
            best = ({pc: c for pc, (_, c) in insts.items()}, len(sampled), mismatched)
    return best


if __name__ == "__main__":
    print(f"ncu_report: {Path(ncu_report.__file__).parent}")
    print(f"ncu: {selected_ncu_executable() or 'not found'}")
