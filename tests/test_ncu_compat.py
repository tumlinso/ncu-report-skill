"""Focused compatibility checks; these run without an installed profiler or GPU."""
from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
HELPERS = REPO / "helpers"
sys.path.insert(0, str(HELPERS))

# Keep the unit tests independent of locally installed NVIDIA Python bindings.
fake_module = types.ModuleType("ncu_report")
fake_module.__file__ = "test-ncu_report.py"
sys.modules.setdefault("ncu_report", fake_module)

import ncu_utils  # noqa: E402
import extract_stall_hotspots  # noqa: E402
import plot_timeline  # noqa: E402


class Correlations:
    def __init__(self, values):
        self.values = values

    def as_uint64(self, index):
        return self.values[index]


class Metric:
    ValueKind_UINT64 = 0
    ValueKind_DOUBLE = 1
    ValueKind_FLOAT = 2
    ValueKind_STRING = 3

    def __init__(self, values, timestamps=None, unit="count"):
        self.values = values
        self.timestamps = timestamps
        self._unit = unit

    def value(self):
        return self.values[0] if len(self.values) == 1 else self.values

    def unit(self):
        return self._unit

    def kind(self):
        return self.ValueKind_UINT64

    def num_instances(self):
        return len(self.values)

    def as_uint64(self, index):
        return self.values[index]

    def has_correlation_ids(self):
        return self.timestamps is not None

    def correlation_ids(self):
        return Correlations(self.timestamps)


class Action:
    NameBase_DEMANGLED = 0
    NameBase_FUNCTION = 1
    NameBase_MANGLED = 2

    def __init__(self, metrics=None, **attrs):
        self.metrics = dict(metrics or {})
        self.attrs = attrs

    def metric_names(self):
        return list(self.metrics)

    def __getitem__(self, name):
        if name in self.metrics:
            return self.metrics[name]
        if name in self.attrs:
            return Metric([self.attrs[name]])
        raise KeyError(name)


class DiscoveryTests(unittest.TestCase):
    def run_import(self, env):
        return subprocess.run(
            [sys.executable, "-c", "import ncu_utils; import ncu_report; print(ncu_report.__file__)"],
            cwd=REPO,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def base_env(self):
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(HELPERS)}
        return env

    @staticmethod
    def fake_binding(path: Path, identity="fixture"):
        path.mkdir(parents=True, exist_ok=True)
        (path / "ncu_report.py").write_text(f"IDENTITY = {identity!r}\n")

    @staticmethod
    def fake_ncu(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n")

    def test_selected_install_discovery_for_standalone_toolkit_and_hpc_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cases = []

            standalone = base / "standalone" / "2025.2"
            self.fake_ncu(standalone / "bin" / "ncu")
            self.fake_binding(standalone / "extras" / "python", "standalone")
            cases.append((standalone / "bin" / "ncu", {},
                          standalone / "extras" / "python" / "ncu_report.py"))

            toolkit = base / "cuda-toolkit"
            self.fake_ncu(toolkit / "bin" / "ncu")
            self.fake_binding(toolkit / "nsight-compute-2025.2" / "extras" / "python", "toolkit")
            cases.append((toolkit / "bin" / "ncu", {},
                          toolkit / "nsight-compute-2025.2" / "extras" / "python" / "ncu_report.py"))

            hpc_root = base / "hpc-sdk"
            hpc_version = hpc_root / "Linux_x86_64" / "26.1"
            self.fake_ncu(hpc_version / "compilers" / "bin" / "ncu")
            self.fake_binding(hpc_version / "profilers" / "12.9" / "Nsight_Compute" / "extras" / "python",
                              "hpc")
            cases.append((hpc_version / "compilers" / "bin" / "ncu", {"NVCOMPILERS": str(hpc_root)},
                          hpc_version / "profilers" / "12.9" / "Nsight_Compute" / "extras" /
                          "python" / "ncu_report.py"))

            for executable, extra_env, expected_module in cases:
                with self.subTest(layout=str(executable)):
                    env = self.base_env()
                    env.update(extra_env)
                    env["NCU_BIN"] = str(executable)
                    result = self.run_import(env)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(Path(result.stdout.strip()), expected_module)

    def test_python_import_precedes_auto_discovery_and_report_path_is_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            normal = base / "normal-python"
            normal.mkdir()
            (normal / "ncu_report.py").write_text("IDENTITY = 'normal'\n")
            install = base / "selected" / "ncu"
            self.fake_ncu(install)
            binding = base / "selected" / "extras" / "python"
            self.fake_binding(binding, "discovered")

            env = self.base_env()
            env["PYTHONPATH"] = os.pathsep.join((str(HELPERS), str(normal)))
            env["NCU_BIN"] = str(install)
            result = self.run_import(env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(normal), result.stdout)

            explicit = base / "explicit" / "extras" / "python"
            self.fake_binding(explicit, "explicit")
            env["NCU_REPORT_PYTHON_PATH"] = str(explicit)
            env["PYTHONPATH"] = str(HELPERS)
            result = self.run_import(env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(explicit), result.stdout)

            env["NCU_REPORT_PYTHON_PATH"] = str(base / "missing")
            result = self.run_import(env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("NCU_REPORT_PYTHON_PATH is set", result.stderr)

    def test_broken_standard_import_and_invalid_profiler_do_not_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            normal = base / "normal-python"
            normal.mkdir()
            (normal / "ncu_report.py").write_text("raise ImportError('broken standard binding')\n")
            selected = base / "ncu"
            self.fake_ncu(selected)
            self.fake_binding(base / "extras" / "python")

            env = self.base_env()
            env["PYTHONPATH"] = os.pathsep.join((str(HELPERS), str(normal)))
            env["NCU_BIN"] = str(selected)
            result = self.run_import(env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("found ncu_report but could not import", result.stderr)

            env = self.base_env()
            env["NCU_BIN"] = str(base / "missing-ncu")
            env["CUDA_HOME"] = str(base)
            self.assertEqual(ncu_utils._candidate_ncu_python_dirs(environ=env), [])


class ReportApiTests(unittest.TestCase):
    def test_missing_and_zero_metrics_remain_distinct(self):
        action = Action({"measured_zero": Metric([0])})
        self.assertEqual(ncu_utils.safe(action, "measured_zero"), 0)
        self.assertIsNone(ncu_utils.safe(action, "not_collected"))

        class Report:
            def get_version(self):
                return "reader-api-version"

        metadata = ncu_utils.report_metadata(
            Report(), action, rule_summary=([], "unsupported", None)
        )
        self.assertIsNone(metadata["producer_profiler_version"])
        self.assertEqual(metadata["reader_api"]["version"], "reader-api-version")

    def test_optional_rule_api_distinguishes_unsupported_empty_and_error(self):
        class NoRules:
            pass

        class EmptyRules:
            def rule_results(self):
                return []

        class BrokenRules:
            def rule_results(self):
                raise RuntimeError("bad rule data")

        self.assertEqual(ncu_utils.rule_speedups_result(NoRules()), ([], "unsupported", None))
        self.assertEqual(ncu_utils.rule_speedups_result(EmptyRules()), ([], "no_results", None))
        result, status, error = ncu_utils.rule_speedups_result(BrokenRules())
        self.assertEqual((result, status), ([], "error"))
        self.assertIn("bad rule data", error)

    def test_legacy_raw_and_modern_aligned_pm_series(self):
        name = "pmsampling:SM.sm__cycles_active.avg"
        raw = Metric([1, 2], [10, 20])
        legacy = Action({name: raw})
        self.assertEqual(ncu_utils.pm_series_result(legacy, name), (([10, 20], [1, 2]), "raw", None))

        aligned = Metric([7, 8], [30, 40])

        class Timeline:
            def aligned_metric(self, metric):
                return aligned

        class Modern(Action):
            def timelines(self):
                return Timeline()

        modern = Modern({name: raw})
        self.assertEqual(ncu_utils.pm_series_result(modern, name), (([30, 40], [7, 8]), "aligned", None))

        class BrokenTimeline:
            def aligned_metric(self, metric):
                raise RuntimeError("alignment failed")

        class ModernBroken(Action):
            def timelines(self):
                return BrokenTimeline()

        fallback, status, error = ncu_utils.pm_series_result(ModernBroken({name: raw}), name)
        self.assertEqual(fallback, ([10, 20], [1, 2]))
        self.assertEqual(status, "alignment_error")
        self.assertIn("alignment failed", error)

    def test_timed_warp_samples_capability_and_modern_timeline(self):
        class NoSamples:
            pass

        self.assertEqual(ncu_utils.timed_warp_sample_info(NoSamples()), ("unsupported", None, None))
        self.assertIsNone(ncu_utils.warp_state_timeline(NoSamples(), 4))

        ncu_utils.ncu_report.StallReason_long_scoreboard = 0

        class Modern:
            def timed_warp_samples(self):
                return [
                    {"timestamp": 10, "stall_reason": 0},
                    {"timestamp": 20, "stall_reason": 0},
                ]

        timeline, status, error = ncu_utils.warp_state_timeline_result(Modern(), 2)
        self.assertEqual(status, "available")
        self.assertIsNone(error)
        self.assertEqual(timeline[0], 10)
        self.assertEqual(timeline[1]["long_scoreboard"], [1.0, 1.0])

        class Malformed:
            def timed_warp_samples(self):
                return [{"timestamp": "bad", "stall_reason": 0}]

        self.assertEqual(ncu_utils.timed_warp_sample_info(Malformed())[0], "available")
        timeline, status, error = ncu_utils.warp_state_timeline_result(Malformed(), 2)
        self.assertIsNone(timeline)
        self.assertEqual(status, "error")
        self.assertTrue(error)

    def test_source_stall_absent_zero_and_unmapped_are_separate(self):
        absent, status, _ = extract_stall_hotspots.collect_per_pc_result(Action())
        self.assertFalse(absent)
        self.assertEqual(status, "not_collected")

        metric_name = "smsp__pcsamp_warps_issue_stalled_long_scoreboard"
        zeros = Action({metric_name: Metric([0, 0], [100, 200])})
        zero_data, status, _ = extract_stall_hotspots.collect_per_pc_result(zeros)
        self.assertFalse(zero_data)
        self.assertEqual(status, "zero_stalls")

        class BadCorrelation:
            def as_uint64(self, index):
                raise ValueError("not an integer PC")

            def as_double(self, index):
                raise ValueError("not a floating PC")

        class NullCorrelation:
            def as_uint64(self, index):
                return None

            def as_double(self, index):
                raise ValueError("no correlated PC")

        class PerPcMetric(Metric):
            def __init__(self, values, correlation):
                super().__init__(values, [0] * len(values))
                self.correlation = correlation

            def correlation_ids(self):
                return self.correlation

        for value in (0, 4):
            with self.subTest(value=value):
                mixed = Action({metric_name: PerPcMetric([value], BadCorrelation())})
                mixed_data, status, detail = extract_stall_hotspots.collect_per_pc_result(mixed)
                self.assertFalse(mixed_data)
                self.assertEqual(status, "error")
                self.assertIn("PC correlation unreadable", detail)
                self.assertNotEqual(status, "zero_stalls")

        null_pc = Action({metric_name: PerPcMetric([0], NullCorrelation())})
        _, status, detail = extract_stall_hotspots.collect_per_pc_result(null_pc)
        self.assertEqual(status, "error")
        self.assertIn("PC correlation unreadable", detail)

        for readable_value, expected_status in ((0, "error"), (4, "partial")):
            with self.subTest(unreadable_value_after=readable_value):
                unreadable = Action({metric_name: Metric([readable_value, None], [0x100, 0x200])})
                unreadable_data, status, detail = extract_stall_hotspots.collect_per_pc_result(unreadable)
                self.assertEqual(status, expected_status)
                self.assertIn("value unavailable", detail)
                if readable_value:
                    self.assertEqual(unreadable_data[0x100][metric_name], readable_value)

        counts = Action({metric_name: Metric([3], [0x100])})
        per_pc, status, _ = extract_stall_hotspots.collect_per_pc_result(counts)
        self.assertEqual(status, "available")
        coverage = extract_stall_hotspots.source_line_coverage(counts, per_pc)
        self.assertEqual(coverage["status"], "unavailable")
        self.assertEqual(coverage["unmapped_pcs"], 1)

    def test_timeline_cli_skips_volta_with_explanation(self):
        class VoltaAction(Action):
            def __init__(self):
                super().__init__(**{
                    "device__attribute_compute_capability_major": 7,
                    "device__attribute_compute_capability_minor": 0,
                })

            def source_info(self, pc):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "volta.ncu-rep"
            report.touch()
            args = ["plot_timeline.py", "--run-dir", tmp, "--report", str(report), "--tag", "v100"]
            output = io.StringIO()
            with mock.patch.object(sys, "argv", args), \
                 mock.patch.object(plot_timeline, "select_actions", return_value=(None, [VoltaAction()])), \
                 contextlib.redirect_stderr(output):
                plot_timeline.main()
            self.assertIn("unsupported on Volta/SM70", output.getvalue())

    def test_malformed_warp_samples_are_reported_as_processing_errors(self):
        class MalformedAction(Action):
            def __init__(self):
                super().__init__(**{
                    "device__attribute_compute_capability_major": 8,
                    "device__attribute_compute_capability_minor": 6,
                })

            def name(self, form):
                return "sample_kernel"

            def timed_warp_samples(self):
                return [{"timestamp": "bad", "stall_reason": 0}]

        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "malformed.ncu-rep"
            report.touch()
            args = ["plot_timeline.py", "--run-dir", tmp, "--report", str(report), "--tag", "bad"]
            output = io.StringIO()
            with mock.patch.object(sys, "argv", args), \
                 mock.patch.object(plot_timeline, "select_actions", return_value=(None, [MalformedAction()])), \
                 contextlib.redirect_stdout(output):
                plot_timeline.main()
            self.assertIn("->", output.getvalue())
            timeline_path = Path(tmp) / "analysis" / "pm_timeline_bad.txt"
            timeline = timeline_path.read_text()
            self.assertIn("Warp-state timeline: error", timeline)
            self.assertIn("timed warp-sample processing failed", timeline)


if __name__ == "__main__":
    unittest.main()
