# Focused compatibility checks

Run the CPU-only tests without a GPU or Nsight Compute installation:

```bash
python3 -m unittest discover -s tests -v
```

## Native source-attribution check

[`fixtures/source_stalls.cu`](fixtures/source_stalls.cu) is a small deterministic
target, not a throughput benchmark. It follows 32,768 independent chains through
a 16 MiB link array for 1,024 steps, then checks every result against a CPU
oracle. The dependent global load is marked `SAMPLE_TARGET`.

Select a compiler and profiler supporting the actual GPU. For V100, use a
CUDA 12.x compiler and Nsight Compute 2025.2.x. Example commands from the
repository root, using a new run directory:

```bash
export NCU="${NCU:-ncu}"
RUN_DIR=/path/to/new/profile-run
mkdir -p "$RUN_DIR/harness" "$RUN_DIR/reports"
"${NVCC:-nvcc}" -O2 -lineinfo -arch=sm_70 \
    tests/fixtures/source_stalls.cu -o "$RUN_DIR/harness/source_stalls"
"$NCU" --section SourceCounters --import-source yes --clock-control base \
    -k 'regex:source_stalls' -c 1 \
    -o "$RUN_DIR/reports/source.ncu-rep" "$RUN_DIR/harness/source_stalls"
python3 helpers/extract_stall_hotspots.py --run-dir "$RUN_DIR" \
    --report "$RUN_DIR/reports/source.ncu-rep" --tag source \
    --kernel source_stalls --binary "$RUN_DIR/harness/source_stalls"
"$NCU" --import "$RUN_DIR/reports/source.ncu-rep" --page source \
    --print-source cuda,sass --csv > "$RUN_DIR/analysis/ncu-source.csv"
```

Check the program's `PASS` output, report device identity, actual per-PC
samples, source coverage, and SASS match counts. Compare the helper's per-line
counts with NCU's CUDA rows, especially `stall_long_sb` at the marked load.
Compare the same sample population: the helper uses all-sample metrics, not
the separate not-issued-only counters. Sampling counts vary between captures;
the recorded counts below are evidence from one run, not test thresholds.

## Recorded V100 result

On 2026-10-17, a Tesla V100-SXM2-16GB (SM70), CUDA 12.9.86, and Nsight Compute
2025.2.1.0 (build 35987062) passed the native check. The accepted capture
collected SourceCounters alongside a compact metric set.

- Correctness: `PASS: 32768 chains, 1024 dependent loads each`.
- Per-PC sample count: 16,992; 44 distinct sampled PCs, all mapped to source.
- Exact-binary validation: 44/44 sampled PCs matched the report's SASS.
- Five source lines had sampled scheduler states. At fixture line 22, the
  helper and NCU's source CSV both reported 16,871 all-state samples, including
  16,288 long-scoreboard samples. The latter also matched the raw per-PC sum.
- Source contents were not embedded in this capture; NCU's source export used
  the exact source checkout. The reproduction command above embeds source to
  make future reports easier to read elsewhere.

Fixture SHA-256:
`9c78e5290e5c5e6dfd086c5d79c22c6434a055b0c9334dbf8f2bda78aa291ed6`.
Accepted report SHA-256:
`9185eec3137812dded9e7b02bf81653efde840491e8620d84f7c60aaa6373c44`.

Raw reports, binaries, and host-control artifacts are not test-suite dependencies
or committed fixtures. This check validates ordinary SM70 line attribution and
the binary-matching path for this kernel. It does not qualify complex inline
chains, every profiler release, or performance improvements.
