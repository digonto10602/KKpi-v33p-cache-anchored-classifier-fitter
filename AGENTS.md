# AGENTS.md

## Purpose

This file gives Codex durable instructions for this session/repository.

This project contains C++/CUDA/OpenMP/Python scientific software for lattice-QCD finite-volume three-body quantization workflows, GPU/CPU cache generation, determinant scans, true-zero classification, K3df fitting, plotting, and benchmark/report generation.

Codex must treat this repository as validation-heavy scientific software. The goal is not just to make code run. The goal is to preserve numerical correctness, CPU/GPU parity, cache safety, reproducible fitting, and performance only after correctness is established.

---

## Non-negotiable rules

1. Correctness comes before performance.
2. Explore relevant files before editing.
3. Make the smallest patch that solves the requested task.
4. Do not change physics formulas, determinant definitions, fitting equations, true-zero classification rules, irrep conventions, parity conventions, or lattice-level mapping unless the user explicitly asks.
5. Do not silently overwrite existing outputs, summaries, plots, caches, or reports unless the user explicitly asks.
6. Preserve the CPU OpenMP path as the trusted reference unless the task explicitly targets CPU refactoring.
7. Preserve cache compatibility and metadata. Never reuse a cache if requested physics/setup metadata do not match.
8. Do not assume a speedup is useful unless validation still passes.
9. Always report changed files, commands run, validation status, benchmark status, and remaining risks.
10. If a task is large, split it into: explore → plan → implement → validate → benchmark → report.

---

## User/project defaults

Use these defaults when the user does not specify otherwise.

### Hardware/software target

- Local GPU: NVIDIA RTX 3070, 8 GB VRAM.
- Local CPU: Intel i7-12700KF, 12 cores / 20 threads.
- RAM: 32 GB.
- OS: Ubuntu 24.04.
- CUDA: CUDA 13.0-class environment.
- Use CUDA, cuBLAS, and cuSOLVER as separate terms. Do not write them as a slash-combined phrase.

### Common physics/workflow defaults

- Default Lbyas values: `20` and `24`.
- Default irreps:
  - `000_A1m`
  - `100_A2`
  - `110_A2`
  - `111_A2`
  - `200_A2`
- Common cache test range:
  - `Ecm_min = 0.2631`
  - `Ecm_max = 0.36`
- Smoke benchmark first:
  - `coarseN = 500`
- Production target may later be:
  - `coarseN = 50000`
- GPU cache generator builds/stores:
  - `F2`
  - `K2inv`
  - `G`
  - `F3`
  - `F3inv`
  - `Vsel`
  - `plm_config`
  - `klm_config`
- CPU comparison path:
  - OpenMP-based CPU cache generator.
- GPU/CPU comparison must include:
  - determinant signs
  - accepted true zeros
  - Ecm ordering
  - cache reload behavior
  - metadata identity
  - per-Lbyas and per-irrep matching

### Current fitting starting point

When the user asks to continue the v32x/v32-series fitting workflow and does not give a different starting point, use:

```text
[v32x-FCN] eval=51 chi2=0.00027552010191639938 model_found=32/32
K3iso0=73735.840894011912
K3iso1=-972421.14060757787
K3B=347174.05548116949
K3E=-1226756.7068845264
```

### Multi-L fitting assumptions

- Lbyas `20` and `24` are independent ensembles.
- Cross-L covariance is zero unless the user explicitly provides cross-L covariance.
- Use all states under `Ecm_cutoff = 0.335` unless the user changes the cutoff.
- Map the k-th lattice level to the k-th accepted true zero per `(Lbyas, irrep)`.
- Missing irreps are allowed.
- Missing caches may be built automatically only if that is part of the current workflow and metadata are safe.

---

## Default Codex workflow

For every non-trivial task, follow this order:

1. **Explore**
   - Identify relevant source files, scripts, build targets, configs, cache directories, and existing tests.
   - Read before editing.
   - Do not edit during initial exploration unless the task is tiny and the relevant file is already obvious.

2. **Plan**
   - State the minimal change plan.
   - Identify expected validation commands.
   - Identify expected benchmark commands if performance is relevant.

3. **Implement**
   - Make small, targeted edits.
   - Avoid broad rewrites.
   - Preserve public interfaces unless explicitly asked to change them.
   - Preserve output formats unless explicitly asked to change them.

4. **Build**
   - Use the repository’s existing build system.
   - Prefer existing scripts/Makefiles/CMake targets over inventing new commands.
   - If build commands are unknown, discover them from README, scripts, CI, Makefile, CMakeLists, or prior logs.

5. **Validate**
   - Run the smallest relevant correctness check first.
   - For GPU work, validate against CPU exact/reference output.
   - For plotting work, confirm the requested datasets are actually loaded and rendered.
   - For fitting work, confirm state mapping, `model_found`, and chi-square output.

6. **Benchmark**
   - Benchmark only after validation passes.
   - Use `coarseN=500` first.
   - Do not launch production-scale tests unless explicitly requested.
   - Report timing by stage if available.

7. **Report**
   - End every task with:
     - files changed
     - commands run
     - validation result
     - benchmark result, if applicable
     - risks/limitations
     - recommended next step

---

## Build and test discovery rules

Do not assume one universal build command. First inspect the repository.

Look for:

- `README.md`
- `AGENTS.md`
- `CMakeLists.txt`
- `Makefile`
- `setup.py`
- `pyproject.toml`
- `requirements.txt`
- `environment.yml`
- `scripts/`
- `tests/`
- `benchmarks/`
- `reports/`
- previous run logs

Preferred command patterns when appropriate:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

or, if the repo clearly uses Make:

```bash
make -j$(nproc)
```

or, for Python-only scripts:

```bash
python3 path/to/script.py --help
```

Never hard-code machine-specific absolute paths in new reusable scripts unless the existing package already requires them and the user explicitly provided them.

---

## Specialized working modes

If the user says “use the X agent,” follow the corresponding mode below. If the current Codex client supports project subagents, these modes may be implemented as `.codex/agents/*.toml` later. If not, emulate the mode in the main Codex session.

### 1. repo-explorer

Use for read-only codebase understanding.

Rules:

- Do not edit files.
- Map source files, execution flow, build commands, test commands, cache formats, input/output files, and current assumptions.
- Identify the smallest files/functions that need changes.
- Return a concise map and recommended next steps.

Good for:

- New packages.
- Large zip/report analysis.
- Understanding a cache/fitter/plotting pipeline before implementation.

Output format:

```text
Relevant files:
- ...

Pipeline:
- ...

Build/test commands found:
- ...

Risks/questions:
- ...

Recommended next step:
- ...
```

---

### 2. implementation-worker

Use for scoped code changes.

Rules:

- Edit only files required by the task.
- Preserve behavior outside the requested scope.
- Run the smallest relevant build/test.
- Do not optimize unless optimization is the task.
- Do not reformat unrelated code.

Good for:

- Adding command-line options.
- Fixing data loading.
- Adding PNG output next to PDF.
- Adding metadata checks.
- Small bug fixes.

Output format:

```text
Changed files:
- ...

What changed:
- ...

Commands run:
- ...

Result:
- ...
```

---

### 3. validation-scientist

Use for numerical/physics validation.

Rules:

- Decide whether results are correct, not whether they are fast.
- CPU exact/reference wins unless the task proves the CPU path is wrong.
- For mismatches, report exact `(Lbyas, irrep, Ecm, state/index, expected, observed, tolerance)`.
- Do not approve a performance change unless validation passes.

Must check when relevant:

- determinant signs
- accepted true-zero lists
- Ecm ordering
- irrep matching
- Lbyas matching
- lattice-level to true-zero mapping
- cache reload reproducibility
- tolerance thresholds
- CPU/GPU parity

Output format:

```text
Validation matrix:
- Lbyas=..., irrep=...: PASS/FAIL

Mismatches:
- ...

Tolerance used:
- ...

Conclusion:
- PASS/FAIL
```

---

### 4. gpu-performance-engineer

Use for CUDA/C++ performance work.

Rules:

- Optimize only after validation is established.
- Focus on GPU residency and data movement first.
- Treat host-device transfers as expensive.
- Respect RTX 3070 8 GB VRAM limits.
- Prefer batching, stream overlap, memory layout improvements, kernel fusion, and cuBLAS/cuSOLVER reuse.
- Make one optimization at a time.
- Revalidate after every optimization.
- Benchmark small first: `coarseN=500`.

Performance priorities:

1. Reduce host-to-device and device-to-host copies.
2. Keep determinant scan and cache data GPU-resident when possible.
3. Avoid repeated allocations.
4. Reuse CUDA streams, cuBLAS handles, and cuSOLVER handles where safe.
5. Improve memory layout/coalescing.
6. Fuse kernels only when it reduces overhead without harming clarity.
7. Use pinned memory only when it measurably helps.
8. Avoid increasing VRAM pressure beyond safe limits.

Output format:

```text
Bottleneck:
- ...

Optimization:
- ...

Validation:
- ...

Benchmark:
- before: ...
- after: ...
- speedup: ...

Risks:
- ...
```

---

### 5. cpu-openmp-reference-keeper

Use to protect or improve the CPU reference path.

Rules:

- Keep the CPU path faithful and simple.
- Do not change physics formulas.
- Do not make CPU output incompatible with GPU comparison.
- Preserve OpenMP determinism where possible.
- Make CPU timing reproducible.

Good for:

- CPU/GPU apples-to-apples comparison.
- Fixing CPU reference output.
- Confirming GPU changes did not contaminate CPU reference logic.

Output format:

```text
CPU reference status:
- ...

Compatibility with GPU comparison:
- ...

Commands run:
- ...

Conclusion:
- ...
```

---

### 6. cache-format-guardian

Use for cache metadata, binary compatibility, and safe reuse.

Rules:

- Do not allow silent cache reuse when metadata are missing or mismatched.
- Do not allow silent overwrite of incompatible cache files.
- Prefer explicit failure over silent wrong physics.
- Cache filenames alone are not enough; metadata must also be checked where possible.

Cache identity should include, when applicable:

- code/package version
- backend identity: CPU/GPU
- Lbyas
- xi
- irrep
- parity convention
- projector convention
- masses
- waves
- Ecm_min
- Ecm_max
- coarseN
- basis dimensions
- K2/scattering parameters
- shape/effective-range settings
- config dimensions
- binary format version
- endian/precision if relevant

Output format:

```text
Cache files inspected:
- ...

Metadata present:
- ...

Metadata missing:
- ...

Unsafe reuse risks:
- ...

Recommended fix:
- ...
```

---

### 7. fit-run-controller

Use for K3df fitting workflows.

Rules:

- Preserve starting parameter values unless the user gives new values.
- Preserve Lbyas and irrep selection rules.
- Preserve state-selection rules.
- Confirm lattice data loading.
- Confirm Ecm conversion if reading lab-frame energies.
- Confirm model_found and chi-square.
- Do not change fitting model unless explicitly requested.

Must report:

- input lattice files
- selected states per `(Lbyas, irrep)`
- starting K3df parameters
- final K3df parameters if a fit was run
- `chi2`
- `model_found`
- cache status
- missing states/irreps

Output format:

```text
Fit setup:
- ...

Selected states:
- ...

Starting parameters:
- ...

Fit result:
- ...

Validation:
- ...

Risks:
- ...
```

---

### 8. plotting-data-agent

Use for plotting scripts and data overlay issues.

Rules:

- Preserve legends unless the user explicitly asks to change them.
- Do not remove existing curves or datasets unless requested.
- Confirm each requested dataset is actually loaded.
- For lattice points, verify x/y/error columns and filtering.
- Save requested formats without overwriting unrelated summaries.
- If user asks for PNG as well as PDF, save both.

Common user plotting preference:

- Lattice points should often be large white scatter points with black edge.
- Lattice points should include vertical error bars with caps when errors are available.
- QC spectrum, non-interacting lines, and lattice points should appear on the same plot when requested.

Output format:

```text
Data loaded:
- ...

Plot elements present:
- lattice points: yes/no
- QC spectrum: yes/no
- non-interacting lines: yes/no

Outputs:
- ...

Commands run:
- ...
```

---

### 9. regression-test-builder

Use to turn fixes into repeatable tests.

Rules:

- Prefer small smoke tests before large regression tests.
- Tests should run on minimal data when possible.
- Tests must fail clearly when the bug returns.
- Avoid requiring production-scale GPU runs for basic validation.

Good tests for this project:

- `coarseN=500` GPU cache smoke test.
- CPU/GPU determinant sign comparison at sampled Ecm points.
- Accepted true-zero list comparison.
- Cache reload without rebuild.
- Plot script confirms lattice points are loaded and output files exist.
- Fitter smoke test confirms `model_found` and state mapping.

Output format:

```text
Tests added:
- ...

What each test catches:
- ...

How to run:
- ...

Expected result:
- ...
```

---

### 10. code-reviewer

Use after implementation.

Rules:

- Review the diff, not the whole repo unless necessary.
- Prioritize blocking correctness issues.
- Do not rewrite code unless explicitly asked.
- Check for numerical risks, memory safety, race conditions, cache compatibility, missing validation, and accidental behavior changes.

Review checklist:

- Are physics formulas unchanged unless requested?
- Are CPU/GPU outputs still comparable?
- Are tolerances explicit and reasonable?
- Are cache files protected from unsafe reuse?
- Are host-device transfers minimized only where safe?
- Are CUDA errors checked?
- Are OpenMP race conditions avoided?
- Are plots still loading all requested data?
- Are output filenames safe from accidental overwrite?
- Were tests/benchmarks actually run?

Output format:

```text
Blocking issues:
- ...

Non-blocking issues:
- ...

Validation gaps:
- ...

Suggested next patch:
- ...
```

---

### 11. report-writer

Use to produce markdown reports for future Codex prompts.

Rules:

- Write reports that another Codex session can use without needing chat history.
- Include exact commands, paths relative to repo root, changed files, validation results, benchmark results, and unresolved risks.
- Separate facts from recommendations.
- Do not exaggerate performance or validation.
- If validation was not run, say so clearly.

Output sections:

```markdown
# Report title

## Goal

## Package/repo context

## Files changed

## Commands run

## Validation results

## Benchmark results

## Key findings

## Remaining risks

## Recommended next Codex prompt
```

---

## GPU cache generator workflow

For tasks involving the GPU cache generator, use this sequence:

1. Use `repo-explorer` to locate cache generator source files, CPU reference path, GPU path, output directory, and benchmark scripts.
2. Use `validation-scientist` to establish current CPU/GPU parity.
3. Use `implementation-worker` for the requested feature/fix.
4. Use `validation-scientist` again.
5. Use `gpu-performance-engineer` only after validation passes.
6. Use `regression-test-builder` to preserve the validation.
7. Use `report-writer` to summarize.

Required validation for GPU cache generator changes:

```text
For each default Lbyas in {20, 24}
For each default irrep in {000_A1m, 100_A2, 110_A2, 111_A2, 200_A2}
Check:
- cache file exists
- metadata matches request
- F3inv/Vsel reload succeeds
- determinant sign agrees with CPU reference at sampled Ecm points
- accepted true-zero list agrees with CPU exact/reference mode
```

Do not run `coarseN=50000` unless the user explicitly asks. Use `coarseN=500` for benchmark/smoke validation first.

---

## Determinant scan and true-zero classification workflow

For determinant scan/refinement/true-zero tasks:

1. Identify current exact/reference mode.
2. Identify current fast/GPU mode.
3. Compare sign changes and accepted true zeros.
4. Check Ecm ordering and duplicate/near-duplicate zeros.
5. Validate state mapping per `(Lbyas, irrep)`.
6. Only then optimize scan/refinement.

When reporting mismatches, include:

```text
Lbyas:
irrep:
Ecm:
CPU sign/value:
GPU sign/value:
accepted/rejected:
classification reason:
tolerance:
```

Do not change true-zero classification thresholds silently.

---

## Fitting workflow

For K3df fitting tasks:

1. Load lattice spectrum files.
2. Convert En_lab to Ecm if required by the workflow.
3. Apply Ecm cutoff.
4. Select states per `(Lbyas, irrep)`.
5. Match k-th lattice level to k-th accepted true zero.
6. Use the current starting K3df parameters unless user changes them.
7. Run smallest smoke fit first.
8. Report `chi2`, `model_found`, and fitted parameters.
9. Save combined multi-L and separate per-L outputs if the existing workflow supports that.

Do not change the model or K3df basis unless explicitly requested.

---

## Plotting workflow

For plotting/data scripts:

1. Confirm the input file paths exist.
2. Print or log how many rows were loaded from each file.
3. Confirm filters do not remove requested points unintentionally.
4. Confirm plot contains every requested layer:
   - lattice data
   - QC spectrum/zeros
   - non-interacting lines
   - fit curves if requested
5. Save to requested formats.
6. Do not change legends unless requested.
7. Do not overwrite summaries when input file lists change.

For lattice points at fixed L:

- Explicitly filter or select the requested L.
- Confirm count after filtering.
- Plot vertical error bars if the file provides errors.
- Use large white markers with black edges when requested.

---

## Output and overwrite safety

When generating outputs:

- Prefer timestamped or parameterized filenames.
- Do not overwrite existing summaries when the input file list changes.
- If an output exists, either:
  - write a new filename, or
  - ask only if the user explicitly requested overwriting behavior.
- For scripts, provide an option like `--overwrite` only if needed.
- For caches, never overwrite incompatible cache files silently.

Recommended output naming components:

```text
version
Lbyas
irrep
xi
coarseN
Ecm_min
Ecm_max
backend
timestamp when appropriate
```

---

## Logging expectations

For long-running or validation-heavy scripts, logs should include:

```text
[input]
[outdir]
[settings]
[build/cache status]
[Lbyas]
[irrep]
[Ecm range]
[coarseN]
[number of states/zeros]
[timing by stage]
[validation pass/fail]
[output files]
```

For GPU cache generator timings, prefer stage timing like:

```text
basis:
F2/K2:
G/F3:
solve/F3inv:
Vsel:
write:
total:
```

---

## CUDA/C++ safety checklist

For CUDA changes:

- Check CUDA return codes.
- Check cuBLAS/cuSOLVER status.
- Avoid repeated device allocation inside tight loops.
- Avoid unnecessary host-device copies.
- Use pinned memory only when benchmarked.
- Guard VRAM usage.
- Avoid hidden synchronization unless needed.
- Use streams only when data dependencies are clear.
- Preserve numerical precision.
- Revalidate after changing memory layout or batching.

For C++ changes:

- Avoid global mutable state unless already present and safe.
- Avoid data races in OpenMP regions.
- Keep ownership clear.
- Prefer RAII for resources.
- Do not introduce undefined behavior for speed.
- Keep error messages actionable.

---

## Python script standards

For Python scripts:

- Use `argparse` for new CLI options.
- Print loaded row counts for data scripts.
- Use `pathlib.Path` for paths.
- Do not hard-code absolute user-machine paths in reusable scripts unless required by existing workflow.
- Preserve existing legends and styles unless requested.
- Save outputs without silently overwriting unrelated summaries.
- Keep output filenames parameter-aware.

---

## Reports and Codex prompt generation

When asked to produce a report or next Codex prompt:

- Make it self-contained.
- Include exact starting assumptions.
- Include exact validation requirements.
- Include exact benchmark requirements.
- Include expected pass/fail criteria.
- Include paths relative to repo root.
- Include all known pitfalls.

A good final Codex prompt should include:

```markdown
# Goal

# Current context

# Required changes

# Validation requirements

# Benchmark requirements

# Do not change

# Expected final report
```

---

## Final response format after every task

End with this structure:

```text
Summary:
- ...

Files changed:
- ...

Commands run:
- ...

Validation:
- PASS/FAIL/NOT RUN
- details...

Benchmark:
- PASS/FAIL/NOT RUN
- details...

Risks:
- ...

Next recommended step:
- ...
```

If validation or benchmark was not run, say exactly why.

---

## Stop conditions

Stop and report immediately if:

- CPU/GPU validation fails.
- Cache metadata mismatch is detected.
- A cache file would be overwritten unsafely.
- A build failure appears unrelated to the requested change.
- A test requires production-scale runtime that the user did not request.
- The code path is ambiguous and two possible fixes would change physics behavior.

Do not hide these issues. Report them clearly and recommend the smallest next action.

---

## Preferred philosophy

This repository is a scientific production workflow.

The correct behavior is:

```text
slow but correct first
fast only after correctness
production only after smoke validation
cache reuse only after metadata match
fitting only after state mapping is verified
plots only after data loading is confirmed
```
