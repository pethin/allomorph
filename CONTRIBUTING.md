# Contributing to Allomorph

Thank you for your interest in contributing to **Allomorph**! We welcome contributions ranging from new SPICE netlists and pickup profiles to performance optimizations in the native simulation engine, physical acoustics math, and documentation.

---

## 1. Contributor License Agreement (CLA)

Allomorph is distributed under the [PolyForm Noncommercial License 1.0.0](LICENSE) for the community and is dual-licensed for commercial deployments.

To ensure that contributions can be safely integrated, distributed, and maintained under both open and commercial terms, **all contributors must agree to the [Allomorph Contributor License Agreement (CLA)](CLA.md)**.

### How to Sign the CLA
You do not need to sign a separate paper form. Simply include a standard Git Developer Certificate of Origin / CLA sign-off line in your commit message:

```bash
git commit -s -m "feat(circuits): add vintage Thunderbird humbucker netlist"
```
Or include the line in your commit message / pull request description:
```
Signed-off-by: Your Legal Name <your.email@example.com>
```

By including this line, you certify that your contribution complies with the terms set forth in [`CLA.md`](CLA.md).

---

## 2. Development Setup

Allomorph strictly uses **`uv`** as its Python package and environment manager:

```bash
# Clone the repository
git clone https://github.com/pethin/allomorph.git
cd allomorph

# Install dependencies and sync environment
uv sync

# Run linting and code quality checks
uv run ruff check

# Run static type checking (strict preset)
uv run pyrefly check

# Run the automated test suite with coverage (configured in pyproject.toml)
uv run pytest
```

---

## 3. Engineering Guidelines & Guardrails

When writing code or adding netlists, please review and adhere to the project standards defined in [`AGENTS.md`](AGENTS.md) and [`docs/architectural_guardrails.md`](docs/architectural_guardrails.md):

* **Package & Dependencies:** Use `uv`. Strictly do not add `scipy`, `pandas`, `matplotlib`, or `soundfile`. Audio I/O is handled by `pedalboard`, dataframes by `polars`, and charts by `altair`.
* **Code Quality & Linting (`ruff`):** Code must pass `uv run ruff check` with zero errors. Run `uv run ruff check --fix` to automatically format import blocks and resolve standard stylistic rules. Target version is Python 3.14 (`py314`).
* **Strict Static Typing (`pyrefly`):** All code (library modules, scripts, and tests) is strictly checked under pyrefly's `strict` preset (`preset = "strict"` in `pyproject.toml`). Every function signature, parameter, container, and return value must have explicit type annotations. Do NOT disable type checking rules, downscale presets, or use suppressions (`# type: ignore`) unless proven fundamentally impossible. Use `pydantic` for structured schemas and runtime validation where appropriate.
* **Testing:** All pull requests must pass the test suite (`uv run pytest`) without regressions. New features, pickup profiles, or filter mathematics must include corresponding unit tests in `tests/`.
* **Code Style & Performance:** Keep Python code clean, vectorized with NumPy where appropriate, and formatted. Avoid interpreted Python loops over audio sample buffers (use Numba JIT `@njit(fastmath=True)` for iterative DSP state solvers).

---

## 4. Test Coverage & Quality Standards

All contributions must adhere to the following test coverage and performance benchmarks:

### 4.1 Coverage Target & Critical Modules
* **Repository-Wide Coverage ($\ge 85\%$):** Overall statement and branch coverage across `src/allomorph/` must meet or exceed **85%**.
* **High-Criticality Modules ($\ge 90\text{--}100\%$):** Core configuration schemas, routing, and packaging orchestration must maintain near-complete test coverage:
  - `allomorph.config.strings` (100%)
  - `allomorph.config.scales` (99%)
  - `allomorph.circuit.schema` (99%)
  - `allomorph.config.preamps` (98%)
  - `allomorph.config.geometry` (97%)
  - `allomorph.physics.aperture` (97%)
  - `allomorph.physics.strings` (97%)
  - `allomorph.circuit.cli` (96%)
  - `allomorph.visualizer.__init__` (94%)
  - `allomorph.circuit.audit` (93%)
  - `allomorph.pipeline.batch` (92%)
  - `allomorph.circuit.parser` (91%)
  - `allomorph.version` (91%)
  - `allomorph.pipeline.pack` (90%)
  - `allomorph.circuit.solver` (90%)
  - `allomorph.naming` (90%)

### 4.2 Fast Execution Guardrail ($< 25\text{ seconds}$)
* The entire test suite (all 340+ tests across `circuit/`, `config/`, `dsp/`, `physics/`, `pipeline/`, `tone3000/`, and `visualizer/`) must execute in **under 25 seconds** on modern hardware.
* **Bounded Synthetic Buffers:** File I/O tests and forward audio simulations must use small, bounded synthetic audio buffers (e.g. 50–100 ms / 2,400–4,800 samples) in temporary directories (`tmp_path`) rather than writing to workspace directories (`audio/`, `tone3000/`) or simulating full 4-minute dry tracks.
* **Algorithmic Articulation Tests:** Long-duration dry signal subroutines (plucks, harmonics, vibrato, slap/pop) are validated via dedicated unit tests in [`tests/dsp/test_optimal_dry.py`](tests/dsp/test_optimal_dry.py).

### 4.3 Test-Driven Bug Fixes (TDD)
* When fixing a bug, first write a minimal, failing regression test that reproduces the exact failure mode.
* Verify the test fails for the expected reason, implement the fix in the source module, and verify the test passes cleanly without regressions.

### 4.4 Strict Typing in Test Files
* Under `preset = "strict"`, test files are fully type-checked.
* Every fixture parameter (`monkeypatch: pytest.MonkeyPatch`, `tmp_path: Path`, `capsys: pytest.CaptureFixture[str]`), helper function, and mock callback must include explicit type annotations.

### 4.5 Architectural Invariants Verification
* All mathematical invariants defined in [`AGENTS.md`](AGENTS.md) and [`docs/architectural_guardrails.md`](docs/architectural_guardrails.md) are verified by [`tests/test_guardrails.py`](tests/test_guardrails.py), including:
  - $C^\infty$ algebraic rail saturation ceiling ($V_{\text{sat}} \le 0.985$).
  - Asymmetric standing-wave displacement ratio limits (`alg4`, $+12.0\text{ dB}$ boost / $-16.0\text{ dB}$ cut).
  - Exact inharmonicity RBF solver monotonicity across bass fundamental registers ($< 10^{-10}$ relative error).
  - Sub-audible DC transmission ($H_{\text{preamp}}(0) \ge 1.0$).
  - Zero-scroll tone naming display budgets ($\le 34$ chars).
  - Strict fail-fast declarative integrity with zero silent fallbacks.

---

## 5. Submitting a Pull Request

1. **Fork the repository** on GitHub.
2. **Create a topic branch** (`git checkout -b feat/my-new-pickup`).
3. **Verify code quality, typing, and tests:**
   ```bash
   uv run ruff check
   uv run ruff format --check
   uv run pyrefly check
   uv run pytest
   ```
4. **Sign off your commits:**
   ```bash
   git commit -s -m "feat: description of changes"
   ```
5. **Open a Pull Request** against the `main` branch with a clear description of your changes and motivation.
