# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the three kernel packages: `state_kernel/`, `tool_kernel/`, and `orchestration_kernel/`.
- `docs/design.md` is the core architecture spec (three-kernel separation and composition).
- `docs/instruction.md` captures current implementation constraints and priorities.
- `pyproject.toml` defines the Python package metadata.
- No `tests/` directory exists yet; add one when introducing tests.

## ライブラリ
- langchain v1.0以降、langgraph v1.0以降を使うこと
- logger、richライブラリでログ出力すること
- llmはlangchain経由でollamaを利用すること
- .envで設定を保持すること

## Build, Test, and Development Commands
- `python -m pip install -e .` installs the package in editable mode for local development.
- `python -m pytest` runs the test suite once tests are added.
- `python -m build` builds a wheel/sdist if you need distribution artifacts.

## Coding Style & Naming Conventions
- Use 4-space indentation and standard Python naming: `snake_case` for functions/vars, `PascalCase` for classes, `lower_case` module names.
- Prefer explicit type hints for kernel public interfaces.
- Keep kernels independent; access across kernels only through their public APIs (see `docs/design.md`).
- Implementations should be swappable (e.g., state stored in JSON/YAML initially, with a future DB backend).

## Testing Guidelines
- Use `pytest` with files named `test_*.py` under `tests/`.
- Keep unit tests focused on each kernel’s public API; avoid cross-kernel coupling in test helpers.
- No coverage target is defined yet; document any new requirements in this file.

## Commit & Pull Request Guidelines
- There is no existing Git history, so no established commit message convention.
- Use short, imperative summaries (e.g., “Add state task claim API”) and include rationale in the body when needed.
- PRs should include: a short description, linked issues (if any), and test notes (commands run or “not run”).

## Architecture Overview
- This project implements three independent kernels (state, tool, orchestration) composed by a thin composition layer.
- Current scope is Single Turn Strategy only; PDCA is deferred (see `docs/instruction.md`).
- Tool definitions should be created from Python functions via `StructuredTool.from_function`.
