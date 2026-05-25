# Contributing to DGP_Protocol

Thanks for your interest.  This package is a foundational dependency
for analog-estimation toolkits (see [`README.org`](./README.org) for
the framing); changes here ripple downstream into consumer packages,
so the design discipline is somewhat tight.

## Quick start

```sh
git clone https://github.com/ligon/DGP_Protocol.git
cd DGP_Protocol
poetry install
poetry run pytest
```

Python 3.11+ required.  Numpy is the only runtime dependency; dev
dependencies (pytest, ruff, black, mypy) are managed by poetry.

## Development workflow

The three quality gates that must pass before a PR lands:

```sh
poetry run ruff check .
poetry run black --check .
poetry run mypy dgp_protocol tests
poetry run pytest
```

If `black --check` fails, run `poetry run black .` to apply the
formatting.  If `ruff` flags fixable issues, `poetry run ruff check
. --fix` will resolve them.

## Submitting a pull request

1. Branch off `main` with a descriptive name.
2. Make small, focused commits with messages that explain *why*,
   not just *what*.
3. Ensure all four gates above are green.
4. Open the PR against `main`; include a brief design rationale in
   the body if your change touches the public API surface.

## Design context — please read before substantive changes

The conceptual framing for this package is captured in two places:

- **[`AGENTS.md`](./AGENTS.md)** at the root of this repo gives the
  full scope discipline: what the package *is* (Protocol +
  combinators + thin wrappers), what it *is not* (a library of
  working DGPs), and the conventions that follow.  Originally
  written as a briefing for AI agents picking up the work, it's
  equally useful for human contributors.
- The **design note** in the sibling
  [`ligon/ManifoldGMM`](https://github.com/ligon/ManifoldGMM) repo,
  at `docs/design/dgp.org`, captures the design conversation that
  motivated DGP_Protocol's existence (the Manskian
  analog-estimation framing, the three-layer responsibility map,
  the migration story for the first consumer, open design
  questions).

If you're adding a feature that wouldn't be useful to any consumer
*other than* ManifoldGMM, please file it on ManifoldGMM instead —
this package stays estimator-agnostic by design.

## Reporting issues

Use the GitHub issue tracker.  A useful issue includes:

- A short prose description of the problem or proposed change.
- A minimal reproducer (a few lines of Python that demonstrate the
  issue).
- The package version (`pip show DGP_Protocol`) and Python version.

## License + citation

This package is released under BSD-3-Clause (see `LICENSE`).  If you
use it in academic work, citation is appreciated; see the
"How to cite" section of `README.org` or the GitHub
"Cite this repository" button (driven by `CITATION.cff`).

## Contact

Ethan Ligon, UC Berkeley.  See `pyproject.toml` for the email.
