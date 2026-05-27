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

A `Makefile` wraps the common flows -- run `make help` for the full
catalog.  The four quality gates that must pass before a PR lands:

```sh
make check          # = ruff + black-check + mypy + test
```

or, individually:

```sh
poetry run ruff check .          # make ruff
poetry run black --check .       # make black-check
poetry run mypy dgp_protocol tests   # make mypy
poetry run pytest                # make test
```

To apply fixes in place:

```sh
make format         # ruff --fix + black .
```

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

## Building and releasing

### Local build

The PyPI long-description is generated from `README.org` (which is the
canonical doc) via pandoc.  The `make build` target chains that step
in for you:

```sh
make build          # = pandoc README.org -> README.md, then poetry build
```

`README.md` is `.gitignore`d on purpose — the publish workflow
regenerates it from `README.org` on every release.

### Cutting a release

The Makefile wraps the bump + tag flow.  A typical alpha → next-alpha
release:

```sh
make bump-alpha               # 0.1.0a0 -> 0.1.0a1
make release-check            # runs all quality gates + builds (dry run)
v=$(make -s version)
git commit -am "Release $v"
git push
make release-tag              # creates annotated tag for current version
git push origin "v$v"
```

Other bump targets (`make help` for the full list):

| Target | Effect | Underlying command |
|---|---|---|
| `bump-alpha` | `0.1.0a0 → 0.1.0a1` | `poetry version prerelease` |
| `bump-beta` | `0.1.0a3 → 0.1.0b0` | `poetry version prerelease --next-phase` |
| `bump-finalize` | `0.1.0a3 → 0.1.0` | `poetry version patch` |
| `bump-patch` | `0.1.0 → 0.1.1a0` | `poetry version prepatch` |
| `bump-minor` | `0.1.0 → 0.2.0a0` | `poetry version preminor` |
| `bump-major` | `0.1.0 → 1.0.0a0` | `poetry version premajor` |
| `bump-stable-{patch,minor,major}` | Stable → stable, no prerelease tag | `poetry version {patch,minor,major}` |

The publish workflow triggers on any tag matching `v*`.  The `build`
job runs unconditionally; the `publish-pypi` job runs only for tag
pushes and published Releases.  Watch progress under the *Actions*
tab.

### One-time PyPI setup (trusted publishing)

Before the first publish, register the repository as a trusted
publisher on PyPI:

1. Sign in at https://pypi.org/manage/account/publishing/.
2. Click *Add a new pending publisher* and fill in:
   - **Owner:** `ligon`
   - **Repository:** `DGP_Protocol`
   - **Workflow filename:** `publish.yml`
   - **Environment:** `pypi`
3. (Recommended) In GitHub, go to *Settings → Environments* and
   create an environment named `pypi`.  Add yourself as a required
   reviewer so that an actual publish requires manual approval.

After the first successful run, PyPI promotes the pending publisher
to a real one and no further setup is needed.

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
