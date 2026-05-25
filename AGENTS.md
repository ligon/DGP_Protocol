# Briefing for AI agents working on DGP_Protocol

This file briefs a fresh AI agent (Claude Code, Codex, etc.) on the
context the repo itself doesn't carry.  Read this *first* before
making changes.

Human contributors should read [`CONTRIBUTING.md`](./CONTRIBUTING.md)
for the development workflow; this file focuses on conceptual context
and scope discipline.

## 1. What this package is — and is NOT

**This package is a Protocol + a few combinators + thin convenience
wrappers.**  Concretely:

- `DataGeneratingProcess` — a runtime-checkable Protocol with two
  members: `data` (frozen property) and `draw(size=None)` (method).
  Two members.  That's it.  The DGP owns its own RNG; `draw()` does
  not accept an `rng` argument.
- `EmpiricalDGP`, `ParametricDGP` — thin containers that wrap
  user-supplied content (an observed matrix; a generator callable
  or a scipy.stats-style frozen distribution) into Protocol-
  conformant objects.  No data-generating logic of their own.
  Carry a `seed` constructor kwarg and a `with_rng(rng)` method for
  reproducibility and parallel-worker fan-out.
- `TwoStageDGP`, `with_data` — composition primitives that take
  DGPs and return DGPs.  No data-generating logic of their own.
- `IIDSampling`, `ClusteredSampling` — internal helpers for
  `EmpiricalDGP`'s bootstrap-resampling.  Not part of the public
  Protocol surface.
- **Distributional-feature surface**: free functions `expect`,
  `mean`, `var`, `cov` (per-observation marginal moments, Reading
  P) and the `SampleDistribution` view accessed via
  `dgp.sample_distribution` (dataset-level sampling moments,
  Reading D, with a `moment_covariance(theta, gi)` entry point for
  analog-estimation consumers).  Both surfaces dispatch to analytic
  overrides when concrete DGPs supply them and fall back to
  adaptive batched Monte Carlo otherwise.  `AnalyticUnavailable`
  and `NumericalWarning` are the corresponding public exception /
  warning categories.

**This package is NOT:**

- A library of working DGPs.  No `GaussianDGP`, `LinearModelDGP`,
  `TimeSeriesARMA_DGP`, etc.  Specific DGPs live in *consumer*
  packages.
- A statistical-inference toolkit.  No `omega_hat`, no specification
  tests, no confidence-interval machinery.
- A simulation framework.  `ParametricDGP` accepts a user-supplied
  generator; the package doesn't ship simulators.

**The single most important rule**: the package must stay
**estimator-agnostic**.  No moment-vector knowledge, no manifold
knowledge, no likelihood knowledge.  If a proposed feature requires
importing from any specific estimator package (e.g.\\ `manifoldgmm`),
the feature belongs in that consumer package, not here.

## 2. Conceptual lineage

A `DataGeneratingProcess` is the *stand-in distribution* from
Manski (1988), *Analog Estimation Methods in Econometrics*
(Chapman and Hall).  The Protocol's two members map directly onto
Manski's framework:

- `data` ≈ the observed sample (the privileged realization of the
  stand-in).
- `draw(...)` ≈ the operation that produces more realizations from
  the stand-in.

Consumers compute *functionals* of the stand-in (the moment vector
in GMM, the log-likelihood in MLE, the kernel density estimate in
nonparametrics).  Different stand-ins yield different analog
estimators — the empirical distribution gives nonparametric
plug-in estimators, a parametric family gives MLE-style estimators,
a bootstrap distribution gives bootstrap inference, etc.

If you haven't internalised this framework, read at least the
introductory chapter of Manski (1988) before making design choices.

## 3. The design conversation lives in ManifoldGMM

The substantive design conversation that motivated this package is
captured in:

```
ligon/ManifoldGMM:docs/design/dgp.org   (PR #45 on that repo)
```

**Read that document first** before touching code here.  It covers:

- The three-layer responsibility map (Model on `MomentRestriction`,
  Data on `DataGeneratingProcess`, Bridge on `GMM` / `GMMResult`).
- Why moment-function placement is on the model, not on the DGP.
- The scope clarification (Protocol + combinators, NOT a library of
  DGPs).
- Composition: `TwoStageDGP`, callable-inner family-of-DGPs
  convention, recursive composition for >2 stages.
- The migration story for the eventual ManifoldGMM-side consumer
  refactor.
- The "settled this round" record of decisions taken: extract-now,
  package naming (`DGP_Protocol` PyPI, `dgp_protocol` import),
  PEP-8 imports, frozen-data semantics.
- Open design questions explicitly NOT settled (see §7 below).

That document is the design log; this repo is the implementation
arm.  Choices that look like they're settled here often have
substantial discussion behind them in the design note.

## 4. The consumer relationship

DGP_Protocol exists because ManifoldGMM needed it, but ManifoldGMM
is one consumer of many that the design accommodates.  Know:

- **ManifoldGMM does not yet depend on DGP_Protocol.**  The
  consumer-side migration is documented in the design note but not
  implemented.  ManifoldGMM's existing `MomentRestriction(data=array,
  clusters=...)` surface is the *legacy* form; the *target* form is
  `MomentRestriction(gi_jax=..., manifold=...)` + a separate
  `EmpiricalDGP(observation=array, sampling=ClusteredSampling(...))`
  passed alongside.
- **Hypothetical future consumers** — an MLE package, an M-estimator
  package, a density-estimator package, a kernel-regression package
  — should be able to consume the same Protocol without DGP_Protocol
  caring about their specifics.

If you see an opportunity to add a feature that's clearly only
useful to ManifoldGMM, **file it on ManifoldGMM, not here**.

## 5. Conventions

### Code

- **Python 3.11+.**  Use modern type-annotation syntax.
- **Frozen dataclasses** for value-type semantics on containers.
  Rebinding observed data uses the `with_data(new_observation)`
  pattern, which returns a fresh instance with structural attributes
  preserved.
- **Runtime-checkable Protocols** (`@runtime_checkable`) so
  `isinstance(obj, DataGeneratingProcess)` works for any
  duck-typed object exposing both members.
- **Optional methods via `hasattr` dispatch.**  The `with_data`
  free function in `composition.py` is the canonical example; the
  free `expect`/`mean`/`var`/`cov` in `marginal.py` and the
  `SampleDistribution` methods in `sample_distribution.py` follow
  the same pattern.  New optional methods should: define on
  concrete types that want to opt in; have consumers detect via
  `hasattr` and fall back gracefully when absent.  When a concrete
  type has the method but can't provide an analytic answer in
  context (e.g., `ParametricDGP.mean` when constructed with a
  custom `generator` rather than a `distribution`), raise
  `AnalyticUnavailable` -- free-function consumers catch that and
  fall back to MC.  Reserve bare `NotImplementedError` for
  principled refusals where MC would be wrong or meaningless
  (e.g., `TwoStageDGP.mean`); free functions propagate it.
- **Numpy is the only runtime dependency.**  Resist adding pandas,
  scipy, jax, etc. unless there's a hard requirement that can't be
  met with numpy alone.

### Tooling

- `poetry install` to set up the dev environment.
- `poetry run pytest` for tests.
- `poetry run ruff check .`, `poetry run black --check .`,
  `poetry run mypy dgp_protocol tests`, and `poetry run pytest`
  are the four gates.  All must pass before a commit lands.
- No `Makefile` yet; commands are run via `poetry run` directly.

### Git

- Commits are authored by a **human** (the git `user.name` /
  `user.email` already configured on the sucoder coder account
  resolves to a real person — currently
  `Ethan Ligon & Sue Coder <ligon+sucoder@berkeley.edu>`).  Do
  NOT change the git config; do NOT substitute a non-human author
  (neither `Coder` nor any AI agent can hold copyright, so neither
  can be the `Author`).
- AI agent identity goes in the `Co-Authored-By:` trailer.  Example:
  `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`.
  If you genuinely don't know your model version, look it up
  before fabricating — under Claude Code the model name is
  recorded per assistant turn in the session jsonl, e.g.

  ```sh
  find ~/.claude/projects -name "${CLAUDE_CODE_SESSION_ID}.jsonl" \
    -exec grep -m1 -oE '"model":"[^"]*"' {} \;
  ```

  Only fall back to `Anthropic Claude, version unknown
  <noreply@anthropic.com>` if no such record exists.
- Pushes go directly to GitHub over HTTPS (the `coder` account cannot
  push to the ligon-side mirror).  `gh auth setup-git` is already
  configured.

## 6. What's deliberately not built (and why)

A new agent might look around and think "I should add X."  For the
items below, X has been considered and parked:

- **`bootstrap_dgp(dgp, scheme=...)` constructor** — discussed in
  the design note as planned but not implemented.  The cluster-block
  bootstrap is already available via `ClusteredSampling` on
  `EmpiricalDGP`.  Wild-bootstrap-of-moment-errors is
  estimator-specific (belongs in ManifoldGMM, not here).  What
  `bootstrap_dgp` would add is the **raw-data bootstrap with
  selectable schemes** (`iid`, `cluster_block`, future
  `block_bootstrap` for time series).  File-on-demand.
- **Analytic `_sd_moment_covariance` on `EmpiricalDGP`** — the
  hook is in place on the `SampleDistribution` view (D-side),
  defaulting to MC.  An iid outer-product implementation for
  `IIDSampling` and a cluster-robust outer-product for
  `ClusteredSampling` would make the analog-estimation
  `omega_hat` consumer call (in ManifoldGMM) skip MC.
  Probably belongs on `SamplingDesign` as
  `moment_covariance_estimator(observation, theta, gi)` (per the
  design note's reference code) with `EmpiricalDGP` delegating;
  not built yet.
- **Free `with_rng(dgp, rng)` / `sample_distribution(dgp)`** —
  parallel to the existing free `with_data(dgp, obs)`.  Useful for
  framework code that rebinds randomness on arbitrary DGP-like
  objects without knowing the concrete type.  One-liners; add
  on demand.
- **Multi-stage composition convenience constructors** (e.g.
  `Hierarchy([level1, level2_factory, level3_factory])`) — nested
  `TwoStageDGP` already works for >2 stages; a flatter API is
  convenience, not capability.  File-on-demand.
- **A `BootstrapDGP` distinct from `TwoStageDGP`** — likely
  redundant.  The design note explicitly maps cluster-wild bootstrap
  onto `TwoStageDGP` composition.

## 7. Open design questions — do NOT resolve alone

These need Ethan's input before any code lands:

- **`with_data` on composite DGPs**: how to split a flat observed
  realization back into outer / inner parts.  The design note's
  Composition section lays out three workable options; none is
  picked yet.
- **`bootstrap_dgp`'s scheme taxonomy**: start narrow (iid,
  cluster_block), grow on demand.  New schemes should be discussed
  before being added — the Protocol's minimality is the constraint.
- **Pickle semantics for parametric DGPs with closure-based
  generators**: cloudpickle support could be added but introduces
  an optional dependency.  Discuss before adding.
- **Whether to add a top-level `draws(dgp, n)` convenience**
  that batches `draw()` calls.  The design note says "no --
  plurality is the caller's concern."  Do not add it without
  explicit user direction.

## 8. Avoid scope creep

The temptation will be to "round out" the package by adding specific
DGP types, convenience functions, or pre-built estimator-specific
helpers.  Resist.

The package's value proposition is *the contract and the
combinators that compose into bigger contracts*.  Specific DGPs and
estimator-specific consumers belong elsewhere.  If you find yourself
wanting to import from `manifoldgmm` (or any other estimator
package), stop: that's a sign the work belongs there, not here.

## 9. Sucoder workspace notes

(These apply to whatever AI-agent harness is running.  Skip if
already familiar.)

- The repo lives in the sucoder coder mirror at
  `/home/coder/mirrors/DGP_Protocol/` with a `ligon` fetch remote
  (read-only-via-fetch, no-push by design).  Pushes go to GitHub
  via HTTPS directly.
- `gh` CLI is authed as `ligon` (Berkeley account, same as the
  ManifoldGMM repo).
- GitHub-side `gitnexus` may eventually index the repo for
  cross-symbol analysis.  When it does, the auto-generated
  `gitnexus:start`/`gitnexus:end` markers should NOT overwrite this
  hand-authored AGENTS.md — gitnexus typically *appends* its
  managed block to an existing file rather than replacing it.  If
  you see unexpected gitnexus content in AGENTS.md, treat it as a
  bug in the indexer's append logic, not as expected behaviour.
