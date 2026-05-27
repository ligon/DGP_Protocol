# DGP_Protocol -- developer task runner.
#
# Wraps the common Poetry / quality-gate / release flows documented
# in CONTRIBUTING.md.  Run `make help` for the catalog.
#
# Conventions:
#   - All real work goes through `poetry run ...` so the local venv
#     is used regardless of what's on PATH.
#   - README.md is generated from README.org via pandoc.  It is
#     .gitignored; the publish workflow regenerates it on release.
#   - Version semantics follow PEP 440 (Poetry's default), which is
#     semver-compatible for the X.Y.Z core but uses aN/bN/rcN for
#     prereleases rather than -alpha.N etc.

POETRY ?= poetry
PANDOC ?= pandoc

# Resolve current version once for use in release-tag etc.
VERSION = $(shell $(POETRY) version -s)

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Help -- self-documenting from `## ` comments on target lines.
# ---------------------------------------------------------------------------

.PHONY: help
help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n\nTargets:\n"} \
	      /^[a-zA-Z0-9_.-]+:.*##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Build artifacts
# ---------------------------------------------------------------------------

.PHONY: readme
readme:  ## Regenerate README.md from README.org (pandoc).
	$(PANDOC) -f org -t gfm README.org -o README.md

.PHONY: build
build: readme  ## Build sdist + wheel into dist/.
	$(POETRY) build

.PHONY: clean
clean:  ## Remove build artifacts and the generated README.md.
	rm -rf build/ dist/ *.egg-info/ README.md

# ---------------------------------------------------------------------------
# Quality gates (mirror .github/workflows/ci.yml + CONTRIBUTING.md)
# ---------------------------------------------------------------------------

.PHONY: ruff
ruff:  ## Lint with ruff.
	$(POETRY) run ruff check .

.PHONY: black-check
black-check:  ## Check formatting with black (no changes).
	$(POETRY) run black --check .

.PHONY: mypy
mypy:  ## Type-check with mypy.
	$(POETRY) run mypy dgp_protocol tests

.PHONY: test
test:  ## Run pytest.
	$(POETRY) run pytest

.PHONY: check
check: ruff black-check mypy test  ## Run all four quality gates (ruff, black, mypy, pytest).

.PHONY: format
format:  ## Apply ruff fixes + black formatting in place.
	$(POETRY) run ruff check . --fix
	$(POETRY) run black .

# ---------------------------------------------------------------------------
# Version bumps (PEP 440 via Poetry).
#
# Poetry's `version` subcommand has some quirks on prereleases.  From
# 0.1.0a0 the bumps land as follows:
#
#   bump-alpha    -> 0.1.0a1     (next alpha in the same X.Y.Z)
#   bump-beta     -> 0.1.0b0     (drop to beta of the same X.Y.Z)
#   bump-rc       -> 0.1.0rc0
#   bump-finalize -> 0.1.0       (drop the prerelease tag entirely)
#   bump-patch    -> 0.1.1a0     (next patch, fresh alpha)
#   bump-minor    -> 0.2.0a0     (next minor, fresh alpha)
#   bump-major    -> 1.0.0a0     (next major, fresh alpha)
#
# If you're already on a stable X.Y.Z, prepatch/preminor/premajor
# behave the same; `bump-stable-patch` etc. give you stable-to-stable
# bumps without dipping back through alpha.
# ---------------------------------------------------------------------------

.PHONY: bump-alpha
bump-alpha:  ## Bump prerelease counter (0.1.0a0 -> 0.1.0a1).
	$(POETRY) version prerelease

.PHONY: bump-beta
bump-beta:  ## Move to next prerelease phase (alpha -> beta -> rc).
	$(POETRY) version prerelease --next-phase

.PHONY: bump-finalize
bump-finalize:  ## Drop the prerelease tag (0.1.0a3 -> 0.1.0).
	$(POETRY) version patch

.PHONY: bump-patch
bump-patch:  ## Bump patch + start fresh alpha (0.1.0 -> 0.1.1a0).
	$(POETRY) version prepatch

.PHONY: bump-minor
bump-minor:  ## Bump minor + start fresh alpha (0.1.0 -> 0.2.0a0).
	$(POETRY) version preminor

.PHONY: bump-major
bump-major:  ## Bump major + start fresh alpha (0.1.0 -> 1.0.0a0).
	$(POETRY) version premajor

.PHONY: bump-stable-patch
bump-stable-patch:  ## Stable patch bump (no prerelease tag).
	$(POETRY) version patch

.PHONY: bump-stable-minor
bump-stable-minor:  ## Stable minor bump (no prerelease tag).
	$(POETRY) version minor

.PHONY: bump-stable-major
bump-stable-major:  ## Stable major bump (no prerelease tag).
	$(POETRY) version major

# ---------------------------------------------------------------------------
# Release helpers
# ---------------------------------------------------------------------------

.PHONY: version
version:  ## Print the current version (poetry version -s).
	@echo $(VERSION)

.PHONY: release-tag
release-tag:  ## Create an annotated git tag for the current version (does not push).
	@v=$(VERSION); \
	  if git rev-parse "v$$v" >/dev/null 2>&1; then \
	    echo "error: tag v$$v already exists" >&2; exit 1; \
	  fi; \
	  git tag -a "v$$v" -m "Release $$v" && \
	  echo "Tagged v$$v.  Push with:  git push origin v$$v"

.PHONY: release-check
release-check: check build  ## Pre-release dry run: gates + build.
	@echo
	@echo "Release check passed for version $(VERSION)."
	@echo "Next steps:"
	@echo "  1. git commit -am 'Release $(VERSION)' && git push"
	@echo "  2. make release-tag"
	@echo "  3. git push origin v$(VERSION)"
