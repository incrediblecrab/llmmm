PY := ./.venv/bin/python
PIP := ./.venv/bin/pip
BUNDLE ?= ../.artifacts/workspace-v2-20260909.tar.gz

.PHONY: help setup setup-train data import-legacy test test-code gate list baselines report clean check check-core check-code sanity sanity-derived recipes text coverage aliases rebuild docs snapshot restore verify train status

help:
	@echo "setup      create .venv and install the package"
	@echo "setup-train install the package, tests and local PyTorch training support"
	@echo "restore    restore a private snapshot (BUNDLE=/path/to/archive.tar.gz)"
	@echo "verify     verify local artifacts against artifacts.lock.json"
	@echo "snapshot   create a private local snapshot and its tracked checksum lock"
	@echo "data       alias for restoring the declared private snapshot"
	@echo "import-legacy import the older prior-study layout (not canonical restoration)"
	@echo "gate       Phase-0 control gate — must pass before any result counts"
	@echo "test       run the test suite"
	@echo "sanity     adversarial checks against the recorded results"
	@echo "docs       fail if a number in the docs matches no artefact"
	@echo "check      gate + tests + sanity + docs, the full pre-commit check"
	@echo "check-core check a restored training workspace without raw-source replay"
	@echo "check-code check code and recorded results without private datasets"
	@echo "train      train the current experiment declared in workspace.json"
	@echo "status     print current project status from its exact source artifacts"
	@echo "recipes    prove every recipe is readable, and show the source mix"
	@echo "text       rejoin titles, quantities and instructions to the corpus"
	@echo "coverage   measure what normalisation discards, and why"
	@echo "aliases    propose new ingredient aliases via Azure, bind them, measure the gain"
	@echo "rebuild    rebuild the corpus with the corrected normaliser (recipe_ids_v2)"
	@echo "baselines  train every family on the honest split"
	@echo "report     the leaderboard"

setup:
	python3 -m venv .venv
	$(PIP) install -q -e '.[dev]'
	@echo "installed. next: make restore BUNDLE=/private/path/archive.tar.gz"

setup-train: setup
	$(PIP) install -q -e '.[torch]'

snapshot:
	$(PY) -m ingredient_model.recovery pack --output "$(BUNDLE)"

restore:
	$(PY) -m ingredient_model.recovery restore --bundle "$(BUNDLE)"

verify:
	$(PY) -m ingredient_model.recovery verify

train: gate
	$(PY) -m ingredient_model.workspace train

status:
	$(PY) scripts/project_status.py

data: restore

import-legacy:
	$(PY) scripts/import_data.py
	$(PY) scripts/build_splits.py

gate:
	$(PY) -m ingredient_model.cli gate

list:
	$(PY) -m ingredient_model.cli list

test:
	$(PY) -m pytest tests/ -q

test-code:
	$(PY) -m pytest tests/ -m "not data" -q

# Re-derives the headline numbers by an independent path, and re-runs the
# controls that would expose a metric paying for something other than skill.
sanity:
	$(PY) scripts/sanity_check.py

sanity-derived:
	$(PY) scripts/sanity_check.py --derived-only

recipes:
	$(PY) -m ingredient_model.cli recipes --verify

# Rejoins titles, quantities, URLs and instructions to the corpus. Check
# alignment first — it finds a drifting reader in seconds rather than at minute
# twenty of the full replay.
text:
	$(PY) scripts/check_text_alignment.py
	$(PY) scripts/build_text_index.py

# Fails when a number in the documentation matches no artefact. Cheap, and it
# catches the failure that review does not: a figure that was correct when it
# was written and quietly stopped being correct when something upstream moved.
docs:
	$(PY) scripts/project_status.py --check
	$(PY) scripts/check_docs.py

# The gate comes first deliberately. If the metrics rate random vectors above
# chance, every number produced afterwards is void, so there is no point running
# anything else.
check: gate test sanity docs

check-core: gate test sanity-derived docs

check-code: test-code
	$(PY) scripts/project_status.py --check

baselines:
	$(PY) -m ingredient_model.cli sweep experiments/baselines.yaml

report:
	$(PY) -m ingredient_model.cli report

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache

coverage:
	$(PY) scripts/analyse_coverage.py

# Mining and alias proposal need the Azure Foundry key in $FOUNDRY_KEY.
aliases:
	$(PY) scripts/mine_unmapped.py
	$(PY) scripts/propose_aliases.py
	$(PY) scripts/measure_alias_gain.py

rebuild:
	$(PY) scripts/rebuild_corpus.py --out recipe_ids_v2.npz
