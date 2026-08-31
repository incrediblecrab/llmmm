PY := ./.venv/bin/python
PIP := ./.venv/bin/pip

.PHONY: help setup data test gate list baselines report clean check sanity recipes text coverage aliases rebuild docs

help:
	@echo "setup      create .venv and install the package"
	@echo "data       import artefacts and build the evaluation splits"
	@echo "gate       Phase-0 control gate — must pass before any result counts"
	@echo "test       run the test suite"
	@echo "sanity     adversarial checks against the recorded results"
	@echo "docs       fail if a number in the docs matches no artefact"
	@echo "check      gate + tests + sanity + docs, the full pre-commit check"
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
	@echo "installed. next: make data"

data:
	$(PY) scripts/import_data.py
	$(PY) scripts/build_splits.py

gate:
	$(PY) -m ingredient_model.cli gate

list:
	$(PY) -m ingredient_model.cli list

test:
	$(PY) -m pytest tests/ -q

# Re-derives the headline numbers by an independent path, and re-runs the
# controls that would expose a metric paying for something other than skill.
sanity:
	$(PY) scripts/sanity_check.py

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
	$(PY) scripts/check_docs.py

# The gate comes first deliberately. If the metrics rate random vectors above
# chance, every number produced afterwards is void, so there is no point running
# anything else.
check: gate test sanity docs

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
