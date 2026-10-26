PYTHON ?= python

test:
	$(PYTHON) -m pytest -m "not data"

cache:
	$(PYTHON) -m scripts.build_cache --splits TRAIN VAL

mask-check:
	$(PYTHON) -m scripts.mask_check

train:
	$(PYTHON) -m src.train --budget-min 85 --epochs 20 --batch-size 8 --train-limit 2000

evaluate:
	$(PYTHON) -m src.evaluate

echonet:
	$(PYTHON) -m src.echonet_eval

results:
	$(PYTHON) scripts/make_results_md.py

reproduce: test cache train evaluate results
