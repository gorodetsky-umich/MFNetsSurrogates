init-e:
	python -m pip install -e .

init:
	python -m pip install .

check:
	ruff check 

format:
	ruff format

lint:
	python -m pylint mfnets_surrogates
	python -m flake8 mfnets_surrogates

type-check:
	python -m mypy 

test-mfnet:
	python -m unittest tests/test_mfnet.py -v

test-mfnet-torch:
	python -m unittest tests/test_mfnet_torch.py -v

test-mfnet-pyro:
	 python -m unittest tests/test_mfnet_pyro.py -v

test: test-mfnet test-mfnet-torch test-mfnet-pyro

ci: check format lint type-check test

.PHONY: init check format lint test type-check test-mfnet \
		test-mfnet-torch test-mfnet-pyro
