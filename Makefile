autoformat:
	set -e
	isort .
	black --config pyproject.toml .
	docformatter --config pyproject.toml -r --in-place .
	flake8

lint:
	set -e
	isort -c .
	black --check --config pyproject.toml .
	docformatter --config pyproject.toml -r --check .
	flake8

test:
	set -e
	coverage run -m pytest tests/

test-cov:
	set -e
	pytest tests/ --cov=./ --cov-report=xml

test-like-ga:
	set -e
	VOXEL_UNITTEST_DISABLE_DATA=true pytest tests/

build-docs:
	set -e
	mkdir -p docs/source/_static
	rm -rf docs/build
	rm -rf docs/source/generated
	cd docs && make html

dev:
	pip install -e .[dev,docs]
	pip install -r docs/requirements.txt
	pre-commit install

all: autoformat test build-docs