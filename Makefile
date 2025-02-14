# Makefile for Divine Algorithm project

.PHONY: install test lint clean build docs

# Python interpreter to use
PYTHON = python3
PIP = pip3

# Project directories
SRC_DIR = src
TEST_DIR = src/tests
BUILD_DIR = build
DIST_DIR = dist

# Install dependencies and package in development mode
install:
	$(PIP) install -e ".[dev]"

# Run tests with pytest
test:
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing

# Run quick tests without coverage
test-quick:
	pytest $(TEST_DIR) -v

# Run linting checks
lint:
	black $(SRC_DIR)
	isort $(SRC_DIR)
	pylint $(SRC_DIR)
	mypy $(SRC_DIR)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(DIST_DIR)
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# Build package distribution
build: clean
	$(PYTHON) setup.py sdist bdist_wheel

# Run all quality checks
quality: lint test

# Install in development mode with all extras
dev-install:
	$(PIP) install -e ".[dev]"

# Run the main program
run:
	$(PYTHON) -m divine_algorithm

# Create virtual environment
venv:
	$(PYTHON) -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

.DEFAULT_GOAL := help

# Help target
help:
	@echo "Divine Algorithm Makefile commands:"
	@echo "make install      - Install package and dependencies"
	@echo "make test        - Run tests with coverage"
	@echo "make test-quick  - Run tests without coverage"
	@echo "make lint        - Run linting checks"
	@echo "make clean       - Clean build artifacts"
	@echo "make build       - Build package distribution"
	@echo "make quality     - Run all quality checks"
	@echo "make dev-install - Install in development mode"
	@echo "make run         - Run the main program"
	@echo "make venv        - Create virtual environment"