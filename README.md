# SWE-bench Data Point Validator

A tool for validating SWE-bench data points using the official evaluation harness.

## Description

This validator allows you to:
- Load data points from JSON files located in the `data_points/` directory
- Convert them into the SWE-bench predictions format
- Run the evaluation using the official SWE-bench evaluation harness
- Verify that all tests in `FAIL_TO_PASS` and `PASS_TO_PASS` pass after applying the patch

## Installation

### Requirements
- Python 3.10+
- Docker (for running the evaluation harness)
- UV package manager

### Installing Dependencies
```bash
# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
uv sync
