# SWE-bench Docker Architecture

This document outlines the Docker-based architecture used by SWE-bench for evaluating data points through automated patch testing. It details the multi-layer Docker system, image building workflow, test execution mechanisms, integration points, and how dependencies are managed.

---

## Docker Architecture Overview

SWE-bench employs a **3-layer Docker image architecture**:

```
Base Image  →  Environment Image  →  Instance Image
```

### 1. Base Image (`swe-bench/base`)

* **Purpose**: Provides foundational tools required for SWE-bench evaluation.
* **Contents**:

  * Ubuntu Linux (20.04 or similar)
  * Python (multiple versions supported)
  * Git, curl, unzip, etc.
  * `conda` or `pyenv` (optional)
  * SWE-bench CLI tools

### 2. Environment Image

* **Purpose**: Encapsulates the state of a specific repository (e.g., `pytorch/pytorch`) at a certain commit.
* **Built from**: Base image
* **Contents**:

  * Full clone of the repo at the evaluation commit (test or patch commit)
  * Environment setup (Python version, system-level packages)
  * All pip or conda dependencies required to run the repo's tests

### 3. Instance Image

* **Purpose**: Represents a fully evaluable snapshot for a single data point (e.g., a SWE-bench entry).
* **Built from**: Environment image
* **Contents**:

  * Source code at the patch commit (with or without patch applied)
  * Data point-specific dependencies (if required)
  * Test patch ready for evaluation

---

## Image Building Process

Docker images are built in the following stages:

### Base Image (One-time Setup)

```bash
cd docker/base
docker build -t swe-bench/base .
```

### Environment Image (Per Repo/Commit)

Built by the CLI tool during validation:

```bash
swebench validate --build-env-image --repo-id pytorch/pytorch --commit <sha>
```

Process:

* Clones the repo at target commit.
* Installs dependencies listed in `requirements.txt`, `environment.yml`, or `setup.py`.
* Uses pip or conda depending on repo.
* May patch the repo to bypass flaky tests or hardcoded failures.

### Instance Image (Per Data Point)

Automatically built before each test case:

```bash
swebench validate --build-instance --instance-id 12345
```

Process:

* Starts from the environment image.
* Applies data-point-specific setup (patch files, extra dependencies).
* Injects the `golden patch` if `--golden` mode is used.

---

## Test Execution Flow

Tests are executed inside Docker containers created from the Instance image. The following flow describes the execution:

### 1. Container Startup

```bash
docker run --rm -v /tmp:/tmp swe-bench/instance-12345
```

### 2. Patch Application Process

* If patch is required:

  * Patch is copied into the repo directory.
  * Applied using `git apply` or `patch -p1`.
  * Patch conflicts cause immediate failure.

### 3. Test Command Execution

* Entry point script defines test execution:

```bash
./run_tests.sh
```

* This may include:

  * Python test runners (e.g., `pytest`, `unittest`)
  * Custom test scripts (e.g., `scripts/run_tests.sh`)
* Timeout mechanism ensures test suite exits after a fixed duration (e.g., 600 seconds) to prevent hanging.

### 4. Output Parsing and Result Extraction

* Standard output and error are captured and logged.
* Success criteria are based on:

  * Zero exit code
  * Presence of “PASSED”/“FAILED” markers
  * Repo-specific regex matchers (if needed)
* Results are compiled into structured JSON for further scoring.

---

## Example: `pytorch/pytorch` FAIL\_TO\_PASS Test

### Setup:

* Base image is prebuilt
* Environment image built for commit: `a1b2c3d`
* Instance image created for data point `12345` with golden patch applied

### Execution:

```bash
docker run --rm swe-bench/instance-12345
```

### Internal Flow:

```bash
# Inside container
cd /workspace/repo
patch -p1 < patch.diff
./run_tests.sh > /tmp/test.log
```

### Output:

```json
{
  "result": "PASS",
  "duration_seconds": 182,
  "logs": "/tmp/test.log"
}
```

---

## Integration Points

The SWE-bench validator uses this architecture as follows:

* **Environment Detection**: Chooses the correct environment image for each repo and commit.
* **Instance Management**: Builds per-data-point images on-the-fly if not cached.
* **Execution Control**: Orchestrates test execution, timeouts, and result collection.
* **Cleanup**: Removes instance containers/images after execution to save space.

Validator CLI commands involved:

```bash
swebench validate \
  --repo-id pytorch/pytorch \
  --instance-id 12345 \
  --golden \
  --timeout 600 \
  --verbose
```

---

## Data Point Dependency Handling

Some SWE-bench data points require custom Python packages or tools to be installed.

### Where They Are Installed:

* **During instance image creation** (after environment image)
* Using pip or conda:

```bash
pip install -r datapoint-requirements.txt
```

* Custom install hooks may be provided in the data point JSON metadata.

### Example:

```json
"additional_requirements": [
  "numpy==1.23.0",
  "scikit-learn>=0.24"
]
```

Injected automatically by the validator into the Dockerfile of the instance image.

---

## Summary

SWE-bench uses a layered, efficient Docker architecture to isolate and evaluate software patches across a wide range of real-world repositories. The use of three-layer images ensures reproducibility, efficiency, and flexibility. The validator tool acts as the orchestrator, driving the image lifecycle and test execution within sandboxed environments.
