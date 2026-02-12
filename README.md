# Benchmarking Framework for Concept Drift Detectors

This repository contains the **reference implementation** of the experimental framework presented in the paper:

**Yet Another Concept Drift Benchmark? A Principled Framework for Evaluating Drift Detectors**  
L. Iovine, G. Ziffer, E. Della Valle — *KDD 2026 (under review)*

The goal of this framework is to provide a **reproducible and principled benchmark** for evaluating **concept drift detectors** under well-defined alarm semantics and evaluation metrics.

---

## What is Included

- A unified **benchmarking pipeline** for concept drift detectors
- Formal **alarm semantics** (true detections, false alarms, repeated alarms)
- Normalized **evaluation metrics** and composite scores
- A **deterministic reference learner** (Sliding Heatmap) used as a controlled performance observer
- Integration with **MOA** and **CapyMOA**

> Sliding Heatmap is **not** proposed as a state-of-the-art classifier.  
> It is intentionally simple and deterministic, and is used only to ensure observability.

---

## Repository Structure

code:
- moa_java/ # Java (MOA) components
- python/ # Python wrappers and evaluation
- experiments/ # Benchmark scripts
- requirements.txt

---

## Dataset Setup

Due to size constraints, datasets are not included in this repository.

They can be downloaded automatically using:

```bash
pip install requests
python code/data/download_datasets.py
```

Alternatively, they are available at:
https://doi.org/10.5281/zenodo.18621582

---

## Setup

### 1. Create Environment

```bash
conda create -n shm-paper -c conda-forge python=3.11 -y
conda activate shm-paper
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Building Java Components

```bash
cd code/moa_java
mvn -DskipTests clean package
cp target/slidingheatmap-moa-1.0.0.jar ../python/slidingheatmap_capymoa/jars/
```

## Pyhton Dependencies
```makefile
numpy==1.26.4
torch==2.2.2
capymoa==0.12.0
jpype1==1.6.0
typing_extensions
```

## Citation
