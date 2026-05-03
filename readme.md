# Sky Writer

This repository contains the code used in our research:
**"Automatic Recognition of Air-Writing Using Machine Learning"**

**Authors:** Taichi Oikawa, Yuki Sano, and Toichi Hiratsuka

## Overview

* **Background**: To solve the issue of stroke overlap in continuous air-writing, this study focuses on automatic character segmentation.
* **Methodology**: Hand landmarks captured via MediaPipe are classified into "Writing" or "Not Writing" states using a PyTorch-based LSTM model.
* **Key Results**: The optimized model achieved a high recall of over 95% for stroke detection, demonstrating precise recognition of writing intent.
* **Conclusion**: We proved that machine learning can effectively detect character boundaries, paving the way for intuitive air-writing interfaces.

## Requirements

- Python 3.11.9
- uv (>=0.11.0)

If GPU available:
- CUDA

## Used Libraries

- PyTorch
- scikit-learn
- MediaPipe
- google-cloud-vision
- opencv-python
- opencv-contrib-python
- Pillow
- seaborn
- matplotlib
- pandas
- numpy


## Get Started

```shell
# Clone repository
$ git clone https://github.com/TaichiOikawa/SkyWriter.git
$ cd SkyWriter
# Install libraries
$ uv sync
```

> [!NOTE]
> **PyTorch Installation**<br>
> This project uses `uv` for dependency management. If you have a different CUDA version, please update the index URL and `torch` version in `pyproject.toml` before running `uv sync`.

### 1. Collect Training Data

```shell
$ uv run capture_airwriting.py
# --movie <path>: Path to the movie file to process.
```

### 2. Prepare Data

```shell
# STEP1
$ uv run data_preparation/create_data_csv.py
# STEP2
$ uv run data_preparation/normalize_data.py
# STEP3
$ uv run data_preparation/add_features.py
```

### 3. Train

```shell
$ uv run train_separator.py
```

### 4. Run main.py

```shell
$ uv run main.py
```

### (if you need) Evaluate

```shell
$ uv run evaluate_separator.py
```
## References

- Fujimoto, T. (2020). *Automatic recognition of Kusyo by a monocular camera using deep learning* [Master's thesis, Japan Advanced Institute of Science and Technology]. JAIST Repository. https://hdl.handle.net/10119/16417

## Citation

See the [CITATION.cff](./CITATION.cff) file for details.

## License

Copyright 2026 Taichi Oikawa, Yuki Sano, and Toichi Hiratsuka

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
