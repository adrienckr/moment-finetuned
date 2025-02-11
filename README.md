# MOMENT: A Family of Open Time-series Foundation Models

<div align="center">
    <img width="60%" alt="MOMENT" src="assets/MOMENT Logo.png">
    <h1>MOMENT: A Family of Open Time-series Foundation Models</h1>
    
    [![preprint](https://img.shields.io/static/v1?label=arXiv&message=2402.03885&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2402.03885)
    [![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/AutonLab/MOMENT-1-large)
    [![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/AutonLab/Timeseries-PILE)
    [![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/license/MIT)
    [![Python: 3.11](https://img.shields.io/badge/Python-3.11-blue)]()
</div>

## Overview

MOMENT is a family of open-source foundation models designed specifically for advanced time-series analysis. By leveraging large-scale multi-dataset pre-training and specialized architectural design, MOMENT supports a wide range of tasks including forecasting, classification, anomaly detection, imputation, and representation learning.

This document provides a comprehensive technical introduction, including details on model architecture, usage, and recent fine-tuning experiments on a prominent real-world dataset.

## 🔥 Latest Updates

- **New Fine-Tuning Experiment:**  
    The model has been fine-tuned on the [Store Sales Time-Series Forecasting dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) from Kaggle. This extends MOMENT's capability on retail and sales forecasting.
    
- **Forecasting Visualization:**  
    The output of the fine-tuned forecasting experiment is saved as `assets/output.png`. This graph demonstrates the model performance on the forecasting task using the Kaggle dataset.

- Fixed minor issues in multi-channel classification and improved data preprocessing routines.
- MOMENT has received further validation at ICML 2024.
    
## 📖 Introduction

Pre-training large models on time-series data is challenging due to the lack of a large cohesive repository and the diverse characteristics inherent to time-series data. MOMENT addresses these challenges by:

1. Compiling a diverse collection of public time-series from varied domains.
2. Employing patch-wise embedding where each time-series is segmented into fixed-length patches.
3. Using masking strategies during pre-training to allow reconstruction and task-specific adaptation.

The architecture has been carefully designed to handle multiple tasks without significant parameter updates, unifying forecasting, anomaly detection, classification, and more under one framework.

## Architecture in a Nutshell

- **Patch Embedding:**  
    A time-series is divided into fixed-length sub-sequences (patches). Each patch is embedded into a D-dimensional representation.

- **Masking Strategy:**  
    During pre-training, a uniform random subset of patch embeddings is replaced with a special `[MASK]` embedding. The model is trained to reconstruct the original time-series.

- **Task Adaptation:**  
    By masking and reconstruction, the model learns robust representations, readily adaptable to forecasting, imputation, and classification with minor modifications such as adding a final linear layer.

<div align="center">
    <img src="assets/moment_architecture.png" width="60%">
</div>

## 🧑‍💻 Usage

### Installation

Ensure Python 3.11 is installed. Install the MOMENT package via pip:

```bash
pip install momentfm
```

Alternatively, install the latest version directly from GitHub:

```bash
pip install git+https://github.com/moment-timeseries-foundation-model/moment.git
```

### Loading the Pre-trained Model

Below are usage examples for different tasks.

#### Forecasting

This example shows how to load the pre-trained forecasting model. For the fine-tuned version on the Kaggle Store Sales dataset, the forecast horizon can be adjusted according to the experimental setup.

```python
from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": 96  # Adjust horizon based on dataset specifics
        },
)
model.init()
```

The forecast output can be visualized using the graph stored at `assets/output.png`.

#### Classification

```python
from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
                "task_name": "classification",
                "n_channels": 1,
                "num_class": 2
        },
)
model.init()
```

#### Anomaly Detection, Imputation, and Pre-training

```python
from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "reconstruction"},
)
model.init()
```

#### Representation Learning

```python
from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "embedding"},
)
model.init()
```

## 📊 Fine-Tuning on the Kaggle Store Sales Dataset

The model was fine-tuned on the [Store Sales Time-Series Forecasting dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) to specifically target retail sales forecasting challenges.

- **Dataset Details:**  
    The dataset comprises historical store sales data requiring robust handling of seasonal trends, promotions, and external factors.
    
- **Technical Setup:**  
    The fine-tuning involved adjusting the pre-training model with task-specific layers while maintaining core patch embeddings. The forecasting head was optimized using customized loss functions reflective of retail sales metrics.
    
- **Results:**  
    The performance is visually summarized in `assets/output.png`, demonstrating competitive forecasting accuracy.

## 🧑‍🏫 Tutorials and Reproducible Experiments

- [Forecasting Tutorial](./tutorials/forecasting.ipynb)
- [Classification Tutorial](./tutorials/classification.ipynb)
- [Anomaly Detection Tutorial](./tutorials/anomaly_detection.ipynb)
- [Imputation Tutorial](./tutorials/imputation.ipynb)
- [Representation Learning Tutorial](./tutorials/representation_learning.ipynb)
- [Real-world ECG Case Study (Classification)](./tutorials/ptbxl_classification.ipynb)

These tutorials provide step-by-step instructions on using and fine-tuning MOMENT, with experiments reproducible on modest hardware resources (e.g., single NVIDIA A6000 GPU with 48 GiB RAM).

## BibTeX Reference

For academic citations:

```bibtex
@inproceedings{goswami2024moment,
    title={MOMENT: A Family of Open Time-series Foundation Models},
    author={Mononito Goswami and Konrad Szafer and Arjun Choudhry and Yifu Cai and Shuo Li and Artur Dubrawski},
    booktitle={International Conference on Machine Learning},
    year={2024}
}
```

## 🤝 Contributions

Contributions are welcome. Researchers and practitioners are encouraged to extend MOMENT by adding their datasets, new methods, and fine-tuning experiments. Special thanks to the contributors and community members.

**Fine-tuning and additional experiments by Adrien Cohen.**

## ⛑️ Research Code

For comprehensive experiments, refer to the complete research code available at [MOMENT Research](https://github.com/moment-timeseries-foundation-model/moment-research). This repository includes preprocessing scripts, training routines, and evaluation metrics for various baselines alongside MOMENT.

## 📰 Coverage & Related Work

For further reading on time-series foundation models and related efforts, please consult the following:

- [Moment: A Family of Open Time-Series Foundation Models](https://ai.plainenglish.io/moment-a-family-of-open-time-series-foundation-models-80f5135ca35b)
- [MOMENT: A Foundation Model for Time Series Forecasting, Classification, Anomaly Detection](https://towardsdatascience.com/moment-a-foundation-model-for-time-series-forecasting-classification-anomaly-detection-1e35f5b6ca76)
- [ICML 2024 Announcement](https://www.marktechpost.com/2024/05/15/cmu-researchers-propose-moment-a-family-of-open-source-machine-learning-foundation-models-for-general-purpose-time-series-analysis/)

## 🪪 License

MOMENT is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 Auton Lab, Carnegie Mellon University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:

[Full license text available at](https://opensource.org/license/MIT)
```

<div align="right">
    <img alt="CMU Logo" height="120px" src="assets/cmu_logo.png">
    <img alt="Auton Lab Logo" height="110px" src="assets/autonlab_logo.png">
</div>
# moment-finetuned
