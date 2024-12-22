# A deep neural network for oxidative coupling of methane trained on high-throughput experimental data

## Abstract

In this work, we develop a deep neural network model for the reaction rate of oxidative coupling of methane (OCM) using high-throughput experimental catalysis data. The model is formulated such that the reaction rate satisfies the plug flow reactor design equation. The neural network is then employed to understand the variation of reactant and product compositions within the reactor for the reference catalyst Mn–Na₂WO₄/SiO₂ at different temperatures. Additionally, the model is used to identify new catalysts and combinations of known catalysts that could increase yield and selectivity compared to the reference catalyst.

The model reveals that methane is primarily converted in the first half of the catalyst bed, while the second part consolidates the products and increases the ethylene to ethane ratio. A screening study of catalyst pairs consisting of previously studied catalysts (e.g., M₁(M₂)M₃Ox/support) demonstrates that a reactor configuration with two sequential catalyst beds leads to synergistic effects, resulting in increased C₂ yields compared to the reference catalyst under identical conditions and contact time.

An expanded screening study of 7400 combinations, including both previously studied and new catalyst permutations, identifies multiple catalyst combinations with enhanced C₂ yields. This study showcases the value of learning a deep neural network model for instantaneous reaction rates directly from high-throughput data and represents a significant step towards constraining a data-driven reaction model to adhere to domain-specific information.

## Table of Contents
1. [Installation](#installation)
2. [Model Overview](#model-overview)
3. [BibTeX](#bibtex)
4. [Acknowledgments](#acknowledgments)

## Installation

To use this model, ensure that you have Python 3.8+ and the necessary dependencies installed.

### Clone the repository

```bash
git clone https://github.com/kz29/OCM.git
cd OCM
```

### Install dependencies
```
pip install -r requirements.txt
```
### Running the model 
```
python main.py
```
## Model Overview
The deep neural network model is built using the PyTorch framework. It predicts the instantaneous reaction rate of oxidative coupling of methane based on input data such as inflows, temperature, catalyst properties, and contact time. The model satisfies the plug flow reactor design equation and is trained on high-throughput experimental data.

### Key Components:

* OCMmodule: This module defines the neural network layers and the forward pass logic, which combines the input features and passes them through multiple fully connected layers.
* OCMmodel: This class orchestrates the overall model, taking in multiple input variables (inflows, catalyst atom data, temperature, etc.), embedding the features, and passing them through the OCMmodule for prediction.
* Training: The model is trained on experimental data with a custom loss function and optimizer (Adam optimizer with a learning rate scheduler).
* Data
The model uses high-throughput experimental data to train and evaluate the performance. This includes inflows, outflows, catalyst properties, temperature, and contact time for various catalysts.

## BibTeX
If you find this work useful, please cite our paper:
```
@article{Ziu_2023,
doi = {10.1088/2515-7655/aca797},
url = {https://dx.doi.org/10.1088/2515-7655/aca797},
year = {2022},
month = {dec},
publisher = {IOP Publishing},
volume = {5},
number = {1},
pages = {014009},
author = {Klea Ziu and Ruben Solozabal and Srinivas Rangarajan and Martin Takáč},
title = {A deep neural network for oxidative coupling of methane trained on high-throughput experimental data},
journal = {Journal of Physics: Energy},
abstract = {In this work, we develop a deep neural network model for the reaction rate of oxidative coupling of methane from published high-throughput experimental catalysis data. A neural network is formulated so that the rate model satisfies the plug flow reactor design equation. The model is then employed to understand the variation of reactant and product composition within the reactor for the reference catalyst Mn–Na2WO4/SiO2 at different temperatures and to identify new catalysts and combinations of known catalysts that would increase yield and selectivity relative to the reference catalyst. The model revealed that methane is converted in the first half of the catalyst bed, while the second part largely consolidates the products (i.e. increases ethylene to ethane ratio). A screening study of  combinations of pairs of previously studied catalysts of the form M1(M2)M3O x /support (where M1, M2 and M3 are metals) revealed that a reactor configuration comprising two sequential catalyst beds leads to synergistic effects resulting in increased yield of C2 compared to the reference catalyst at identical conditions and contact time. Finally, an expanded screening study of 7400 combinations (comprising previously studied metals but with several new permutations) revealed multiple catalyst choices with enhanced yields of C2 products. This study demonstrates the value of learning a deep neural network model for the instantaneous reaction rate directly from high-throughput data and represents a first step in constraining a data-driven reaction model to satisfy domain information.}
}

```
## Acknowledgments
We thank the [paper](https://pubs.acs.org/doi/abs/10.1021/acscatal.9b04293) for sharing the dataset.
