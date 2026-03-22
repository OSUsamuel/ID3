# ID3 from Scratch

Self-paced project where I implemented the ID3 algorithm from scratch using NumPy.

[ID3 Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/ID3_algorithm)

## Overview

The ID3 (Iterative Dichotomiser 3) algorithm is a classic decision tree algorithm that uses **information gain** to select the best feature to split on at each node. This implementation is built from scratch without using any ML libraries for the core algorithm.

## How It Works

1. **Entropy** — measures the impurity of a dataset
2. **Conditional Entropy** — measures the entropy of the target given a feature
3. **Information Gain** — the reduction in entropy after splitting on a feature
4. The feature with the highest information gain is chosen as the split node
5. The process repeats recursively until a stopping condition is met

## Features

- ID3 decision tree built from scratch using NumPy and Pandas
- Feature discretization using `pd.cut` with consistent bin edges across train/test
- Nearest-interval fallback for out-of-range test values
- Tested on the Iris dataset

## Requirements

```
numpy
pandas
scikit-learn
```

Install with:

```bash
pip install numpy pandas scikit-learn
```

## Usage

```bash
python ID3.py
```
