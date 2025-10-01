# Basics of Kolmogorov-Arnold Networks (KAN)

This repo contains notebooks with toy examples to build intuitive understanding of [Kolmogorov-Arnold Networks (KAN)](https://arxiv.org/abs/2404.19756). The repo contains a series of Jupyter notebooks to explore concepts and code to build KANs, designed to build your understanding of KANs gradually, starting from the basics of B-splines used as activation functions and progressing through more complex scenarios including symbolic regression.  

Original paper: [Liu et al. 2024, KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)

Original repository: [Prateek Gupta](https://github.com/pg2455/KAN-Tutorial)

## About the Tutorials
With the help of toy examples, notebooks are structured to help in understanding both the theoretical underpinnings and practical applications of KANs. 

1. [B-Splines for KAN](1_splines.ipynb): 
    - Understanding the mathematical construction of B-splines.
    - Exploring how B-splines are used for functional approximation.

2. [Deeper KANs](2_stacked_splines.ipynb)
   - Constructing and understanding [1, 1, 1, ..., 1] KAN configurations.
   - Implementing and exploring backpropagation through stacked splines.

3. [Grid Manipulation in KANs](3_grids.ipynb)
   - How to expand model's capacity through grid manipulation.
   - How KANs prevent catastrophic forgetting in continual learning?

4. [Symbolic Regression using KANs](4_symbolic_learning.ipynb)
   - Training KANs with fixed symbolic activation functions.
   - Understanding the implications of symbolic regression within neural networks.

## Prerequisites

To follow these tutorials, you should have a basic understanding of machine learning concepts and be familiar with Python programming. Experience with PyTorch and Jupyter Notebooks is also recommended.

## Contributions

Contributions to this tutorial series are welcome! If you have suggestions for improvement or want to add new examples, please feel free to submit a pull request or open an issue.

If you'd like to support my work, you can:
<a href="https://www.buymeacoffee.com/pratgpt" target="_blank">
  <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 45px; width: 162px;">
</a>

## Setup (Windows, macOS, Linux)

Follow these steps to run the notebooks:

### Option A: Using conda (recommended)

1. Install Miniconda or Anaconda
2. From the project root, create the environment:

```bash
conda env create -f environment.yml
```

3. Activate the environment:

```bash
conda activate kan-tutorial
```

4. (Optional) Create a .env from the example and adjust as desired:

```bash
copy example.env .env   # on Windows
# cp example.env .env   # on macOS/Linux
```

5. Launch JupyterLab:

```bash
jupyter lab
```

### Option B: Using pip + venv

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. (Optional) Copy environment example:

```bash
copy example.env .env   # on Windows
# cp example.env .env   # on macOS/Linux
```

4. Start JupyterLab:

```bash
jupyter lab
```

### Notes
- The notebooks and `utils.py` use `torch`, `numpy`, and `matplotlib`.
- The provided `environment.yml` pins Python 3.10 and CPU-only PyTorch by default. If you have a CUDA-capable GPU, replace `cpuonly` with an appropriate CUDA package per PyTorch installation instructions.