# MNIST Lab

Comparing different neural network approaches (MLP, CNN, and an MLP written from scratch in NumPy) on MNIST and Fashion-MNIST.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Project structure

```
mnist_lab/
└── approaches/
    └── base.py      # the Approach interface every approach implements
```

## Adding a new approach

1. Create a file in `mnist_lab/approaches/`, for example `cnn.py`.
2. Subclass `Approach` from `base.py` and implement `fit` and `predict`.
3. Follow the data contract described in `base.py`:
   - `x`: images, shape `(n, 28, 28)`, float32, values in `[0, 1]`
   - `y`: labels, shape `(n,)`, int64, values `0–9`
   - `predict` returns probabilities, shape `(n, 10)`

```python
from src.mnist_lab.approaches.base import Approach

class CNNApproach(Approach):
    def fit(self, x_train, y_train):
        ...

    def predict(self, x):
        ...
```
