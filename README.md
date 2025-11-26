# 🔢 Numerical Methods Toolkit

> A comprehensive Python implementation of classical numerical methods for root finding and solving systems of linear equations

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-orange.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Latest-green.svg)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Latest-red.svg)](https://matplotlib.org/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methods Implemented](#-methods-implemented)
- [Examples](#-examples)
- [Mathematical Background](#-mathematical-background)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project provides a hands-on implementation of fundamental numerical methods used in computational mathematics and engineering. It includes both **root-finding algorithms** for solving nonlinear equations and **direct methods** for solving systems of linear equations.

Perfect for:
- 📚 Students learning numerical analysis
- 🔬 Researchers needing quick numerical solutions
- 👨‍💻 Developers implementing computational algorithms
- 🎓 Educators demonstrating numerical methods

---

## ✨ Features

### 🔍 Root Finding Methods
- **Graphical Method** - Visual representation of roots
- **Bisection Method** - Reliable bracketing method
- **False Position (Regula Falsi)** - Improved bracketing approach
- **Fixed-Point Iteration** - Simple iterative technique
- **Newton-Raphson Method** - Fast convergence with derivatives
- **Modified Secant Method** - Derivative-free Newton variant

### 📐 Linear System Solvers
- **Naive Gauss Elimination** - Classic direct method
- **LU Decomposition** - Efficient factorization approach
- **Cholesky Decomposition** - Optimized for symmetric matrices
- **Matrix Inversion** - Using LU decomposition

---

## 🚀 Installation

### Prerequisites

```bash
Python 3.7 or higher
```

### Required Libraries

```bash
pip install numpy scipy matplotlib
```

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/numerical-method.git
cd numerical-method
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open the Jupyter notebook:
```bash
jupyter notebook Numerical-Method-Project.ipynb
```

---

## 💻 Usage

### Root Finding Example

```python
import numpy as np
from scipy import optimize

# Define the function
def f(x):
    return x**3 - 6*x**2 + 11*x - 6.1

# Using Bisection Method
def bisection_method(a, b, tol):
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

# Find root
root = bisection_method(0.5, 1.5, 0.0005)
print(f"Root: {root}")
```

### Linear System Example

```python
import numpy as np

# Define system: Ax = b
A = np.array([[10, 2, -1],
              [-3, 6, 2],
              [1, 1, 5]])
b = np.array([27, -61.5, -21.5])

# Solve using LU Decomposition
from scipy.linalg import lu_factor, lu_solve

lu, piv = lu_factor(A)
x = lu_solve((lu, piv), b)
print(f"Solution: {x}")
```

---

## 🧮 Methods Implemented

### 1️⃣ Root Finding Methods

| Method | Convergence | Requires Derivative | Best For |
|--------|-------------|---------------------|----------|
| Graphical | Visual | ❌ | Understanding behavior |
| Bisection | Linear | ❌ | Guaranteed convergence |
| False Position | Superlinear | ❌ | Better than bisection |
| Fixed-Point | Linear | ❌ | Simple iterations |
| Newton-Raphson | Quadratic | ✅ | Fast convergence |
| Modified Secant | Superlinear | ❌ | Derivative-free speed |

### 2️⃣ Linear System Solvers

#### Problem Statement
Solve the system:
```
10x₁ + 2x₂ - x₃ = 27
-3x₁ - 6x₂ + 2x₃ = -61.5
x₁ + x₂ + 5x₃ = -21.5
```

| Method | Complexity | Memory | Special Requirements |
|--------|-----------|---------|---------------------|
| Gauss Elimination | O(n³) | O(n²) | None |
| LU Decomposition | O(n³) | O(n²) | None |
| Cholesky | O(n³/3) | O(n²) | Symmetric positive-definite |

---

## 📊 Examples

### Visualizing Root Finding

The project includes beautiful visualizations of the root-finding process:

```python
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 1000)
plt.plot(x, f(x), label='f(x)', color='darkred', linewidth=1.25)
plt.axhline(y=0, color='darkblue', linestyle='--', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Graphical Method')
plt.legend()
plt.show()
```

### Newton-Raphson Visualization

The notebook includes plots showing both the function and its derivative, with iteration points marked in red.

---

## 📐 Mathematical Background

### Root Finding Problem

Find **x** such that:
```
f(x) = x³ - 6x² + 11x - 6.1 = 0
```

**Positive real roots:**
- x ≈ 1.0543
- x ≈ 1.8990
- x ≈ 3.0467

### Linear System Problem

Matrix form: **Ax = b**

```
⎡ 10   2  -1 ⎤ ⎡ x₁ ⎤   ⎡  27.0 ⎤
⎢ -3   6   2 ⎥ ⎢ x₂ ⎥ = ⎢ -61.5 ⎥
⎣  1   1   5 ⎦ ⎣ x₃ ⎦   ⎣ -21.5 ⎦
```

**Solution:**
- x₁ ≈ 3.7693
- x₂ ≈ -7.1579
- x₃ ≈ -3.6223

---

## 🎨 Features Highlights

- 📈 **Interactive Visualizations** - See methods in action
- 🎯 **High Precision** - Configurable tolerance levels
- 🔄 **Iterative Tracking** - Monitor convergence progress
- 📝 **Well-Documented** - Clear explanations and comments
- 🧪 **Tested Algorithms** - Verified against known solutions

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔃 Open a Pull Request

### Ideas for Contributions
- Add more numerical methods (Secant, Muller's, etc.)
- Implement iterative methods (Jacobi, Gauss-Seidel)
- Add error analysis and convergence plots
- Create interactive widgets for parameter tuning
- Improve documentation and examples

---

## 📚 References

- Chapra, S. C., & Canale, R. P. (2015). *Numerical Methods for Engineers*
- Burden, R. L., & Faires, J. D. (2010). *Numerical Analysis*
- Press, W. H., et al. (2007). *Numerical Recipes: The Art of Scientific Computing*

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

Created with ❤️ for the numerical methods community in Petra Christian University

---

## 🌟 Acknowledgments

- NumPy and SciPy communities for excellent numerical libraries
- Matplotlib for powerful visualization tools
- All contributors and users of this project

---

<div align="center">

**⭐ Star this repo if you find it helpful! ⭐**

Made with 🔢 and Python

</div>
