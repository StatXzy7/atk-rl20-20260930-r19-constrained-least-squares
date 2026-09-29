# Rank-deficient constrained least squares

A NumPy-only numerical utility for

\[
\min_x \|Ax-b\|_2 \quad\text{subject to } Cx=d,
\]

including rank-deficient `A` and `C`. When multiple constrained least-squares
solutions exist, the utility returns the minimum Euclidean-norm minimizer.

The algorithm uses truncated SVD null-space methods. It never forms or inverts
a singular KKT matrix and does not use an arbitrary ridge penalty.

## Install

No package installation is required beyond NumPy:

```bash
python3 -m unittest discover -s tests
```

## Usage

```python
import numpy as np
from constrained_lsq import constrained_lstsq

A = np.diag([1.0, 2.0, 0.0])
b = np.array([0.0, 1.0, 1.0])
C = np.array([[1.0, 1.0, 1.0]])
d = np.array([1.0])

result = constrained_lstsq(A, b, C, d)
print(result.x)
print(result.status)
print(result.constraint_residual_norm)
print(result.projected_gradient_norm)
```

For ordinary rank-deficient least squares, omit `C` and `d`. If constraints
are numerically incompatible, `result.status` is
`incompatible_constraints`, `result.x` is `None`, and
`result.infeasibility_certificate` contains a normalized left-null direction
satisfying `y.T @ C ≈ 0` with nonzero `y.T @ d`.

## Numerical controls

- `rank_tol`: relative SVD rank tolerance; defaults to `max(rows, cols)*eps`.
  Singular values are retained when `sigma_i > rank_tol * sigma_1`.
- `feasibility_tol`: normalized constraint compatibility tolerance, default
  `1e-11`.
- `uncertain_factor`: warning-band factor, default `8`.

Thresholds are relative and returned explicitly in `result.thresholds`; all
singular values, ranks, effective condition numbers, residuals, particular and
null-space components, and projected-gradient diagnostics are available on the
result object. A warning near a rank or feasibility threshold reports
uncertainty rather than claiming an exact floating-point algebraic proof.

## Documentation and report

- Mathematical derivation: `docs/math.md`
- Generated diagnostics: `docs/diagnostic_report.md`
- Regenerate the report:

```bash
python3 examples/generate_report.py
```

The report covers an integer/rational case, redundant and incompatible
constraints, orthogonal coordinate changes, zero matrices, a singular-value
threshold sweep with solution sensitivity, and a column-scaling example
showing why selecting the minimum norm in scaled coordinates changes the
meaning of “minimum norm”.
