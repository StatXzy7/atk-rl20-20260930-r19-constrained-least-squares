"""Rank-deficient constrained linear least squares.

Solve  min ||A x - b||_2  subject to  C x = d  in the *raw* Euclidean
coordinates of x.  Rank-deficient A and C are supported.  When several
minimizers exist, the minimum-Euclidean-norm minimizer is returned.

The algorithm uses only SVD bases and explicit null-space projections.
It never forms or factors a KKT matrix, adds a ridge, or calls a black-box
least-squares solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

EPS = np.finfo(float).eps


@dataclass
class SVDModel:
    """SVD, numerical-rank decision, and rank diagnostics for a matrix."""

    name: str
    shape: tuple[int, int]
    U: np.ndarray
    s: np.ndarray
    Vt: np.ndarray
    rank: int
    cutoff: float
    atol: float
    rtol: float
    uncertainty_band: float
    borderline: bool
    nearest_threshold_gap: float
    condition_number_full: float
    condition_number_effective: float

    @property
    def smallest_retained_singular_value(self) -> float | None:
        return float(self.s[self.rank - 1]) if self.rank > 0 else None

    @property
    def largest_excluded_singular_value(self) -> float | None:
        return float(self.s[self.rank]) if self.rank < len(self.s) else None


@dataclass
class ConstrainedLSResult:
    success: bool
    status: str
    message: str
    x: np.ndarray | None
    objective: float | None = None
    least_squares_residual_norm: float | None = None
    constraint_residual_norm: float | None = None
    projected_gradient_norm: float | None = None
    x_norm: float | None = None
    particular_norm: float | None = None
    nullspace_correction_norm: float | None = None
    particular_correction_inner_product: float | None = None
    constraint_model_residual_norm: float | None = None
    constraint_tol: float | None = None
    achieved_constraint_tol: float | None = None
    projected_gradient_tol: float | None = None
    active_fit_norm: float | None = None
    active_fit_tol: float | None = None
    incompatibility_amount: float | None = None
    incompatibility_direction: np.ndarray | None = None
    left_nullspace_coefficients: np.ndarray | None = None
    C_model: SVDModel | None = None
    reduced_A_model: SVDModel | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-compishable summary containing main diagnostics."""

        def scalar(value: Any) -> Any:
            if value is None or isinstance(value, (str, bool, int)):
                return value
            if isinstance(value, (np.floating, float)):
                if np.isinf(value):
                    return "inf"
                return float(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            return value

        keys = [
            "success",
            "status",
            "message",
            "objective",
            "least_squares_residual_norm",
            "constraint_residual_norm",
            "projected_gradient_norm",
            "x_norm",
            "particular_norm",
            "nullspace_correction_norm",
            "constraint_model_residual_norm",
            "constraint_tol",
            "achieved_constraint_tol",
            "projected_gradient_tol",
            "incompatibility_amount",
            "warnings",
        ]
        out = {key: scalar(getattr(self, key)) for key in keys}
        out["x"] = scalar(self.x)
        if self.C_model is not None:
            out["C_rank"] = self.C_model.rank
            out["C_singular_values"] = self.C_model.s.tolist()
            out["C_cutoff"] = self.C_model.cutoff
            out["C_condition_effective"] = scalar(
                self.C_model.condition_number_effective
            )
        if self.reduced_A_model is not None:
            out["reduced_A_rank"] = self.reduced_A_model.rank
            out["reduced_A_singular_values"] = self.reduced_A_model.s.tolist()
            out["reduced_A_cutoff"] = self.reduced_A_model.cutoff
            out["reduced_A_condition_effective"] = scalar(
                self.reduced_A_model.condition_number_effective
            )
        return out


def _as_vector(name: str, value: Any, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != length:
        raise ValueError(f"{name} must have shape ({length},)")
    return arr


def _full_svd(name: str, M: np.ndarray, rtol: float, atol: float) -> SVDModel:
    p, q = M.shape
    if p == 0 or q == 0:
        U = np.eye(p)
        s = np.zeros(0)
        Vt = np.eye(q)
    else:
        U, s, Vt = np.linalg.svd(M, full_matrices=True)

    smax = float(s[0]) if len(s) else 0.0
    cutoff = float(atol + rtol * smax)
    rank = int(np.count_nonzero(s > cutoff))
    band = cutoff

    positive = s[s > 0.0]
    nearest_gap = float(np.min(np.abs(positive - cutoff))) if len(positive) else np.inf
    if cutoff > 0.0 and len(positive):
        ratios = positive / cutoff
        borderline = bool(np.any((ratios >= 0.5) & (ratios <= 2.0)))
    else:
        borderline = False

    cond_full = float(s[0] / s[-1]) if len(s) and s[-1] > 0 else np.inf
    cond_eff = float(s[0] / s[rank - 1]) if rank > 0 else np.inf

    return SVDModel(
        name=name,
        shape=(p, q),
        U=U,
        s=s,
        Vt=Vt,
        rank=rank,
        cutoff=cutoff,
        atol=float(atol),
        rtol=float(rtol),
        uncertainty_band=float(band),
        borderline=borderline,
        nearest_threshold_gap=nearest_gap,
        condition_number_full=cond_full,
        condition_number_effective=cond_eff,
    )


def constrained_lsq(
    A: Any,
    b: Any,
    C: Any = None,
    d: Any = None,
    *,
    rtol: float | None = None,
    atol: float = 0.0,
    c_rtol: float | None = None,
    a_rtol: float | None = None,
    c_atol: float | None = None,
    a_atol: float | None = None,
    constraint_tol: float | None = None,
) -> ConstrainedLSResult:
    """Solve a possibly rank-deficient equality-constrained LS problem.

    Rank convention: singular value ``s_i`` is retained when
    ``s_i > atol + rtol * max_i s_i``.  A per-matrix default
    ``rtol = max(shape) * eps`` is used when an argument is ``None``.

    Constraint feasibility is tested in the numerical left null space of
    ``C``.  The default feasibility tolerance is
    ``max(C.shape) * eps * ||d||_2``.  No implicit absolute tolerance is
    added, so a nonzero unit-sized residual on a nominally zero constraint
    direction is not silently accepted.

    Parameters
    ----------
    A, b:
        Least-squares matrix and right-hand side.
    C, d:
        Equality constraints. ``C=None`` means an unconstrained problem.
    rtol, atol:
        Default relative and absolute singular-value cutoffs for both ``C``
        and the reduced matrix ``A N``, where ``N`` spans null(C).
    c_rtol, a_rtol, c_atol, a_atol:
        Optional matrix-specific overrides.
    constraint_tol:
        Optional override for the numerical feasibility tolerance.
    """

    A_arr = np.asarray(A, dtype=float)
    if A_arr.ndim != 2:
        raise ValueError("A must be two-dimensional")
    p, n = A_arr.shape
    b_arr = _as_vector("b", b, p)

    if C is None:
        C_arr = np.zeros((0, n), dtype=float)
        d_arr = np.zeros(0, dtype=float)
    else:
        C_arr = np.asarray(C, dtype=float)
        if C_arr.ndim != 2 or C_arr.shape[1] != n:
            raise ValueError(f"C must be two-dimensional with {n} columns")
        m = C_arr.shape[0]
        if d is None:
            d_arr = np.zeros(m, dtype=float)
        else:
            d_arr = _as_vector("d", d, m)

    m = C_arr.shape[0]
    c_rtol_eff = max(m, n, 1) * EPS if c_rtol is None else c_rtol
    a_rtol_eff = max(p, n, 1) * EPS if a_rtol is None else a_rtol
    if rtol is not None:
        if c_rtol is None:
            c_rtol_eff = rtol
        if a_rtol is None:
            a_rtol_eff = rtol
    c_atol_eff = atol if c_atol is None else c_atol
    a_atol_eff = atol if a_atol is None else a_atol

    c_model = _full_svd("C", C_arr, c_rtol_eff, c_atol_eff)
    r = c_model.rank
    Uc, sc, Vtc = c_model.U, c_model.s, c_model.Vt

    if constraint_tol is None:
        dnorm = float(np.linalg.norm(d_arr))
        c_feas_tol = max(m, n, 1) * EPS * dnorm
    else:
        c_feas_tol = float(constraint_tol)

    tail = Uc[:, r:].T @ d_arr if m - r > 0 else np.zeros(0)
    tail_norm = float(np.linalg.norm(tail))

    warnings: list[str] = []
    if c_model.borderline:
        warnings.append(
            "A singular value of C is within a factor-of-two uncertainty band "
            "around the chosen rank cutoff; feasibility and rank are model-dependent rather "
            "than an exact algebraic proof."
        )

    if tail_norm > c_feas_tol:
        idx = int(np.argmax(np.abs(tail)))
        direction = Uc[:, r + idx]
        amount = float(direction @ d_arr)
        status = (
            "incompatible_rank_ambiguous"
            if c_model.borderline
            else "incompatible_constraints"
        )
        message = (
            "d has a component in the numerical left null space of C larger "
            "than constraint_tol."
        )
        if c_model.borderline:
            message += " The nearby rank cutoff makes this classification ambiguous."
        return ConstrainedLSResult(
            success=False,
            status=status,
            message=message,
            x=None,
            constraint_model_residual_norm=tail_norm,
            constraint_tol=c_feas_tol,
            incompatibility_amount=amount,
            incompatibility_direction=direction,
            left_nullspace_coefficients=tail,
            C_model=c_model,
            warnings=warnings,
        )

    Ur = Uc[:, :r]
    Vr = Vtc[:r, :].T
    if r > 0:
        x_particular = Vr @ ((Ur.T @ d_arr) / sc[:r])
    else:
        x_particular = np.zeros(n, dtype=float)

    N = Vtc[r:, :].T
    null_dim = n - r
    B = A_arr @ N if null_dim > 0 else np.empty((p, 0), dtype=float)
    a_model = _full_svd("A_N", B, a_rtol_eff, a_atol_eff)
    q = a_model.rank
    Ua, sa, Vta = a_model.U, a_model.s, a_model.Vt

    g = b_arr - A_arr @ x_particular
    if q > 0:
        y = Vta[:q, :].T @ ((Ua[:, :q].T @ g) / sa[:q])
    else:
        y = np.zeros(null_dim, dtype=float)

    correction = N @ y if null_dim > 0 else np.zeros(n, dtype=float)
    x = x_particular + correction
    residual = b_arr - A_arr @ x
    residual_norm = float(np.linalg.norm(residual))
    constraint_residual = C_arr @ x - d_arr if m > 0 else np.zeros(0)
    constraint_residual_norm = float(np.linalg.norm(constraint_residual))

    projected_gradient = B.T @ residual if null_dim > 0 else np.zeros(0)
    projected_gradient_norm = float(np.linalg.norm(projected_gradient))

    active_fit = Ua[:, :q].T @ residual if q > 0 else np.zeros(0)
    active_fit_norm = float(np.linalg.norm(active_fit))

    x_norm = float(np.linalg.norm(x))
    achieved_c_tol = c_feas_tol + max(m, n, 1) * EPS * (
        float(np.linalg.norm(C_arr)) * x_norm + float(np.linalg.norm(d_arr))
    )
    grad_tol = max(p, null_dim, 1) * EPS * (
        float(np.linalg.norm(B)) * residual_norm
    )
    active_tol = max(p, null_dim, 1) * EPS * float(np.linalg.norm(g))

    if a_model.borderline:
        warnings.append(
            "A singular value of the reduced least-squares matrix is within a "
            "factor-of-two uncertainty band around the chosen rank cutoff; the split between "
            "fitted and residual components is sensitive to the threshold."
        )
    if null_dim == 0:
        warnings.append(
            "The numerical feasible subspace has dimension zero; the compatible "
            "constraints determine x uniquely."
        )
    if m == 0:
        warnings.append("No constraints were supplied; this is ordinary rank-deficient LS.")
    if p == 0:
        warnings.append("A has no rows; the least-squares objective is identically zero.")
    if a_model.rank == 0 and null_dim > 0:
        warnings.append(
            "A has no nonzero action on null(C); the minimum-norm feasible point "
            "is therefore selected."
        )

    status = "rank_ambiguous" if c_model.borderline or a_model.borderline else "ok"
    message = (
        "Returning the minimum-Euclidean-norm solution under the stated numerical "
        "rank model."
    )

    return ConstrainedLSResult(
        success=True,
        status=status,
        message=message,
        x=x,
        objective=residual_norm,
        least_squares_residual_norm=residual_norm,
        constraint_residual_norm=constraint_residual_norm,
        projected_gradient_norm=projected_gradient_norm,
        x_norm=x_norm,
        particular_norm=float(np.linalg.norm(x_particular)),
        nullspace_correction_norm=float(np.linalg.norm(y)),
        particular_correction_inner_product=float(x_particular @ correction),
        constraint_model_residual_norm=tail_norm,
        constraint_tol=c_feas_tol,
        achieved_constraint_tol=achieved_c_tol,
        projected_gradient_tol=grad_tol,
        active_fit_norm=active_fit_norm,
        active_fit_tol=active_tol,
        C_model=c_model,
        reduced_A_model=a_model,
        warnings=warnings,
    )
