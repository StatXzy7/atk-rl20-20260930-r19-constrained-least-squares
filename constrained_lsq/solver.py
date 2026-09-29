"""Minimum-norm constrained least squares via null-space SVD.

The solver deliberately avoids forming or inverting a KKT matrix.  It uses only
truncated SVD and dense linear algebra on non-singular reduced problems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

EPS = float(np.finfo(float).eps)


@dataclass
class ConstrainedLSTSQResult:
    """Result of :func:`constrained_lstsq`.

    ``x`` is ``None`` only when the constraints are diagnosed as incompatible.
    Diagnostics are plain numerical values/arrays so that callers can inspect
    every numerical-rank decision independently.
    """

    status: str
    x: np.ndarray | None
    residual_norm: float
    constraint_residual_norm: float
    projected_gradient_norm: float
    rank_A: int
    rank_C: int
    rank_AN: int
    nullity_C: int
    objective_multiplicity: bool
    minimizer_multiplicity: bool
    feasible_affine_dimension: int
    condition_A: float
    condition_C: float
    condition_AN: float
    singular_values_A: np.ndarray
    singular_values_C: np.ndarray
    singular_values_AN: np.ndarray
    thresholds: dict[str, float]
    feasibility_tol: float
    particular_solution: np.ndarray | None = None
    null_basis: np.ndarray | None = None
    null_coordinates: np.ndarray | None = None
    null_correction: np.ndarray | None = None
    infeasibility_certificate: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """True when an affine feasible set was solved numerically."""
        return self.status != "incompatible_constraints"


def _matrix(value: Any, name: str, expected_rows: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a matrix or a 1-D row vector")
    if expected_rows is not None and arr.shape[1] != expected_rows:
        raise ValueError(f"{name} must have {expected_rows} columns")
    return arr


def _vector(value: Any, name: str, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != length:
        raise ValueError(f"{name} must have length {length}")
    return arr


def _svd(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an economy-independent full SVD; empty matrices are handled."""
    return np.linalg.svd(mat, full_matrices=True)


def _rank_info(
    mat: np.ndarray,
    label: str,
    rank_tol: float,
    uncertain_factor: float = 8.0,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float, list[str]]:
    u, s, vh = _svd(mat)
    if s.size:
        threshold = rank_tol * float(s[0])
    else:
        threshold = 0.0
    accepted = s > threshold
    rank = int(np.count_nonzero(accepted))
    warnings: list[str] = []
    if rank > 0 and rank < s.size:
        gap_sigma = float(s[rank - 1])
        next_sigma = float(s[rank])
        if gap_sigma <= uncertain_factor * threshold or next_sigma * uncertain_factor >= threshold:
            warnings.append(
                f"{label}: numerical rank {rank} is threshold-sensitive; "
                f"sigma_k={gap_sigma:.16g}, sigma_k+1={next_sigma:.16g}, "
                f"threshold={threshold:.16g}."
            )
    elif rank == 0 and s.size and threshold > 0.0 and float(s[0]) > 0.0 and float(s[0]) <= uncertain_factor * threshold:
        warnings.append(
            f"{label}: rank is numerically zero but its largest singular value "
            f"is close to the threshold."
        )
    return rank, s, u, vh, threshold, warnings


def _condition(s: np.ndarray, rank: int) -> float:
    if rank == 0:
        return float("inf")
    return float(s[0] / s[rank - 1])


def constrained_lstsq(
    A: np.ndarray,
    b: np.ndarray,
    C: np.ndarray | None = None,
    d: np.ndarray | None = None,
    *,
    rank_tol: float | None = None,
    feasibility_tol: float = 1e-11,
    uncertain_factor: float = 8.0,
) -> ConstrainedLSTSQResult:
    """Solve min ||Ax-b||_2 over Cx=d and return the minimum-norm minimizer.

    Parameters
    ----------
    A, b:
        Objective matrix and right-hand side.
    C, d:
        Equality constraints.  If omitted (or ``C`` has zero rows), the ordinary
        rank-deficient minimum-norm least-squares problem is solved.
    rank_tol:
        Relative singular-value rank tolerance.  Singular value ``sigma_i`` is
        retained when ``sigma_i > rank_tol * sigma_1``.  ``None`` uses the
        LAPACK-style ``max(rows, cols) * eps`` separately for each matrix.
    feasibility_tol:
        Relative feasibility tolerance.  The residual is compared after scaling
        by ``max(||d||_2, ||C||_F ||x_p||_2)``.  The unscaled residual is also
        returned.
    uncertain_factor:
        Factor defining rank/feasibility warning bands around a threshold.

    Returns
    -------
    ConstrainedLSTSQResult
        A numerical solution or a left-null-space infeasibility certificate.

    Notes
    -----
    No KKT matrix is formed.  A particular solution is selected in
    ``range(C.T)``; because the constraint null space is its orthogonal
    complement, this makes the particular solution and every null-space
    correction orthogonal.  Minimizing the norm over objective minimizers then
    requires a minimum-norm solution of the reduced null-space problem.
    """
    if rank_tol is not None and rank_tol < 0.0:
        raise ValueError("rank_tol must be non-negative")
    if not 0.0 <= feasibility_tol:
        raise ValueError("feasibility_tol must be non-negative")
    if uncertain_factor <= 0:
        raise ValueError("uncertain_factor must be positive")

    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a matrix")
    m, n = A.shape
    b = _vector(b, "b", m)

    warnings: list[str] = []
    unconstrained = C is None
    if unconstrained:
        if d is not None:
            raise ValueError("d cannot be supplied without C")
        C = np.empty((0, n), dtype=float)
        d = np.empty(0, dtype=float)
    else:
        C = _matrix(C, "C", n)
        d = _vector(d, "d", C.shape[0])
    p = C.shape[0]

    def tol_for(mat: np.ndarray) -> float:
        if rank_tol is not None:
            return float(rank_tol)
        return float(max(mat.shape) * EPS) if max(mat.shape) else 0.0

    rank_a, sa, ua, vha, at, wa = _rank_info(A, "A", tol_for(A), uncertain_factor)
    warnings.extend(wa)

    if unconstrained:
        rank_c, sc, uc, vhc = 0, np.empty(0), np.empty((0, 0)), np.eye(n)
        ct = 0.0
        xp = np.zeros(n)
        n_basis = np.eye(n)
    else:
        rank_c, sc, uc, vhc, ct, wc = _rank_info(C, "C", tol_for(C), uncertain_factor)
        warnings.extend(wc)
        vr = vhc.T[:, :rank_c]
        ur = uc[:, :rank_c]
        sr = sc[:rank_c]
        rhs = ur.T @ d
        xp = vr @ (rhs / sr)
        raw_res = d - C @ xp
        scale = max(float(np.linalg.norm(d)), float(np.linalg.norm(C, "fro") * np.linalg.norm(xp)))
        rel_res = float(np.linalg.norm(raw_res) / scale) if scale > 0.0 else 0.0
        near = tol_for(C) * uncertain_factor
        # A zero largest singular value can only mean an exactly zero C matrix;
        # flag any numerically tiny d component that may be intended as nonzero.
        if sc.size and float(sc[0]) <= near and np.max(np.abs(d)) <= uncertain_factor * EPS:
            warnings.append("C is numerically zero; feasibility depends on tiny d components.")
        full_row_rank = rank_c == p
        if rel_res > feasibility_tol:
            if full_row_rank:
                # C has no numerical left null space at the selected rank.  A
                # full-row-rank square/wide matrix is compatible in exact
                # arithmetic; a residual here is a conditioning warning.
                warnings.append(
                    "C has full numerical row rank but its pseudoinverse "
                    "residual exceeds the feasibility tolerance; check scaling "
                    "and conditioning rather than interpreting this as an "
                    "incompatible constraint system."
                )
            elif rel_res < feasibility_tol * uncertain_factor:
                warnings.append(
                    "Feasibility residual is within a factor of "
                    f"{uncertain_factor:g} of the tolerance; compatibility is numerically uncertain."
                )
            if rank_c < p:
                y = uc[:, rank_c]
            else:
                y = np.ones(p)
            yn = y / max(float(np.linalg.norm(y)), EPS)
            certificate = {
                "left_null_direction": yn,
                "evidence": float(yn @ d),
                "absolute_evidence": abs(float(yn @ d)),
                "yT_C": C.T @ yn,
                "yT_C_norm": float(np.linalg.norm(C.T @ yn)),
                "d_projection_onto_range_C": uc[:, :rank_c] @ (uc[:, :rank_c].T @ d),
                "feasibility_residual": raw_res,
                "normalized_feasibility_residual": rel_res,
            }
            warnings.append(
                "Incompatibility is a floating-point certificate at the stated rank "
                "and feasibility tolerances, not an exact algebraic proof."
            )
            return ConstrainedLSTSQResult(
                status="incompatible_constraints",
                x=None,
                residual_norm=float("nan"),
                constraint_residual_norm=float(np.linalg.norm(raw_res)),
                projected_gradient_norm=float("nan"),
                rank_A=rank_a,
                rank_C=rank_c,
                rank_AN=0,
                nullity_C=n - rank_c,
                objective_multiplicity=n - rank_a > 0,
                minimizer_multiplicity=False,
                feasible_affine_dimension=0,
                condition_A=_condition(sa, rank_a),
                condition_C=_condition(sc, rank_c),
                condition_AN=float("inf"),
                singular_values_A=sa,
                singular_values_C=sc,
                singular_values_AN=np.empty(0),
                thresholds={"A": at, "C": ct, "AN": 0.0},
                feasibility_tol=float(feasibility_tol),
                infeasibility_certificate=certificate,
                warnings=warnings,
            )
        if rel_res > feasibility_tol / uncertain_factor:
            warnings.append(
                "Constraint residual is close to the feasibility tolerance; "
                "the compatible interpretation is threshold-sensitive."
            )
        n_basis = vhc.T[:, rank_c:]

    q = n - rank_c
    z = np.empty(0)
    correction = np.zeros(n)
    x = xp
    reduced_gradient = np.empty(0)
    grad_proj = np.full(n, np.nan)
    projected_gradient = float("nan")
    sg = np.empty(0)
    gt = 0.0
    rank_g = 0
    condition_g = float("nan")

    if q > 0:
        G = A @ n_basis
        rank_g, sg, ug, vhg, gt, wg = _rank_info(
            G, "A*N(C)", tol_for(G), uncertain_factor
        )
        if not unconstrained:
            warnings.extend(wg)
        rhs = b - A @ xp
        # Minimum-norm z selects the minimum-norm point in the affine set of
        # reduced objective minimizers.
        z_components = (ug[:, :rank_g].T @ rhs) / sg[:rank_g]
        z = vhg.T[:, :rank_g] @ z_components
        correction = n_basis @ z
        x = xp + correction
        reduced_gradient = n_basis.T @ (A.T @ (A @ x - b))
        grad_proj = n_basis @ reduced_gradient
        projected_gradient = float(np.linalg.norm(reduced_gradient))
        condition_g = _condition(sg, rank_g)

    objective_residual = b - A @ x
    residual_norm = float(np.linalg.norm(objective_residual))
    constraint_residual = d - C @ x
    constraint_residual_norm = float(np.linalg.norm(constraint_residual))

    objective_multiplicity = rank_a < n
    feasible_dimension = q
    minimizer_multiplicity = q > rank_g if q > 0 else False

    # Among feasible objective minimizers, x is orthogonal to
    # ker(A) intersect ker(C).  Particular/null orthogonality certifies the
    # separate, norm-separable construction used above.
    if q > 0 and rank_g < q:
        common_null = n_basis @ vhg.T[:, rank_g:]
        min_norm_projection = float(np.linalg.norm(common_null.T @ x))
        common_nullity = q - rank_g
    else:
        common_null = np.zeros((n, 0))
        min_norm_projection = 0.0
        common_nullity = 0
    particular_null_orthogonality = float(abs(xp @ correction))
    normal_residual = projected_gradient if q > 0 else float("nan")
    constraint_scale = max(float(np.linalg.norm(d)), float(np.linalg.norm(C, "fro") * np.linalg.norm(xp)))
    rel_constraint_residual = (
        constraint_residual_norm / constraint_scale if constraint_scale > 0 else 0.0
    )
    if feasible_dimension == 0:
        status = "unique_feasible_solution"
    elif rank_g == 0:
        status = "objective_invariant_on_feasible_set"
    elif rank_g < q:
        status = "multiple_minimizers_minimum_norm_selected"
    else:
        status = "unique_constrained_minimizer"
    if unconstrained:
        status = "unconstrained_" + status

    diagnostics = {
        "x_norm": float(np.linalg.norm(x)),
        "particular_solution_norm": float(np.linalg.norm(xp)),
        "null_correction_norm": float(np.linalg.norm(correction)),
        "particular_null_orthogonality": particular_null_orthogonality,
        "minimum_norm_projection_onto_common_ker_A_ker_C": min_norm_projection,
        "common_nullity": common_nullity,
        "common_null_basis": common_null,
        "objective_gradient": A.T @ (A @ x - b),
        "objective_gradient_norm": float(np.linalg.norm(A.T @ (A @ x - b))),
        "projected_gradient": grad_proj if q > 0 else np.full(n, np.nan),
        "reduced_gradient": reduced_gradient,
        "constraint_normal_gradient_residual": normal_residual,
        "least_squares_residual": objective_residual,
        "constraint_residual": constraint_residual,
        "normalized_constraint_residual": rel_constraint_residual,
        "feasibility_scale": constraint_scale,
        "sigma_min_used_A": float(sa[rank_a - 1]) if rank_a else 0.0,
        "sigma_next_A": float(sa[rank_a]) if rank_a < len(sa) else 0.0,
        "sigma_min_used_C": float(sc[rank_c - 1]) if rank_c else 0.0,
        "sigma_next_C": float(sc[rank_c]) if rank_c < len(sc) else 0.0,
        "sigma_min_used_AN": float(sg[rank_g - 1]) if rank_g else 0.0,
        "sigma_next_AN": float(sg[rank_g]) if rank_g < len(sg) else 0.0,
        "unconstrained": unconstrained,
        "rank_tolerance_parameter": None if rank_tol is None else float(rank_tol),
        "uncertain_factor": float(uncertain_factor),
    }

    return ConstrainedLSTSQResult(
        status=status,
        x=x,
        residual_norm=residual_norm,
        constraint_residual_norm=constraint_residual_norm,
        projected_gradient_norm=projected_gradient,
        rank_A=rank_a,
        rank_C=rank_c,
        rank_AN=rank_g,
        nullity_C=q,
        objective_multiplicity=objective_multiplicity,
        minimizer_multiplicity=minimizer_multiplicity,
        feasible_affine_dimension=feasible_dimension,
        condition_A=_condition(sa, rank_a),
        condition_C=_condition(sc, rank_c),
        condition_AN=condition_g,
        singular_values_A=sa,
        singular_values_C=sc,
        singular_values_AN=sg,
        thresholds={"A": at, "C": ct, "AN": gt},
        feasibility_tol=float(feasibility_tol),
        particular_solution=xp,
        null_basis=n_basis,
        null_coordinates=z,
        null_correction=correction,
        warnings=warnings,
        diagnostics=diagnostics,
    )
