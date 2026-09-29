#!/usr/bin/env python3
"""Generate a reproducible numerical diagnostics report."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from constrained_lsq import constrained_lstsq

REPORT = ROOT / "docs" / "diagnostic_report.md"


def fmt(value, digits=6):
    if value is None:
        return "—"
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=4, max_line_width=88)
    if isinstance(value, float):
        if np.isnan(value):
            return "vacuous"
        if np.isinf(value):
            return "∞"
        return f"{value:.{digits}g}"
    return str(value)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(fmt(v) for v in row) + " |" for row in rows)
    return "\n".join(out)


def analytic_section():
    A = np.diag([1.0, 2.0, 0.0])
    b = np.array([0.0, 1.0, 1.0])
    C = np.array([[1.0, 1.0, 1.0]])
    d = np.array([1.0])
    r = constrained_lstsq(A, b, C, d)
    Cr = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    dr = np.array([2.0, 4.0, -1.0])
    rr = constrained_lstsq(np.array([[1.0, 1.0]]), [0.0], Cr, dr)
    Ci = np.array([[1.0, 0.0], [1.0, 0.0]])
    di = np.array([1.0, 2.0])
    ri = constrained_lstsq(np.eye(2), np.zeros(2), Ci, di)
    cert = ri.infeasibility_certificate
    row = [[r.x, r.diagnostics["particular_solution_norm"],
            r.diagnostics["null_correction_norm"], r.residual_norm,
            r.constraint_residual_norm, r.projected_gradient_norm,
            r.diagnostics["minimum_norm_projection_onto_common_ker_A_ker_C"],
            f"{r.rank_A}/{r.rank_C}/{r.rank_AN}"]]
    return f"""## Analytic and special cases

Integer/rational example: `A=diag(1,2,0)`, `b=(0,1,1)^T`, constraint
`x1+x2+x3=1`. The analytic minimizer is `x=(0,1/2,1/2)^T`. `A` is rank
deficient, but the equality removes the common minimizer-null direction, so the
constrained minimizer is unique and the minimum-norm certificate uses
`ker(A) ∩ ker(C)`.

{table(["x", "||xp||", "||Nz||", "LS residual", "constraint residual",
        "projected gradient", "common-null projection", "ranks A/C/AN"], row)}

Redundant compatible constraints `x1=2, 2x1=4, x2=-1` produce
`x={fmt(rr.x)}`; ranks are `{rr.rank_A}/{rr.rank_C}/{rr.rank_AN}` and raw
constraint residual norm is `{rr.constraint_residual_norm:.3g}`.

Incompatible duplicate constraints `x1=1`, `x1=2` return no x. A normalized
left-null direction is `y={fmt(cert["left_null_direction"])}`, with
`||y^T C||={cert["yT_C_norm"]:.3g}` and
`|y^T d|={cert["absolute_evidence"]:.6f}`.
"""


def threshold_section():
    delta = 1e-8
    A = np.array([[1.0, 1.0], [delta, -delta]])
    b = np.array([0.0, 1.0])
    rows = []
    for rt in [1e-12, 2e-8, 1e-6]:
        r = constrained_lstsq(A, b, rank_tol=rt)
        rows.append([rt, r.rank_A, r.condition_A, r.x,
                     r.diagnostics["x_norm"], r.residual_norm,
                     "yes" if r.warnings else "no"])
    crows = []
    for rt, ft in [(1e-12, 1e-11), (2e-8, 1e-11), (1e-6, 1e-6)]:
        cr = constrained_lstsq(np.eye(2), np.zeros(2), A, b,
                               rank_tol=rt, feasibility_tol=ft)
        crows.append([rt, ft, cr.rank_C, cr.status,
                      "—" if cr.x is None else cr.x,
                      cr.constraint_residual_norm,
                      "yes" if cr.warnings else "no"])
    return f"""## Singular-value threshold sensitivity

`M=[[1,1],[δ,-δ]]`, `δ=1e-8`, has singular values `sqrt(2)` and
`sqrt(2)δ`. As an objective matrix with `b=(0,1)^T`, the right-hand side has a
near-null component.

{table(["rank_tol", "rank(A)", "κ(A)", "x", "||x||", "||Ax-b||", "warning"], rows)}

Full numerical rank nearly interpolates the tiny second equation and gives
norm about `7.07e7`. Treating the second singular value as zero makes that
component unobservable and returns the rank-one minimum-norm answer `0` with
residual `1`. The crossing is marked as numerically uncertain.

Using the same matrix and vector as constraints:

{table(["rank_tol", "feas_tol", "rank(C)", "status", "x",
        "constraint residual", "warning"], crows)}

At rank two this is a very ill-conditioned but compatible coordinate
transform. At rank one, `d` is outside the accepted range and strict
feasibility produces a left-null-space certificate.
"""


def orthogonal_section():
    rng = np.random.default_rng(20260930)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    A = np.diag([1.0, 2.0, 0.0])
    b = np.array([0.0, 1.0, 1.0])
    C = np.array([[1.0, 1.0, 1.0]])
    d = np.array([1.0])
    r0 = constrained_lstsq(A, b, C, d)
    rq = constrained_lstsq(A @ q, b, C @ q, d)
    rows = [["original", r0.x, r0.diagnostics["x_norm"], r0.residual_norm,
             r0.constraint_residual_norm, r0.projected_gradient_norm],
            ["after Q", rq.x, rq.diagnostics["x_norm"], rq.residual_norm,
             rq.constraint_residual_norm, rq.projected_gradient_norm]]
    diff = np.linalg.norm(rq.x - q.T @ r0.x)
    return f"""## Orthogonal coordinate transformation

For orthogonal `Q`, variables transform as `y=Q^T x`, with objective `AQ` and
constraints `CQ`. This preserves the Euclidean norm. Back-transform error
`||y-Q^T x||` is `{diff:.3g}`.

{table(["case", "solution", "norm", "LS residual", "constraint residual",
        "projected gradient"], rows)}
"""


def scaling_section():
    A = np.array([[1.0, 1.0]])
    b = np.array([2.0])
    r = constrained_lstsq(A, b)
    D = np.diag([1.0, 100.0])
    wrong = D @ constrained_lstsq(A @ D, b).x
    rows = [["original Euclidean", r.x, r.diagnostics["x_norm"], r.residual_norm],
            ["minimum ||y||, transformed back", wrong, np.linalg.norm(wrong),
             np.linalg.norm(A @ wrong - b)]]
    return f"""## Column scaling and minimum-norm meaning

With `x=Dy`, `Ax=(AD)y` and `||x||=||Dy||`. Selecting the minimum-norm `y`
therefore minimizes a different metric. Here `D=diag(1,100)`:

{table(["selection rule", "x", "||x||", "||Ax-b||"], rows)}

Both rows fit the objective, but the scaled selection is not the minimum
Euclidean-norm original solution. Scaling must be paired with the matching
weighted norm and consistent back-transformation; the solver defaults to the
unscaled Euclidean convention.
"""


def boundary_section():
    cases = [
        ("zero A", constrained_lstsq(np.zeros((2, 2)), [1.0, 2.0])),
        ("zero C, zero d", constrained_lstsq(np.eye(2), [1.0, 2.0], np.zeros((2, 2)), np.zeros(2))),
        ("zero C, nonzero d", constrained_lstsq(np.eye(2), [1.0, 2.0], np.zeros((1, 2)), [1.0])),
        ("zero feasible nullity", constrained_lstsq(np.array([[1.0, 1.0]]), [7.0], np.eye(2), [2.0, 3.0])),
    ]
    rows = [[name, r.status, "—" if r.x is None else r.x, r.rank_A, r.rank_C,
             r.feasible_affine_dimension, r.constraint_residual_norm,
             r.residual_norm] for name, r in cases]
    return "## Boundary cases\n\n" + table(
        ["case", "status", "x", "rank A", "rank C", "feasible dim",
         "constraint residual", "LS residual"], rows) + "\n"


def main():
    text = "\n".join([
        "# Constrained least-squares diagnostic report\n",
        "Generated by `examples/generate_report.py`. Rank and feasibility "
        "statements are numerical at explicit thresholds, not exact algebraic "
        "proofs.\n",
        analytic_section(), threshold_section(), orthogonal_section(),
        scaling_section(), boundary_section(),
        "## Diagnostic fields\n\nThe result exposes constraint and LS residuals, "
        "projected gradient, ranks `rank_A`, `rank_C`, `rank_AN`, condition "
        "numbers, singular values, absolute thresholds, particular/null "
        "components, common-null projection, normalized residual, warnings, and "
        "left-null infeasibility evidence.\n",
    ])
    REPORT.write_text(text, encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
