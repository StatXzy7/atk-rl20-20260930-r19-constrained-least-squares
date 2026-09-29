import unittest
from fractions import Fraction

import numpy as np

from constrained_lsq import constrained_lsq


def rational_matrix(values):
    return [[Fraction(int(v)) for v in row] for row in np.asarray(values)]


def rational_vector(values):
    return [Fraction(int(v)) for v in np.asarray(values)]


def rref(aug):
    """Return RREF, pivot columns, and free columns for a rational system."""
    mat = [row[:] for row in aug]
    rows = len(mat)
    cols = len(mat[0]) - 1
    pivots = []
    row = 0

    for col in range(cols):
        pivot = next((i for i in range(row, rows) if mat[i][col] != 0), None)
        if pivot is None:
            continue
        mat[row], mat[pivot] = mat[pivot], mat[row]
        pivot_value = mat[row][col]
        mat[row] = [value / pivot_value for value in mat[row]]
        for i in range(rows):
            if i != row and mat[i][col] != 0:
                factor = mat[i][col]
                mat[i] = [
                    a - factor * b
                    for a, b in zip(mat[i], mat[row])
                ]
        pivots.append(col)
        row += 1
        if row == rows:
            break

    for i in range(rows):
        if all(mat[i][j] == 0 for j in range(cols)) and mat[i][cols] != 0:
            raise ValueError("inconsistent rational system")

    free = [col for col in range(cols) if col not in pivots]
    return mat, pivots, free


def rref_particular_null(aug):
    """RREF-derived particular solution and null-space basis."""
    mat, pivots, free = rref(aug)
    rows = len(mat)
    cols = len(mat[0]) - 1
    for i in range(rows):
        if all(mat[i][j] == 0 for j in range(cols)) and mat[i][cols] != 0:
            return None, None

    particular = [Fraction(0) for _ in range(cols)]
    for i, col in enumerate(pivots):
        particular[col] = mat[i][cols]

    nullspace = []
    for free_col in free:
        vec = [Fraction(0) for _ in range(cols)]
        vec[free_col] = Fraction(1)
        for i, pivot_col in enumerate(pivots):
            vec[pivot_col] = -mat[i][free_col]
        nullspace.append(vec)
    return particular, nullspace


def solve_rational_square(matrix, rhs):
    n = len(matrix)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next(i for i in range(col, n) if aug[i][col] != 0)
        aug[col], aug[pivot] = aug[pivot], aug[col]
        value = aug[col][col]
        aug[col] = [item / value for item in aug[col]]
        for i in range(n):
            if i != col and aug[i][col] != 0:
                factor = aug[i][col]
                aug[i] = [a - factor * b for a, b in zip(aug[i], aug[col])]
    return [aug[i][n] for i in range(n)]


def minimum_norm_consistent_solution(aug):
    """Exact minimum-norm solution of a consistent rational full-row system."""
    mat, pivots, _ = rref(aug)
    cols = len(mat[0]) - 1
    rank = len(pivots)
    R = [row[:cols] for row in mat[:rank]]
    rhs = [mat[i][cols] for i in range(rank)]
    RRT = [
        [sum(a * b for a, b in zip(R[i], R[j])) for j in range(rank)]
        for i in range(rank)
    ]
    y = solve_rational_square(RRT, rhs)
    return [sum(R[k][j] * y[k] for k in range(rank)) for j in range(cols)]


def exact_constrained_lsq(A, b, C=None, d=None):
    """Small exact rational minimum-norm constrained LS reference."""
    A = rational_matrix(A)
    b = rational_vector(b)
    n = len(A[0])
    if C is None:
        C = [[Fraction(0) for _ in range(n)] for _ in range(0)]
        d = rational_vector([])
    else:
        C = rational_matrix(C)
        d = rational_vector(d)

    c_aug = [row + [rhs] for row, rhs in zip(C, d)]
    if c_aug:
        xp, cn = rref_particular_null(c_aug)
    else:
        xp, cn = [Fraction(0) for _ in range(n)], [
            [Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)
        ]
    if xp is None:
        return None

    def mat_vec(M, v):
        return [sum(a * b for a, b in zip(row, v)) for row in M]

    def add(u, v, sign=1):
        return [a + sign * b for a, b in zip(u, v)]

    null_columns = list(zip(*cn)) if cn else []
    B = []
    if cn:
        for row in A:
            B.append([sum(row[j] * basis[j] for j in range(n)) for basis in cn])
    else:
        B = [[Fraction(0) for _ in range(0)] for _ in A]

    axp = mat_vec(A, xp)
    g = add(b, axp, -1)
    Btg = [
        sum(B[i][k] * g[i] for i in range(len(B)))
        for k in range(len(cn))
    ]
    QtQ = [
        [
            sum(B[i][k] * B[i][l] for i in range(len(B)))
            for l in range(len(cn))
        ]
        for k in range(len(cn))
    ]
    q_aug = [row + [rhs] for row, rhs in zip(QtQ, Btg)]
    if q_aug:
        zmin, qn = rref_particular_null(q_aug)
    else:
        zmin, qn = [], []
    if zmin is None:
        raise AssertionError("normal equations were unexpectedly incompatible")
    if qn:
        zmin = minimum_norm_consistent_solution(q_aug)

    correction = [Fraction(0) for _ in range(n)]
    for k, basis in enumerate(cn):
        for j in range(n):
            correction[j] += basis[j] * zmin[k]
    return add(xp, correction)


def fraction_array(values):
    return np.array([float(v) for v in values])


class ConstrainedLSTests(unittest.TestCase):
    def assertClose(self, actual, expected, tol=1e-10, atol=None, rtol=None):
        np.testing.assert_allclose(
            actual,
            expected,
            atol=tol if atol is None else atol,
            rtol=tol if rtol is None else rtol,
        )

    def assertExactCase(self, A, b, C=None, d=None, expected=None):
        result = constrained_lsq(A, b, C, d)
        self.assertEqual(result.status, "ok")
        x_exact = exact_constrained_lsq(A, b, C, d)
        self.assertIsNotNone(x_exact)
        self.assertClose(result.x, fraction_array(x_exact))
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        if C is None:
            self.assertEqual(result.constraint_residual_norm, 0.0)
        else:
            C = np.asarray(C, dtype=float)
            d = np.asarray(d, dtype=float)
            self.assertLessEqual(
                result.constraint_residual_norm,
                result.achieved_constraint_tol,
            )
            self.assertClose(C @ result.x, d, tol=result.achieved_constraint_tol)
        residual = b - A @ result.x
        B = None
        if C is not None:
            C_arr = np.asarray(C, dtype=float)
            _, c_s, c_Vt = np.linalg.svd(C_arr, full_matrices=True)
            c_rank = int(np.count_nonzero(c_s > 0))
            null_basis = c_Vt[c_rank:, :].T
            B = A @ null_basis
        else:
            B = A
        if B.size:
            self.assertLessEqual(
                result.active_fit_norm,
                max(1.0, result.active_fit_tol) * 100,
            )
            self.assertClose(B.T @ residual, 0, atol=1e-9)
        if expected is not None:
            self.assertClose(result.x, expected)
        return result

    def test_unconstrained_rank_deficient_and_zero_rows(self):
        A = [[1, 0, 0], [0, 0, 0]]
        b = [1, 2]
        result = self.assertExactCase(A, b, expected=[1, 0, 0])
        self.assertEqual(result.C_model.rank, 0)
        self.assertEqual(result.reduced_A_model.rank, 1)
        self.assertAlmostEqual(result.least_squares_residual_norm, 2.0)

    def test_redundant_constraints_unique_objective(self):
        A = np.eye(2)
        b = [0, 0]
        C = [[1, 1], [2, 2]]
        d = [1, 2]
        result = self.assertExactCase(A, b, C, d, [0.5, 0.5])
        self.assertEqual(result.C_model.rank, 1)

    def test_zero_a_on_nullspace_selects_minimum_norm_feasible(self):
        A = np.zeros((2, 2), dtype=int)
        b = [3, 4]
        C = [[1, 0]]
        d = [1]
        result = self.assertExactCase(A, b, C, d, [1, 0])
        self.assertEqual(result.reduced_A_model.rank, 0)
        self.assertAlmostEqual(result.objective, 5.0)

    def test_zero_constraint_rows_and_zero_a(self):
        result = self.assertExactCase(
            np.zeros((2, 3), dtype=int),
            [3, 4, 0][:2],
            np.zeros((2, 3), dtype=int),
            [0, 0],
            [0, 0, 0],
        )
        self.assertAlmostEqual(result.objective, 5.0)
        self.assertEqual(result.C_model.rank, 0)

    def test_zero_dimensional_feasible_subspace(self):
        A = np.eye(2)
        b = [0.2, -0.7]
        C = [[1, 0], [0, 1], [1, 1]]
        d = [1, 2, 3]
        result = self.assertExactCase(A, b, C, d, [1, 2])
        self.assertEqual(result.C_model.shape[1] - result.C_model.rank, 0)
        self.assertTrue(any("dimension zero" in w for w in result.warnings))

    def test_multiple_objective_minimizers_get_minimum_norm_correction(self):
        A = [[1, 0, 0], [0, 0, 0]]
        b = [1, 2]
        C = [[1, 0, 0]]
        d = [1]
        self.assertExactCase(A, b, C, d, [1, 0, 0])

        A = [[0, 1, 1]]
        b = [1]
        C = [[1, 0, 0]]
        d = [1]
        result = self.assertExactCase(A, b, C, d, [1, 0.5, 0.5])
        self.assertAlmostEqual(result.particular_norm, 1.0)
        self.assertAlmostEqual(result.nullspace_correction_norm, np.sqrt(0.5))
        self.assertAlmostEqual(result.particular_correction_inner_product, 0.0)


    def test_independent_nullspace_perturbation_check(self):
        A = [[0, 1, 0]]
        b = [1]
        C = [[1, 0, 0]]
        d = [1]
        result = constrained_lsq(A, b, C, d)
        A_arr = np.asarray(A, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        self.assertClose(result.x, [1, 1, 0])

        _, _, C_Vt = np.linalg.svd(np.asarray(C, dtype=float), full_matrices=True)
        feasible_basis = C_Vt[1:, :].T
        reduced = A_arr @ feasible_basis
        _, rs, BVt = np.linalg.svd(reduced, full_matrices=True)
        reduced_null = BVt[int(np.count_nonzero(rs > 0)):, :].T
        self.assertEqual(reduced_null.shape, (2, 1))

        base_residual = np.linalg.norm(A_arr @ result.x - b_arr)
        for scale in [-100.0, -1e-4, 1e-4, 100.0]:
            x_perturbed = result.x + feasible_basis @ reduced_null @ [scale]
            self.assertAlmostEqual(
                np.linalg.norm(A_arr @ x_perturbed - b_arr), base_residual
            )
            self.assertGreater(
                np.linalg.norm(x_perturbed), np.linalg.norm(result.x)
            )

    def test_incompatible_constraint_has_verifiable_left_nullspace_witness(self):
        C = np.array([[1, 1], [2, 2]], dtype=float)
        d = np.array([1, 3], dtype=float)
        result = constrained_lsq([[1, 0], [0, 1]], [0, 0], C, d)
        self.assertFalse(result.success)
        self.assertEqual(result.status, "incompatible_constraints")
        w = result.incompatibility_direction
        self.assertClose(w @ C, 0, tol=1e-12)
        self.assertAlmostEqual(abs(w @ d), result.constraint_model_residual_norm)
        self.assertAlmostEqual(abs(w @ d), 1 / np.sqrt(5))
        self.assertGreater(result.constraint_model_residual_norm, result.constraint_tol)

    def test_zero_variables_compatible_and_incompatible(self):
        A = np.zeros((0, 0))
        b = np.zeros(0)
        C = np.zeros((1, 0))
        result = constrained_lsq(A, b, C, [0])
        self.assertTrue(result.success)
        self.assertEqual(result.x.shape, (0,))
        bad = constrained_lsq(A, b, C, [1])
        self.assertEqual(bad.status, "incompatible_constraints")
        self.assertClose(bad.incompatibility_direction, [1.0])

    def test_orthogonal_coordinate_transformation(self):
        rng = np.random.default_rng(20260930)
        A0 = rng.normal(size=(6, 4))
        C0 = rng.normal(size=(2, 4))
        b = rng.normal(size=6)
        d = rng.normal(size=2)
        Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))

        base = constrained_lsq(A0, b, C0, d)
        rotated = constrained_lsq(A0 @ Q, b, C0 @ Q, d)
        self.assertEqual(base.status, "ok")
        self.assertEqual(rotated.status, "ok")
        self.assertClose(rotated.x, Q.T @ base.x, tol=1e-10)
        self.assertAlmostEqual(np.linalg.norm(rotated.x), np.linalg.norm(base.x))
        self.assertClose((C0 @ Q) @ rotated.x, d, tol=1e-10)
        self.assertClose((A0 @ Q) @ rotated.x, A0 @ base.x, tol=1e-10)

    def test_column_scaling_changes_minimum_norm_meaning(self):
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        raw = constrained_lsq(A, b)
        scale = np.diag([2.0, 1.0])
        transformed = constrained_lsq(A @ scale, b)
        x_from_transformed = scale @ transformed.x
        self.assertClose(raw.x, [0.5, 0.5])
        self.assertClose(x_from_transformed, [0.8, 0.2])
        self.assertAlmostEqual(np.linalg.norm(A @ raw.x - b), 0)
        self.assertAlmostEqual(
            np.linalg.norm(A @ x_from_transformed - b), 0
        )
        self.assertLess(np.linalg.norm(raw.x), np.linalg.norm(x_from_transformed))

    def test_repeated_constraints_match_single_constraint(self):
        A = [[0, 1, 1]]
        b = [1]
        one = constrained_lsq(A, b, [[1, 0, 0]], [1])
        repeated = constrained_lsq(
            A,
            b,
            [[1, 0, 0], [1, 0, 0], [-1, 0, 0]],
            [1, 1, -1],
        )
        self.assertEqual(repeated.C_model.rank, 1)
        self.assertClose(repeated.x, one.x)

    def test_reduced_a_threshold_sweep_shows_solution_sensitivity(self):
        epsilon = 1e-8
        A = [[1, 0, 0], [0, 1, 0], [0, 0, epsilon]]
        b = [1, 1, 1]
        C = [[1, 0, 0]]
        d = [1]
        low_rank = constrained_lsq(A, b, C, d, rtol=1e-6)
        high_rank = constrained_lsq(A, b, C, d, rtol=1e-12)
        self.assertEqual(low_rank.reduced_A_model.rank, 1)
        self.assertEqual(high_rank.reduced_A_model.rank, 2)
        self.assertClose(low_rank.x, [1, 1, 0])
        self.assertClose(high_rank.x, [1, 1, 1 / epsilon])
        self.assertAlmostEqual(low_rank.objective, 1.0)
        self.assertAlmostEqual(low_rank.active_fit_norm, 0.0)
        self.assertLess(low_rank.active_fit_norm, low_rank.projected_gradient_norm)
        self.assertLess(high_rank.objective, 1e-8)
        self.assertGreater(
            np.linalg.norm(high_rank.x), 1000 * np.linalg.norm(low_rank.x)
        )
        ambiguous = constrained_lsq(A, b, C, d, a_rtol=1.0)
        self.assertEqual(ambiguous.status, "rank_ambiguous")

    def test_c_threshold_sweep_shows_feasibility_model_change(self):
        epsilon = 1e-8
        A = np.eye(2)
        b = [0, 0]
        C = [[1, 0], [0, epsilon]]
        d = [1, 1]
        high_rank = constrained_lsq(
            A, b, C, d, c_rtol=1e-12, constraint_tol=1e-10
        )
        low_rank = constrained_lsq(
            A, b, C, d, c_rtol=1e-6, constraint_tol=1e-10
        )
        near = constrained_lsq(
            A, b, C, d, c_rtol=5e-9, constraint_tol=1e-10
        )
        self.assertEqual(high_rank.C_model.rank, 2)
        self.assertEqual(low_rank.C_model.rank, 1)
        self.assertEqual(near.status, "rank_ambiguous")
        self.assertClose(high_rank.x, [1, 1 / epsilon])
        self.assertFalse(low_rank.success)
        self.assertEqual(low_rank.status, "incompatible_constraints")
        self.assertAlmostEqual(low_rank.constraint_model_residual_norm, 1.0)


if __name__ == "__main__":
    unittest.main()
