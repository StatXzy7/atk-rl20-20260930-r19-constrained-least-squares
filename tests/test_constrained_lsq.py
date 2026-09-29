import unittest

import numpy as np

from constrained_lsq import constrained_lstsq


def nullspace(mat, tol=None):
    u, s, vh = np.linalg.svd(mat, full_matrices=True)
    if tol is None:
        tol = max(mat.shape) * np.finfo(float).eps * (s[0] if s.size else 1.0)
    rank = int(np.count_nonzero(s > tol))
    return vh[rank:, :].T, rank, s


def random_orthogonal(rng, n):
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q


def assert_first_order(test, result, A, b, C):
    n_basis, _, _ = nullspace(C)
    grad = A.T @ (A @ result.x - b)
    projected = n_basis.T @ grad if n_basis.size else np.zeros(0)
    test.assertLess(np.linalg.norm(projected), 1e-9)


def assert_local_minimum(test, result, A, b, C, d, rng, samples=80):
    base_obj = np.linalg.norm(A @ result.x - b) ** 2
    n_basis, _, _ = nullspace(C)
    for scale in (1e-7, 1e-3, 1.0, 10.0):
        for _ in range(samples):
            z = rng.standard_normal(n_basis.shape[1])
            y = result.x + scale * n_basis @ z
            np.testing.assert_allclose(C @ y, d, atol=1e-9)
            test.assertGreaterEqual(np.linalg.norm(A @ y - b) ** 2, base_obj - 1e-16)


def assert_global_min_norm(test, result, A, b, C, d, rng, samples=120):
    assert_local_minimum(test, result, A, b, C, d, rng, samples)
    base_obj = np.linalg.norm(A @ result.x - b) ** 2
    # Objective minimizers differ by ker(A); among those, feasible ones differ
    # by ker(A) intersect ker(C).  Compute that intersection independently with
    # SVD, instead of reusing the solver's returned null basis.
    ker_a, _, _ = nullspace(A)
    if ker_a.shape[1] == 0:
        return
    proj = C @ ker_a
    common, _, _ = nullspace(proj)
    for _ in range(samples):
        w = rng.standard_normal(common.shape[1]) if common.size else np.zeros(1)
        if not common.size:
            continue
        v = common @ w
        if np.linalg.norm(v) < 1e-12:
            continue
        y = ker_a @ v
        yy = result.x + y
        np.testing.assert_allclose(C @ yy, d, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(A @ yy - b) ** 2, base_obj, atol=1e-10)
        test.assertGreaterEqual(np.linalg.norm(yy), np.linalg.norm(result.x) - 1e-8)


class ConstrainedLSTSQTest(unittest.TestCase):
    def test_unconstrained_rank_deficient_minimum_norm(self):
        A = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        b = np.array([1.0, 0.0])
        r = constrained_lstsq(A, b)
        np.testing.assert_allclose(r.x, [0.5, 0.5, 0.0], atol=1e-12)
        np.testing.assert_allclose(A @ r.x, b, atol=1e-12)
        self.assertLess(r.projected_gradient_norm, 1e-12)
        self.assertLess(r.diagnostics["minimum_norm_projection_onto_common_ker_A_ker_C"], 1e-12)
        self.assertTrue(r.minimizer_multiplicity)

    def test_constrained_rational_case_with_nonzero_particular_and_correction(self):
        C = np.array([[1.0, 1.0, 1.0]])
        d = np.array([1.0])
        A = np.diag([1.0, 2.0, 0.0])
        b = np.array([0.0, 1.0, 1.0])
        r = constrained_lstsq(A, b, C, d)
        np.testing.assert_allclose(r.x, [0.0, 0.5, 0.5], atol=1e-12)
        np.testing.assert_allclose(C @ r.x, d, atol=1e-12)
        self.assertLess(r.projected_gradient_norm, 1e-12)
        self.assertLess(r.diagnostics["particular_null_orthogonality"], 1e-12)
        self.assertLess(r.diagnostics["minimum_norm_projection_onto_common_ker_A_ker_C"], 1e-12)
        self.assertTrue(r.objective_multiplicity)
        self.assertFalse(r.minimizer_multiplicity)

    def test_zero_dimensional_feasible_subspace(self):
        A = np.array([[1.0, 1.0]])
        b = np.array([7.0])
        C = np.eye(2)
        d = np.array([2.0, 3.0])
        r = constrained_lstsq(A, b, C, d)
        np.testing.assert_allclose(r.x, [2.0, 3.0])
        self.assertEqual(r.status, "unique_feasible_solution")
        self.assertEqual(r.feasible_affine_dimension, 0)
        self.assertTrue(np.isnan(r.projected_gradient_norm))

    def test_redundant_compatible_constraints(self):
        C = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
        d = np.array([2.0, 4.0, -1.0])
        A = np.array([[1.0, 1.0]])
        b = np.array([0.0])
        r = constrained_lstsq(A, b, C, d)
        np.testing.assert_allclose(r.x, [2.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(C @ r.x, d, atol=1e-12)
        self.assertEqual((r.rank_C, r.nullity_C), (2, 0))

    def test_incompatible_constraints_have_verifiable_left_null_evidence(self):
        C = np.array([[1.0, 0.0], [1.0, 0.0]])
        d = np.array([1.0, 2.0])
        A = np.eye(2)
        b = np.zeros(2)
        r = constrained_lstsq(A, b, C, d)
        self.assertEqual(r.status, "incompatible_constraints")
        self.assertIsNone(r.x)
        cert = r.infeasibility_certificate
        y = cert["left_null_direction"]
        self.assertAlmostEqual(abs(float(y @ d)), 1.0 / np.sqrt(2.0), places=12)
        self.assertLess(cert["yT_C_norm"], 1e-14)
        self.assertGreater(cert["normalized_feasibility_residual"], r.feasibility_tol)


    def test_dimension_and_tolerance_validation(self):
        with self.assertRaises(ValueError):
            constrained_lstsq(np.eye(2), [1.0])
        with self.assertRaises(ValueError):
            constrained_lstsq(np.eye(2), [1.0, 2.0], np.array([[1.0, 0.0]]), [1.0, 2.0])
        with self.assertRaises(ValueError):
            constrained_lstsq(np.eye(1), [0.0], rank_tol=-1.0)
        with self.assertRaises(ValueError):
            constrained_lstsq(np.eye(1), [0.0], feasibility_tol=-1.0)
        with self.assertRaises(ValueError):
            constrained_lstsq(np.eye(1), [0.0], uncertain_factor=0.0)

    def test_minimum_norm_objective_with_multiple_constrained_minimizers(self):
        # A x depends only on x1; constraint leaves a common ker(A)∩ker(C)
        # direction.  Exact minimizers are x1=1 and x2+x3=1; minimum norm is
        # (1, 1/2, 1/2).
        A = np.array([[1.0, 0.0, 0.0]])
        b = np.array([1.0])
        C = np.array([[0.0, 1.0, 1.0]])
        d = np.array([1.0])
        r = constrained_lstsq(A, b, C, d)
        np.testing.assert_allclose(r.x, [1.0, 0.5, 0.5], atol=1e-12)
        self.assertEqual((r.rank_A, r.rank_C, r.rank_AN), (1, 1, 1))
        self.assertTrue(r.minimizer_multiplicity)
        self.assertEqual(r.diagnostics["common_nullity"], 1)
        self.assertLess(r.diagnostics["minimum_norm_projection_onto_common_ker_A_ker_C"], 1e-12)

    def test_zero_matrices_and_zero_rows(self):
        r0 = constrained_lstsq(np.zeros((2, 2)), [1.0, 2.0])
        np.testing.assert_allclose(r0.x, [0.0, 0.0])
        self.assertEqual(r0.residual_norm, np.sqrt(5.0))
        self.assertEqual(r0.status, "unconstrained_objective_invariant_on_feasible_set")

        r1 = constrained_lstsq(np.eye(2), [1.0, 2.0], np.zeros((2, 2)), np.zeros(2))
        np.testing.assert_allclose(r1.x, [1.0, 2.0])
        self.assertEqual((r1.rank_C, r1.nullity_C), (0, 2))

        r2 = constrained_lstsq(np.eye(2), [1.0, 2.0], np.empty((0, 2)), np.empty(0))
        np.testing.assert_allclose(r2.x, [1.0, 2.0])

        r3 = constrained_lstsq(np.eye(2), [1.0, 2.0], np.zeros((1, 2)), [1.0])
        self.assertEqual(r3.status, "incompatible_constraints")

    def test_orthogonal_coordinate_change_preserves_minimum_norm(self):
        rng = np.random.default_rng(731)
        base_A = np.diag([1.0, 2.0, 0.0])
        A = base_A
        b = np.array([1.0, -1.0, 4.0])
        C = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]])
        d = np.array([2.0, 0.0])
        r0 = constrained_lstsq(A, b, C, d)
        q = random_orthogonal(rng, 3)
        # Orthogonal coordinate changes preserve the Euclidean norm.
        rq = constrained_lstsq(A @ q, b, C @ q, d)
        np.testing.assert_allclose(rq.x, q.T @ r0.x, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(rq.x), np.linalg.norm(r0.x), atol=1e-12)

    def test_independent_random_optimality_and_minimum_norm_checks(self):
        rng = np.random.default_rng(4421)
        cases = [
            (np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), np.array([1.0, 0.0]),
             np.array([[1.0, 0.0, -1.0]]), np.array([1.0])),
            (np.diag([1.0, 2.0, 0.0]), np.array([0.0, 1.0, 1.0]),
             np.array([[1.0, 1.0, 1.0]]), np.array([1.0])),
            (np.array([[1.0, 1.0, 0.0]]), np.array([3.0]),
             np.array([[1.0, 0.0, 0.0]]), np.array([1.0])),
            (np.array([[1.0, 1.0], [0.0, 0.0]]), np.array([1.0, 1.0]), None, None),
        ]
        for A, b, C, d in cases:
            r = constrained_lstsq(A, b, C, d)
            if C is None:
                C = np.empty((0, A.shape[1]))
                d = np.empty(0)
            assert_first_order(self, r, A, b, C)
            assert_global_min_norm(self, r, A, b, C, d, rng)

    def test_near_rank_deficient_threshold_sweep_reports_sensitivity(self):
        delta = 1e-8
        A = np.array([[1.0, 1.0], [delta, -delta]])
        b = np.array([1.0, 0.0])
        high_rank = constrained_lstsq(A, b, rank_tol=1e-12)
        low_rank = constrained_lstsq(A, b, rank_tol=2e-8)
        self.assertEqual((high_rank.rank_A, low_rank.rank_A), (2, 1))
        np.testing.assert_allclose(high_rank.x, [0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(low_rank.x, [0.5, 0.5], atol=1e-12)
        self.assertGreater(high_rank.condition_A, 9e7)
        self.assertEqual(high_rank.warnings, [])
        self.assertTrue(any("threshold-sensitive" in w for w in low_rank.warnings))

    def test_constraint_rank_threshold_changes_compatibility(self):
        delta = 1e-8
        C = np.array([[1.0, 0.0], [1.0 + delta, 0.0]])
        d = np.array([1.0, 1.0])
        rank_two = constrained_lstsq(np.eye(2), np.zeros(2), C, d,
                                     rank_tol=1e-10, feasibility_tol=1e-11)
        rank_one = constrained_lstsq(np.eye(2), np.zeros(2), C, d,
                                     rank_tol=1e-6, feasibility_tol=1e-6)
        # At numerical rank 2 the O(delta) mismatch exceeds a strict feasibility
        # tolerance.  Collapsing the nearly duplicate row plus a loose tolerance
        # gives the compatible limiting problem.
        self.assertEqual(rank_two.status, "incompatible_constraints")
        self.assertEqual(rank_one.status, "unique_constrained_minimizer")
        np.testing.assert_allclose(rank_one.x, [1.0, 0.0], atol=1e-8)
        self.assertGreater(
            rank_two.infeasibility_certificate["normalized_feasibility_residual"],
            rank_two.feasibility_tol,
        )
        self.assertTrue(any("not an exact algebraic proof" in w for w in rank_two.warnings))
        self.assertEqual(rank_one.warnings, [])

    def test_feasibility_threshold_uncertainty_is_not_exact_proof(self):
        C = np.array([[1.0], [1.0]])
        d = np.array([1.0, 1.0 + 5e-11])
        accepted = constrained_lstsq(np.eye(1), [0.0], C, d, feasibility_tol=1e-10)
        rejected = constrained_lstsq(np.eye(1), [0.0], C, d, feasibility_tol=1e-12)
        self.assertTrue(accepted.success)
        self.assertEqual(rejected.status, "incompatible_constraints")
        self.assertTrue(any("threshold-sensitive" in w for w in accepted.warnings))
        self.assertTrue(any("not an exact algebraic proof" in w for w in rejected.warnings))


if __name__ == "__main__":
    unittest.main()
