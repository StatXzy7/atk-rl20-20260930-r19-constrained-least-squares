# Rank-deficient equality-constrained least squares

## Problem

Given `A in R^(m x n)`, `b in R^m`, `C in R^(p x n)`, and `d in R^p`, solve

\[
\operatorname*{minimize}_{x\in\mathbb R^n}\ \|Ax-b\|_2
\quad\text{subject to}\quad Cx=d.
\]

`A` and `C` may both be rank deficient. If several `x` attain the same minimum
objective, the requested answer is the one with minimum Euclidean norm
\(\|x\|_2\).

The implementation uses SVD for both rank decisions and null spaces. It does
not form, solve, factor, or invert a KKT matrix, and it does not add an
arbitrary ridge term.

## Numerical-rank conventions

For a matrix `M` with singular values
\(\sigma_1\ge\cdots\ge\sigma_{\min(r,c)}\), a singular value is retained when

\[
\sigma_i>\tau_M,\qquad
\tau_M=\rho\,\sigma_1,
\]

where `rank_tol` is \(\rho\). The default is the LAPACK/NumPy-style

\[
\rho=\max(r,c)\,\varepsilon_{\rm machine}.
\]

The same relative `rank_tol` is used independently for `A`, `C`, and the
reduced objective `A N`; each returned `thresholds[...]` value is the actual
absolute threshold \(\rho\sigma_1\) for that matrix. Singular values, ranks,
effective condition numbers \(\sigma_1/\sigma_r\), and the first rejected
singular value are returned.

The default warning band is `uncertain_factor = 8`. A rank decision is flagged
when a retained/rejected singular value is within that factor of the threshold.
That warning means the answer should be rerun with nearby thresholds; it does
not mean the floating-point computation has proved exact rank.

Feasibility is tested from the SVD particular solution. The raw and
normalized residuals are both returned. With \(x_p=C^\dagger d\),

\[
\eta=\frac{\|d-Cx_p\|_2}
{\max(\|d\|_2,\|C\|_F\|x_p\|_2)}.
\]

Constraints are numerically compatible when \(\eta\le\) `feasibility_tol`
(default `1e-11`). A residual within a factor of eight of the tolerance is
reported as threshold-sensitive. These thresholds should be adjusted to the
measurement/model uncertainty of a concrete problem.

## Null-space construction

Let the truncated SVD of `C` be

\[
C \approx U_r \Sigma_r V_r^T,
\]

with \(\operatorname{rank}(C)=r\). Set

\[
N=V_0,
\]

where the columns of `V0` are the rejected right singular vectors. Thus
\(CN=0\), \(N^TN=I\), and

\[
\mathbb R^n=\operatorname{range}(C^T)\oplus\ker(C).
\]

The SVD particular solution is

\[
x_p=C^\dagger d=V_r\Sigma_r^{-1}U_r^Td\in\operatorname{range}(C^T).
\]

Every feasible point, when one exists, has the unique orthogonal decomposition

\[
x=x_p+Nz,\qquad x_p\perp Nz.
\]

Compatibility follows exactly from range/null-space orthogonality:

\[
d\in\operatorname{range}(C)
\iff U_0^Td=0,
\]

where the columns of \(U_0\) span \(\ker(C^T)\). When the numerical test says
incompatible, the returned certificate uses \(y=U_0\) with

\[
y^TC=0,\qquad y^Td\ne0.
\]

If \(Cx=d\) held, multiplying by \(y^T\) would force \(y^Td=0\), which is the
checkable contradiction. For duplicate inconsistent rows such as
\(x_1=1,\ x_1=2\), \(y\propto(1,-1)^T\) and
\(y^Td/\|y\|_2=1/\sqrt2\).

## Why both particular and correction matter

Substitute \(x=x_p+Nz\):

\[
\|Ax-b\|_2=\|ANz-(b-Ax_p)\|_2.
\]

Write \(G=AN\) and \(h=b-Ax_p\). Let the truncated SVD of `G` be

\[
G=P_r\Lambda_rQ_r^T.
\]

The set of reduced least-squares minimizers is

\[
\mathcal Z=\arg\min_z\|Gz-h\|_2
     =z_p+\ker(G),\qquad
z_p=G^\dagger h=Q_r\Lambda_r^{-1}P_r^Th.
\]

The minimum-norm representative of that affine minimizer set is \(z_p\),
because \(z_p\in\operatorname{range}(G^T)\) and hence
\(z_p\perp\ker(G)\). Choosing any arbitrary feasible particular solution, or
any arbitrary solution of the reduced normal equations, can introduce a
component in \(\ker(G)\); that component does not change the objective but does
change the norm.

Moreover, because \(x_p\perp\operatorname{range}(N)\),

\[
\|x_p+Nz\|_2^2=\|x_p\|_2^2+\|z\|_2^2.
\]

Since \(x_p\) is fixed for the chosen constraint SVD, minimizing the norm over
objective minimizers is exactly the minimum-norm reduced LS problem. Choosing
the minimum-norm \(x_p=C^\dagger d\) alone is not enough: a non-minimal
null-space correction could give a larger final norm. Choosing only a
minimum-norm correction is also not enough if \(x_p\) contains an arbitrary
\(\ker(C)\) component that cancels or adds to that correction. The two
orthogonal choices together give

\[
x_\star=x_p+Nz_p.
\]

This is the minimum Euclidean-norm constrained least-squares solution.

The feasible directions that preserve an attained objective minimum are

\[
\ker(A)\cap\ker(C).
\]

In reduced coordinates this is \(\ker(AN)=\ker(G)\). The final solution is
orthogonal to that common null space; the code reports the norm of its
projection there as an independent minimum-norm diagnostic. The whole of
\(\ker(A)\) is not the right certificate when constraints force an
otherwise-objective-null component.

## First-order optimality

For a positive-dimensional affine feasible set,

\[
\nabla f(x)=A^T(Ax-b)
\]

must be orthogonal to every feasible direction:

\[
N^T\nabla f(x)=0.
\]

This projected gradient norm is returned. A nonzero component in
\(\operatorname{range}(C^T)\) is not an optimality residual; it is balanced by
the equality multiplier. When \(\ker(C)=\{0\}\), the feasible set has one
point, so the projected-gradient check is vacuous and reported as `nan`.

## Special cases

- No constraints: `C` is treated as a `0 x n` matrix, `N=I`, and the reduced
  problem is the ordinary rank-deficient minimum-norm LS problem.
- Zero-dimensional feasible set: \(\operatorname{rank}(C)=n\); the unique
  feasible \(x_p\) is returned and no objective optimization direction exists.
- Redundant constraints: extra rows lie in the row space; compatibility uses
  the left singular null vectors and the redundant rows impose no new
  directions.
- `A=0`: the objective is constant over the feasible set, so the minimum-norm
  feasible point is \(C^\dagger d\). With no constraints this is zero.
- `C=0`, `d=0`: all variables are feasible.
- `C=0`, nonzero `d`: the normalized left-null direction is a unit direction
  in constraint space, giving \(y^TC=0\) and \(y^Td\ne0\).
- Overdetermined constraints at full numerical rank: if the accepted `C` has
  full row rank, exact arithmetic guarantees compatibility; an observed
  pseudoinverse residual is reported as a conditioning warning, not as a
  left-null-space contradiction.

## Column scaling and norm meaning

For a nonsingular diagonal scaling \(x=Dy\),

\[
\|Ax-b\|_2=\|ADy-b\|_2,\qquad \|x\|_2=\|Dy\|_2.
\]

If the scaled problem simply returns the minimum-norm `y`, it minimizes
\(\|y\|_2\), generally not the requested \(\|Dy\|_2\). Consequently a
column-scaled solver can choose an exact but non-Euclidean-minimum original
solution. Scaling may still be useful for conditioning, but solution selection
must use the matching weighted metric, and residual/norm diagnostics must be
transformed back consistently. This package defaults to the unscaled Euclidean
metric and makes that convention explicit.
