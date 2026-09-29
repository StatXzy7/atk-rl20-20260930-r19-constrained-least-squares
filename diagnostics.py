"""Generate Markdown diagnostics for constrained least-squares examples."""

from __future__ import annotations

from fractions import Fraction
from io import StringIO

import numpy as np

from constrained_lsq import constrained_lsq
from test_constrained_lsq import exact_constrained_lsq


def fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if np.isinf(value):
            return "∞"
        if value == 0:
            return "0"
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.{digits}e}"
        return f"{value:.{digits}f}"
    return str(value)


def vector(v, digits=6):
    if v is None:
        return "—"
    return "(" + ", ".join(f"{x:.{digits}g}" for x in v) + ")"


def row(label, result, extra=""):
    return (
        f"| {label} | {result.C_model.rank} | {result.reduced_A_model.rank} | "
        f"{fmt(result.C_model.condition_number_effective)} | "
        f"{fmt(result.reduced_A_model.condition_number_effective)} | "
        f"{fmt(result.constraint_model_residual_norm)} | "
        f"{fmt(result.constraint_residual_norm)} | "
        f"{fmt(result.least_squares_residual_norm)} | "
        f"{fmt(result.projected_gradient_norm)} | "
        f"{fmt(result.active_fit_norm)} | "
        f"{fmt(result.particular_norm)} | "
        f"{fmt(result.nullspace_correction_norm)} | "
        f"{fmt(result.x_norm)} | {extra} |\n"
    )


def singular_line(prefix, model):
    return (
        f"{prefix}: shape={model.shape}, rank={model.rank}, "
        f"σ={np.array2string(model.s, precision=3, separator=', ')}, "
        f"cutoff={fmt(model.cutoff, 3)}, "
        f"κ_eff={fmt(model.condition_number_effective, 3)}, "
        f"borderline={model.borderline}"
    )


def rational_check(A, b, C, d, name):
    rational = exact_constrained_lsq(A, b, C, d)
    result = constrained_lsq(A, b, C, d)
    expected = np.array([float(v) for v in rational])
    diff = float(np.linalg.norm(result.x - expected))
    return result, diff


def main() -> str:
    out = StringIO()
    out.write("# 约束最小二乘诊断报告\n\n")
    out.write(
        "所有结果由 `constrained_lsq.constrained_lsq` 计算；精确参考值由 "
        "`test_constrained_lsq.py` 中独立的有理数 RREF 实现给出。浮点输出只在"
        "声明的数值秩模型内解释，不作为精确代数证明。\n\n"
    )

    examples = []
    A = [[1, 0, 0], [0, 0, 0]]
    b = [1, 2]
    examples.append(("无约束：A 秩亏", (A, b, None, None), [1, 0, 0]))

    A = np.eye(2)
    b = [0, 0]
    examples.append(
        ("重复/冗余约束", (A, b, [[1, 1], [2, 2]], [1, 2]), [0.5, 0.5])
    )

    A = [[0, 1, 1]]
    b = [1]
    examples.append(
        ("多约束最小二乘解", (A, b, [[1, 0, 0]], [1]), [1, 0.5, 0.5])
    )

    examples.append(
        ("零 A + 零约束行", (np.zeros((2, 3)), [3, 4], np.zeros((2, 3)), [0, 0]), [0, 0, 0])
    )

    A = np.zeros((2, 2))
    examples.append(("可行子空间零维", (A, [3, 4], [[1, 0], [0, 1]], [1, 2]), [1, 2]))

    out.write("## 1. 小型整数/有理数例子\n\n")
    out.write(
        "| 例子 | rank(C) | rank(AN) | κ_eff(C) | κ_eff(AN) | 模型约束尾残差 | 实际 Cx-d | LS 残差 | "
        "完整 Bᵀr | 保留向拟合残差 | ||x_p||₂ | ||Nȳ||₂ | ||x||₂ | 备注 |\n"
    )
    out.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for name, args, expected in examples:
        result, diff = rational_check(*args, name)
        out.write(
            row(
                name,
                result,
                f"x={vector(result.x)}, 有理差={fmt(diff, 2)}",
            )
        )
    out.write(
        "\n独立核验包括 `Cx=d`、完整 `Bᵀr`、保留秩方向一阶残差、目标值以及精确有理 `x`。"
        "阈值截断时，完整投影梯度可能包含被丢弃小奇异方向的分量。\n\n"
    )

    out.write("## 2. 不相容约束的左零空间证据\n\n")
    C = np.array([[1.0, 1.0], [2.0, 2.0]])
    d = np.array([1.0, 3.0])
    bad = constrained_lsq(np.eye(2), np.zeros(2), C, d)
    w = bad.incompatibility_direction
    out.write(
        "对 `C=[[1,1],[2,2]]`, `d=[1,3]`，数值左零空间方向为 "
        f"`w={vector(w, 12)}`。可直接核验：\n\n"
    )
    out.write(
        f"- `||wᵀC||₂ = {np.linalg.norm(w @ C):.3e}`\n"
        f"- `|wᵀd| = {abs(float(w @ d)):.12f}`\n"
        f"- 模型最小约束尾残差 `||U₀ᵀd||₂ = "
        f"{bad.constraint_model_residual_norm:.12f}`\n"
        f"- 可行性容差 = {fmt(bad.constraint_tol, 3)}，因此状态为 "
        "`incompatible_constraints`。\n\n"
    )

    out.write("## 3. 约化矩阵 A N 的阈值扫描\n\n")
    eps = 1e-8
    A = np.diag([1.0, 1.0, eps])
    b = np.array([1.0, 1.0, 1.0])
    C = np.array([[1.0, 0.0, 0.0]])
    d = np.array([1.0])
    out.write(
        "约束固定 `x1=1`，零空间上的矩阵奇异值为 `{1, 1e-8}`。"
        "扫描 `rtol` 等价于扫描相对数值秩阈值。\n\n"
    )
    out.write("| rtol | rank(AN) | x | LS 残差 | ||x||₂ | 状态 |\n")
    out.write("|---:|---:|---|---:|---:|---|\n")
    for tol in [1e-12, 5e-9, 1e-6]:
        r = constrained_lsq(A, b, C, d, rtol=tol)
        out.write(
            f"| {tol:.0e} | {r.reduced_A_model.rank} | {vector(r.x, 3)} | "
            f"{fmt(r.least_squares_residual_norm, 3)} | {fmt(r.x_norm, 3)} | "
            f"{r.status} |\n"
        )
    r_amb = constrained_lsq(A, b, C, d, a_rtol=1.0)
    out.write(
        f"| a_rtol=1 | {r_amb.reduced_A_model.rank} | {vector(r_amb.x, 3)} | "
        f"{fmt(r_amb.least_squares_residual_norm, 3)} | {fmt(r_amb.x_norm, 3)} | "
        f"{r_amb.status} |\n\n"
    )
    out.write(singular_line("基准阈值", r_amb.reduced_A_model) + "\n\n")

    out.write("## 4. C 的阈值扫描\n\n")
    C = np.diag([1.0, eps])
    d = np.array([1.0, 1.0])
    out.write(
        "`C=diag(1,1e-8)`、`d=(1,1)`。保留第二奇异值时，约束给出 "
        "`x2=1e8`；把它判为零奇异值时，`d` 在对应左零空间上有大小为 1 的分量。\n\n"
    )
    out.write("| c_rtol | rank(C) | 状态 | 尾残差 | x |\n")
    out.write("|---:|---:|---|---:|---|\n")
    for tol in [1e-12, 5e-9, 1e-6]:
        r = constrained_lsq(
            np.eye(2), np.zeros(2), C, d, c_rtol=tol, constraint_tol=1e-10
        )
        out.write(
            f"| {tol:.0e} | {r.C_model.rank} | {r.status} | "
            f"{fmt(r.constraint_model_residual_norm, 3)} | {vector(r.x, 3)} |\n"
        )
    out.write(
        "\n当奇异值与阈值距离足够远，证据可用于强数值判断；若进入声明的不确定带，"
        "返回 `rank_ambiguous` 或 `incompatible_rank_ambiguous`。\n\n"
    )

    out.write("## 5. 正交变换与列缩放\n\n")
    rng = np.random.default_rng(20260930)
    A0 = rng.normal(size=(6, 4))
    C0 = rng.normal(size=(2, 4))
    bv = rng.normal(size=6)
    dv = rng.normal(size=2)
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    base = constrained_lsq(A0, bv, C0, dv)
    rotated = constrained_lsq(A0 @ Q, bv, C0 @ Q, dv)
    out.write(
        f"随机满秩例子中，正交坐标变换后的解误差 "
        f"`||Qᵀx-x'||₂={np.linalg.norm(Q.T @ base.x - rotated.x):.3e}`，"
        f"范数差={abs(np.linalg.norm(base.x)-np.linalg.norm(rotated.x)):.3e}。\n\n"
    )
    scaler = np.diag([2.0, 1.0])
    raw = constrained_lsq(np.array([[1.0, 1.0]]), np.array([1.0]))
    scaled = constrained_lsq(np.array([[1.0, 1.0]]) @ scaler, np.array([1.0]))
    x_back = scaler @ scaled.x
    out.write(
        f"列缩放例子：原始最小范数解为 {vector(raw.x)}，在变量 "
        "`z=(x1/2,x2)` 中最小范数后变换回 x 为 "
        f"{vector(x_back)}。两者目标值都为 0，但原坐标范数分别为 "
        f"{np.linalg.norm(raw.x):.6f} 与 {np.linalg.norm(x_back):.6f}。"
        "因此列缩放后必须明确优化的是原 `||x||₂` 还是新坐标范数。\n\n"
    )

    out.write("## 6. 阈值和容差约定\n\n")
    out.write(
        "- 数值秩：保留满足 `σ_i > atol + rtol σ_max` 的奇异值；默认 "
        "`rtol=max(shape)*eps`。\n"
        "- 可行性：检验 `||U₀ᵀd||₂`；默认容差为 "
        "`max(m,n)*eps*||d||₂`，可用 `constraint_tol` 覆盖。\n"
        "- 阈值不确定性：若正奇异值落在 `0.5τ ≤ σ_i ≤ 2τ`，触发 rank "
        "ambiguity；`τ=atol+rtolσ_max`。\n"
        "- 有效条件数：`σ_max/σ_min(retained)`；完整条件数在有零奇异值时为 ∞。\n"
        "- 缩放：默认不做隐式行列缩放，范数是输入原始坐标的欧氏范数。\n"
    )
    return out.getvalue()


if __name__ == "__main__":
    report = main()
    with open("DIAGNOSTICS.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
