# atk-rl20-20260930-r19-constrained-least-squares

Empty baseline.


## 约束最小二乘工具

核心入口是 `constrained_lsq.constrained_lsq(A, b, C, d)`，用于求解秩亏等式约束问题：

```python
from constrained_lsq import constrained_lsq

result = constrained_lsq(A, b, C, d)
x = result.x
```

结果对象包含约束残差、最小二乘残差、投影梯度残差、特解/零空间修正范数、奇异值、数值秩、条件数和阈值不确定性诊断。数学推导见 `MATH.md`，预生成实验报告见 `DIAGNOSTICS.md`。

运行测试和重新生成报告：

```bash
python -m unittest -v
python diagnostics.py
```

测试使用独立的 Python `Fraction` RREF 参考解核验小型整数矩阵，不依赖同一个浮点最小二乘例程。
