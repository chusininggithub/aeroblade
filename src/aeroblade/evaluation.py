"""检测性能评价指标。"""

import numpy as np
from sklearn.metrics import roc_curve


def tpr_at_max_fpr(y_true, y_score, max_fpr) -> float:
    """Return the TPR that ensures a certain maximum FPR."""
    # 论文表格里报告的 TPR@FPR=1%：固定一个可接受的误报率上限，看能抓到
    # 多少生成图。用 sklearn 的 ROC 曲线取点，而不是自己扫阈值。
    fpr, tpr, _ = roc_curve(
        y_true=y_true,
        y_score=y_score,
        # 关掉中间点抽稀，保证每个可能阈值都有一行，取到的 TPR 才最接近
        # 「FPR 刚好不超过 max_fpr」的真实值。
        drop_intermediate=False,
    )
    # np.argmax 返回第一个 True 的下标，即第一个 fpr > max_fpr 的位置；
    # 减 1 后退回满足 fpr <= max_fpr 的最后一个点。
    index = np.argmax(fpr > max_fpr) - 1  # np.argmax returns index of first True
    return tpr[index]
