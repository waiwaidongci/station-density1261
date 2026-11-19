"""收紧内存预算（8MB）验证分块自适应：仍能完成且不物化 n*n 矩阵。

另验证 MemoryBudget 越界即抛 LooMemoryError，杜绝隐式大距离矩阵。
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, '/tmp/pylibs')
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'backend'))

import numpy as np  # noqa: E402
from sklearn.cluster import DBSCAN  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.services import loo_engine as eng  # noqa: E402


def main(n=1200):
    rng = np.random.default_rng(3)
    x = np.vstack([
        rng.normal(mu, 0.9, (n // 4, 3)) for mu in
        ([-4, -4, -4], [4, -4, 4], [-4, 4, 4], [4, 4, -4])
    ] + [rng.uniform(-9, 9, (n - 4 * (n // 4), 3))])
    eps, ms = 0.55, 5
    work = Path('/tmp/loo-tight-work')
    shutil.rmtree(work, ignore_errors=True)

    # 8MB 预算：n*n 浮点矩阵需 ~11.5MB，证明不可能整体物化
    limit = 8 * 1024 * 1024
    budget = eng.MemoryBudget(limit)
    b = eng.build_baseline(x, eps, ms, work, budget)
    assert budget.peak <= limit, budget.snapshot()
    lab = DBSCAN(eps=eps, min_samples=ms).fit(
        StandardScaler().fit_transform(x)).labels_
    assert np.array_equal(lab, b.labels)
    for k in range(0, n, 37):
        res = eng.evaluate_trial(b, k)
        if res['degenerate']:
            continue
        xm = np.delete(x, k, axis=0)
        sc = StandardScaler().fit(xm)
        o = DBSCAN(eps=eps, min_samples=ms).fit(sc.transform(xm)).labels_ == -1
        assert res['noise_after'] == int(o.sum())
    print(f'8MB 预算下完成 n={n}，峰值={budget.peak/1024/1024:.2f}MB，'
          f'抽样试算与 oracle 一致')

    # 预算守卫：直接登记 n*n 矩阵必须被拒绝
    guard = eng.MemoryBudget(limit)
    try:
        guard.alloc('forbidden', (n, n), np.float64)
    except eng.LooMemoryError:
        print('预算守卫正确拒绝全站距离矩阵的整体分配')
    else:
        raise AssertionError('预算守卫失效')


if __name__ == '__main__':
    main()
