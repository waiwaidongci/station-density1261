"""强压力：多随机种子/参数/规模，全量逐站 sklearn oracle 对照（含退化构造）。

重点验证按维独立缩放的 watch 带不漏任何跨 eps 的边。
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

LIMIT = 128 * 1024 * 1024


def oracle_noise(x, k, eps, ms):
    xm = np.delete(x, k, axis=0)
    sc = StandardScaler().fit(xm)
    if np.any(sc.var_ == 0):
        return None
    return DBSCAN(eps=eps, min_samples=ms).fit(sc.transform(xm)).labels_ == -1


def case(seed, n, eps, ms, mode):
    rng = np.random.default_rng(seed)
    if mode == 'blobs':
        x = np.vstack([
            rng.normal(mu, rng.uniform(0.5, 1.4), (n // 3, 3))
            for mu in ([-3, -3, -3], [3, -3, 3], [-3, 3, 3])
        ] + [rng.uniform(-7, 7, (n - 3 * (n // 3), 3))])
    elif mode == 'uniform':
        x = rng.uniform(-5, 5, (n, 3))
    else:  # 近重复 + 离散列，制造退化与贴边
        x = rng.normal(0, 1, (n, 3))
        x[: max(2, n // 8)] = x[0]
        if seed % 2 == 0:
            x[1, 0] = x[0, 0] + 7.0  # 温度列异类唯一值
    work = Path(f'/tmp/loo-stress-{seed}')
    shutil.rmtree(work, ignore_errors=True)
    b = eng.build_baseline(x, eps, ms, work, eng.MemoryBudget(LIMIT))

    # 基线标签等价
    base_lab = DBSCAN(eps=eps, min_samples=ms).fit(
        StandardScaler().fit_transform(x)).labels_
    assert np.array_equal(base_lab, b.labels), f'seed{seed} 基线不等价'
    base_noise = base_lab == -1

    keep_all = np.ones(n, bool)
    for k in range(n):
        r = eng.evaluate_trial(b, k)
        o = oracle_noise(x, k, eps, ms)
        if o is None:
            assert r['degenerate'], f'seed{seed} k{k} 应退化'
            continue
        assert not r['degenerate'], f'seed{seed} k{k} 误判退化'
        keep = keep_all.copy()
        keep[k] = False
        expect = int(np.count_nonzero(base_noise[keep] != o))
        assert r['noise_flips'] == expect, (
            f'seed{seed} k{k} engine={r["noise_flips"]} oracle={expect} '
            f'watch={b.watch_count}')
    assert b.budget.peak <= LIMIT
    return b.watch_count, b.budget.peak


def main():
    settings = [
        (1, 60, 0.55, 5, 'blobs'),
        (2, 45, 0.35, 3, 'uniform'),
        (3, 80, 0.7, 4, 'blobs'),
        (4, 38, 1.1, 3, 'uniform'),
        (5, 50, 0.9, 5, 'discrete'),
        (6, 72, 0.5, 6, 'blobs'),
        (7, 30, 2.0, 2, 'uniform'),
        (8, 55, 0.65, 4, 'discrete'),
        (9, 90, 0.45, 5, 'blobs'),
        (10, 42, 1.3, 3, 'uniform'),
    ]
    for seed, n, eps, ms, mode in settings:
        w, peak = case(seed, n, eps, ms, mode)
        print(f'seed={seed} n={n} eps={eps} ms={ms} mode={mode} '
              f'watch={w} peak={peak/1024:.0f}KB OK')
    print('STRESS ORACLE PASSED')


if __name__ == '__main__':
    main()
