"""5000 站规模：内存预算、流式距离、watch 边落盘与计时；并抽样 oracle 复核。"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, '/tmp/pylibs')
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'backend'))

import numpy as np  # noqa: E402
from sklearn.cluster import DBSCAN  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.services import loo_engine as eng  # noqa: E402

LIMIT = 128 * 1024 * 1024


def main(n=5000):
    rng = np.random.default_rng(5)
    # 混合团结构 + 约 12% 稀疏点，制造噪声站与边界站
    dense_n = int(n * 0.88)
    parts = [rng.normal(mu, 0.9, (dense_n // 4, 3)) for mu in
             ([-4, -4, -4], [4, -4, 4], [-4, 4, 4], [4, 4, -4])]
    x = np.vstack(parts + [rng.uniform(-9, 9, (n - dense_n, 3))])
    eps, ms = 0.55, 5
    work = Path('/tmp/loo-scale-work')
    shutil.rmtree(work, ignore_errors=True)
    budget = eng.MemoryBudget(LIMIT)

    t0 = time.time()
    b = eng.build_baseline(x, eps, ms, work, budget)
    t1 = time.time()
    print(f'基线构建 {t1 - t0:.1f}s n={n} watch={b.watch_count} '
          f'watch文件={ (work/"watch.bin").stat().st_size/1024/1024:.1f}MB '
          f'内存峰值={budget.peak/1024/1024:.1f}MB')
    assert budget.peak <= LIMIT

    # 基线标签与 sklearn 一致
    lab = DBSCAN(eps=eps, min_samples=ms).fit(
        StandardScaler().fit_transform(x)).labels_
    assert np.array_equal(lab, b.labels)
    print('基线标签与 sklearn 一致，噪声数 =', int((lab == -1).sum()))

    # 抽 20 个站做完整 oracle 复核（含噪声站/核心站/随机站）
    rng2 = np.random.default_rng(9)
    sample = set(rng2.choice(n, 14, replace=False).tolist())
    sample |= set(np.nonzero(b.labels == -1)[0][:3])
    sample |= set(np.nonzero(b.core)[0][:3])
    t2 = time.time()
    for k in sorted(sample):
        res = eng.evaluate_trial(b, k)
        xm = np.delete(x, k, axis=0)
        sc = StandardScaler().fit(xm)
        if np.any(sc.var_ == 0):
            assert res['degenerate']
            continue
        o_lab = DBSCAN(eps=eps, min_samples=ms).fit(sc.transform(xm)).labels_ == -1
        keep = np.ones(n, bool)
        keep[k] = False
        base_noise = b.labels == -1
        assert res['noise_after'] == int(o_lab.sum())
        assert res['noise_flips'] == int(np.count_nonzero(base_noise[keep] != o_lab))
    t3 = time.time()
    print(f'抽样 {len(sample)} 站 oracle 复核全部通过，单站均值 {(t3-t2)/len(sample)*1000:.1f}ms')

    # 全量 5000 站计时（不重跑聚类，仅图增量）
    t4 = time.time()
    degen = 0
    flips = []
    for k in range(n):
        r = eng.evaluate_trial(b, k)
        if r['degenerate']:
            degen += 1
        else:
            flips.append(r['noise_flips'])
    t5 = time.time()
    print(f'全量 {n} 站试算 {t5-t4:.1f}s（均值 {(t5-t4)/n*1000:.2f}ms/站），'
          f'退化={degen}，最大翻转={max(flips)}，非零翻转站数={sum(v>0 for v in flips)}')
    assert budget.peak <= LIMIT, budget.snapshot()
    print('5000 站规模测试 PASSED，最终内存', budget.snapshot())


if __name__ == '__main__':
    main()
