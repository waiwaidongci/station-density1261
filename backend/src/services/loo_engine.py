"""逐站剔除（LOO）噪声影响引擎。

业务口径（与已冻结研究的 algorithm.calculate 对齐）：
- 输入为已完成密度任务冻结的样本，邻域参数 eps / min_samples 照旧；
- 每次剔除一座微站后，用剩余 n-1 个样本重新标准化，再判定 DBSCAN 噪声身份；
- 排名指标 = 其他站（剩余样本）噪声身份相对基线的翻转数；
- 剔除后出现常量列（尺度退化）的试算单列，不参与排名。

工程约束：
- 额外内存预算默认 128MB：所有引擎内大块 ndarray 经 MemoryBudget 计数，
  距离按块流式计算，不物化 n*n 全站距离矩阵（n=5000 时该矩阵约 200MB，必然越界）；
- 不逐站重跑聚类：基线只构建一次邻域图与 DBSCAN 标签，每个剔除试算仅做
  图增量判定（删顶点、watch 边重判、核心状态更新、噪声可达性判定）；
- watch 边（在任一非退化试算中可能跨越 eps 的边，按各维独立缩放求逐边距离极值）
  落盘为 memmap，不占预算内存；
- 重标化的均值/尺度用解析公式更新，数学上等价于对剩余样本重新 fit StandardScaler。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURES = ('temperature', 'humidity', 'particles')
EPS_TOL = 1e-12
DEFAULT_MEMORY_LIMIT = 128 * 1024 * 1024


class LooMemoryError(MemoryError):
    """额外内存预算越界（区别于普通 MemoryError，便于上层转 422/任务失败）。"""


class MemoryBudget:
    """登记引擎申请的大块 ndarray，超出预算立即拒绝，杜绝隐式距离矩阵。"""

    def __init__(self, limit_bytes: int = DEFAULT_MEMORY_LIMIT):
        self.limit = int(limit_bytes)
        self.used = 0
        self.peak = 0
        self._blocks: dict[str, int] = {}

    def register(self, name: str, array: np.ndarray) -> np.ndarray:
        size = int(array.nbytes)
        if name in self._blocks:
            raise RuntimeError('内存块重复登记: ' + name)
        if self.used + size > self.limit:
            raise LooMemoryError(
                f'额外内存预算不足：本块 {size} 字节，已用 {self.used}，'
                f'预算 {self.limit}；已改用分块流式计算，不应出现此情况')
        self._blocks[name] = size
        self.used += size
        self.peak = max(self.peak, self.used)
        return array

    def release(self, name: str) -> None:
        self.used -= self._blocks.pop(name)

    def alloc(self, name: str, shape, dtype=np.float64) -> np.ndarray:
        return self.register(name, np.empty(shape, dtype=dtype))

    def release_all(self) -> None:
        self._blocks.clear()
        self.used = 0

    def snapshot(self) -> dict:
        return {'limit_bytes': self.limit, 'used_bytes': self.used,
                'peak_bytes': self.peak,
                'headroom_bytes': self.limit - self.used}


def _block_rows(n: int, d: int, budget: MemoryBudget,
                target_bytes: int = 4 * 1024 * 1024) -> int:
    """按目标块大小（默认 4MB）计算距离块行数，并保证不越过剩余预算。"""
    row_bytes = max(1, n * d * 8)
    rows = max(8, target_bytes // row_bytes)
    while rows > 8 and budget.used + rows * n * 8 > budget.limit:
        rows //= 2
    if budget.used + rows * n * 8 > budget.limit:
        raise LooMemoryError('即使最小距离块也超出额外内存预算')
    return int(rows)


def _pairwise_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a×b 欧氏距离块；内部再按行切片，限制广播临时数组规模。"""
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.float64)
    inner_step = max(1, 2_000_000 // max(1, b.shape[0] * a.shape[1]))
    for s in range(0, a.shape[0], inner_step):
        diff = a[s:s + inner_step, None, :] - b[None, :, :]
        out[s:s + inner_step] = np.sqrt(
            np.einsum('ijk,ijk->ij', diff, diff))
    return out


@dataclass
class Baseline:
    z: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    labels: np.ndarray
    core: np.ndarray
    neighbor_count: np.ndarray
    watch: np.memmap | None
    watch_inside: np.memmap | None
    watch_count: int
    sumsq_delta: np.ndarray
    a_min: np.ndarray
    a_max: np.ndarray
    budget: MemoryBudget
    n: int
    d: int
    eps: float
    min_samples: int
    degenerate_indices: np.ndarray  # 剔除即常量列的站（解析预判）


def build_baseline(x: np.ndarray, eps: float, min_samples: int,
                   work_dir: Path,
                   budget: MemoryBudget | None = None) -> Baseline:
    """阶段一：流式构建基线邻域图、watch 边与重标化统计量。

    基线 DBSCAN 标签在本函数内复刻 sklearn 语义生成一次；调用方应再与已冻结
    任务结果做标签等价校验（见 validate_against_run）。
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    budget = budget or MemoryBudget()
    n, d = x.shape
    if n < int(min_samples):
        raise ValueError('样本数少于核心点最少邻居数')
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    if np.any(scale == 0):
        raise ValueError('特征含常量列，无法解释标准化密度，请更换数据')

    z = budget.alloc('z', (n, d))
    np.true_divide(x - mean, scale, out=z)
    neighbor_count = budget.alloc('neighbor_count', n, np.int32)
    sumsq_delta = budget.alloc('sumsq_delta', (n, d))
    np.square(x - mean, out=sumsq_delta)
    total_sumsq = sumsq_delta.sum(axis=0)

    # 解析 LOO 方差：s_{-k,j}^2 = (S_j - n*δ²/(n-1)) / (n-1)
    var_loo = ((total_sumsq[None, :]
                - n * sumsq_delta / (n - 1.0)) / (n - 1.0))
    # 退化判定以“剩余列是否只剩唯一取值”为准（精确），避免平方和相减的
    # 浮点残差（实测可达 1e-8）把真常量列误判为非退化：
    # 某列仅两个取值且被剔点是其中唯一出现者时，剔除后该列才常量；
    # 三个及以上不同取值时，剔除一点至少剩两个值，不可能退化。
    degenerate = np.zeros(n, dtype=bool)
    for j in range(d):
        vals, counts = np.unique(x[:, j], return_counts=True)
        if vals.size == 2:
            singleton_vals = vals[counts == 1]
            if singleton_vals.size:
                hit = np.isin(x[:, j], singleton_vals)
                degenerate |= hit
        # vals.size==1 在基线已被 scale==0 拒绝；>=3 不退化
    # 数值上非正的方差同样视为退化（兜底）
    degenerate |= np.any(var_loo <= 0.0, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        a_all = scale[None, :] / np.sqrt(
            np.where(var_loo > 0, var_loo, np.nan))
    # 退化站不参与任何真实试算，其缩放因子从 watch 带极值统计中剔除
    a_all[degenerate, :] = np.nan
    if degenerate.all():
        a_min = np.full(d, 1.0)
        a_max = np.full(d, 1.0)
    else:
        a_min = np.nanmin(a_all, axis=0)
        a_max = np.nanmax(a_all, axis=0)

    # watch 带（按维独立缩放，欧氏距离极值）：
    # 边 (i,j) 基线坐标差 u_j=z_i-z_j，试算差为 u_j*a_k（平移不改变两点差）。
    # 各维缩放独立落在 [a_min_j, a_max_j]，故该边在任一非退化试算中的距离范围：
    #   d_min = ||u_j * a_min||, d_max = ||u_j * a_max||（取逐维绝对值后的因子）
    # 仅当 d_min <= eps <= d_max 时该边才可能跨越 eps，需落盘 watch 重判。
    lo_factor = np.minimum(a_min, a_max)
    hi_factor = np.maximum(a_min, a_max)

    watch_path = work_dir / 'watch.bin'
    flag_path = work_dir / 'watch_flags.bin'
    block_rows = _block_rows(n, d, budget)
    watch_count = 0
    with open(watch_path, 'wb') as wf, open(flag_path, 'wb') as ff:
        for start in range(0, n, block_rows):
            end = min(n, start + block_rows)
            rows = end - start
            # 每块独立登记、用毕释放；任何时刻至多一个距离块在预算内
            dmat = budget.alloc(f'dist_buf_{start}', (rows, n), np.float64)
            dmat[:] = _pairwise_block(z[start:end], z)
            inside = dmat <= eps + EPS_TOL
            neighbor_count[start:end] = inside.sum(axis=1).astype(np.int32)
            for li in range(rows):
                gi = start + li
                row = dmat[li, gi + 1:]
                if row.size == 0:
                    continue
                # 逐边距离可达极值：基线差向量逐维乘独立缩放因子
                u = z[gi] - z[gi + 1:]                      # (m,d)
                d_min = np.sqrt(
                    np.einsum('ij,ij->i', u * lo_factor, u * lo_factor))
                d_max = np.sqrt(
                    np.einsum('ij,ij->i', u * hi_factor, u * hi_factor))
                # 仅保留在某个非退化试算中可能跨越 eps 的边
                band = (d_min <= eps + EPS_TOL) & (d_max >= eps - EPS_TOL)
                if not band.any():
                    continue
                jdx = np.nonzero(band)[0] + (gi + 1)
                pairs = np.empty((jdx.size, 2), dtype=np.int32)
                pairs[:, 0] = gi
                pairs[:, 1] = jdx
                wf.write(pairs.tobytes())
                ff.write((dmat[li, jdx] <= eps + EPS_TOL)
                         .astype(np.uint8).tobytes())
                watch_count += jdx.size
            budget.release(f'dist_buf_{start}')

    watch = (np.memmap(watch_path, dtype=np.int32, mode='r',
                       shape=(watch_count, 2))
             if watch_count else None)
    watch_inside = (np.memmap(flag_path, dtype=np.uint8, mode='r',
                              shape=(watch_count,))
                    if watch_count else None)

    core = budget.alloc('core', n, np.bool_)
    np.greater_equal(neighbor_count, int(min_samples), out=core)
    labels = _baseline_dbscan(z, eps, core, budget, n)

    return Baseline(z=z, mean=mean, scale=scale, labels=labels, core=core,
                    neighbor_count=neighbor_count, watch=watch,
                    watch_inside=watch_inside, watch_count=watch_count,
                    sumsq_delta=sumsq_delta, a_min=a_min, a_max=a_max,
                    budget=budget, n=n, d=d, eps=eps,
                    min_samples=int(min_samples),
                    degenerate_indices=np.nonzero(degenerate)[0])


def _baseline_dbscan(z: np.ndarray, eps: float, core: np.ndarray,
                     budget: MemoryBudget, n: int) -> np.ndarray:
    """复刻 sklearn DBSCAN：核心点按索引顺序播种，邻域 BFS 密度连通。"""
    labels = budget.alloc('labels', n, np.int32)
    labels.fill(-1)
    cluster_id = 0
    d = z.shape[1]
    block_rows = max(1, min(256, 2_000_000 // max(1, n * d)))
    for seed in np.nonzero(core)[0]:
        if labels[seed] != -1:
            continue
        labels[seed] = cluster_id
        queue = [int(seed)]
        head = 0
        while head < len(queue):
            v = queue[head]
            head += 1
            if not core[v]:
                continue
            for s in range(0, n, block_rows):
                e = min(n, s + block_rows)
                dvec = np.sqrt(np.einsum('ij,ij->i', z[s:e] - z[v],
                                         z[s:e] - z[v]))
                for gi in np.nonzero(dvec <= eps + EPS_TOL)[0] + s:
                    if labels[gi] == -1:
                        labels[gi] = cluster_id
                        if core[gi]:
                            queue.append(int(gi))
        cluster_id += 1
    return labels


def loo_standardization(b: Baseline, k: int):
    """剔除 k 后的解析重标化参数。

    剩余点 j 的新标准化坐标 z'_j = (z_j - t_k) * a_k，
    其中 a_k = scale/scale_-k，t_k = (mean_-k - mean)/scale。
    返回 None 表示尺度退化（剩余样本出现常量列）。
    """
    n = b.n
    x_k = b.z[k] * b.scale + b.mean
    mean_loo = (b.mean * n - x_k) / (n - 1.0)
    var_loo = ((b.sumsq_delta.sum(axis=0)
                - n * b.sumsq_delta[k] / (n - 1.0)) / (n - 1.0))
    if np.any(var_loo <= 0.0):
        return None
    scale_loo = np.sqrt(var_loo)
    a = b.scale / scale_loo
    t = (mean_loo - b.mean) / b.scale
    return mean_loo, scale_loo, a, t


def evaluate_trial(b: Baseline, k: int, read_block: int = 4096) -> dict:
    """单站剔除试算（图增量，不重跑聚类）。

    返回噪声身份翻转数（仅统计剩余的其他站）、试算前后噪声数；退化站单列。
    """
    n, d, eps, min_samples = b.n, b.d, b.eps, b.min_samples
    if k in b.degenerate_indices:
        return {'removed_index': int(k), 'degenerate': True,
                'noise_flips': None, 'noise_before': None,
                'noise_after': None}
    params = loo_standardization(b, k)
    if params is None:
        return {'removed_index': int(k), 'degenerate': True,
                'noise_flips': None, 'noise_before': None,
                'noise_after': None}
    _, _, a, t = params
    keep = np.ones(n, dtype=bool)
    keep[k] = False

    # 1) 删除顶点 k：其试算邻域内的剩余点邻居数减一
    count = b.neighbor_count.astype(np.int32, copy=True)
    diff_k = (b.z - b.z[k]) * a
    d_k = np.sqrt(np.einsum('ij,ij->i', diff_k, diff_k))
    k_neighbors = d_k <= eps + EPS_TOL
    k_neighbors[k] = False
    count[k_neighbors] -= 1

    # 2) watch 边在试算尺度下重判（涉及 k 的边已随顶点消失）
    if b.watch is not None:
        for s in range(0, b.watch_count, read_block):
            e = min(b.watch_count, s + read_block)
            pairs = b.watch[s:e]
            base_inside = b.watch_inside[s:e].astype(bool)
            valid = (pairs[:, 0] != k) & (pairs[:, 1] != k)
            if not valid.any():
                continue
            ii, jj = pairs[valid, 0], pairs[valid, 1]
            delta_ij = (b.z[ii] - b.z[jj]) * a
            dist = np.sqrt(np.einsum('ij,ij->i', delta_ij, delta_ij))
            now_inside = dist <= eps + EPS_TOL
            changed = now_inside != base_inside[valid]
            if changed.any():
                step = np.where(now_inside[changed], 1, -1).astype(np.int32)
                np.add.at(count, ii[changed], step)
                np.add.at(count, jj[changed], step)

    core_now = count >= min_samples
    core_now[k] = False

    # 3) 噪声身份：核心点非噪声；非核心点邻域内有核心点则为边界点（非噪声）
    core_points = np.nonzero(core_now)[0]
    noise_after = np.ones(n, dtype=bool)
    noise_after[k] = False
    noise_after[core_points] = False
    candidates = np.nonzero(keep & ~core_now)[0]
    if candidates.size and core_points.size:
        inner = max(1, 2_000_000 // max(1, core_points.size * d))
        for s in range(0, candidates.size, inner):
            e = min(candidates.size, s + inner)
            cand = candidates[s:e]
            zc = b.z[cand] * a - t
            reachable = np.zeros(cand.size, dtype=bool)
            for cs in range(0, core_points.size, inner):
                ce = min(core_points.size, cs + inner)
                zp = b.z[core_points[cs:ce]] * a - t
                dmat = _pairwise_block(zc, zp)
                reachable |= np.any(dmat <= eps + EPS_TOL, axis=1)
                if reachable.all():
                    break
            noise_after[cand] = ~reachable

    noise_before = b.labels == -1
    return {
        'removed_index': int(k),
        'degenerate': False,
        'noise_flips': int(np.count_nonzero(
            noise_before[keep] != noise_after[keep])),
        'noise_before': int(np.count_nonzero(noise_before[keep])),
        'noise_after': int(np.count_nonzero(noise_after[keep])),
    }


def records_to_matrix(records: list[dict]) -> np.ndarray:
    """冻结快照 records -> (n,3) float64 矩阵，顺序与快照一致。"""
    return np.array([[r['values'][f] for f in FEATURES]
                     for r in records], dtype=np.float64)


def validate_against_run(b: Baseline, run_result: dict) -> None:
    """引擎基线标签必须与已冻结任务的逐行噪声身份/核心口径等价，否则拒绝发布。"""
    rows = run_result['rows']
    if len(rows) != b.n:
        raise ValueError('基线校验失败：冻结样本数与引擎输入不一致')
    run_noise = np.array([r['kind'] == '噪声' for r in rows], dtype=bool)
    if not np.array_equal(run_noise, b.labels == -1):
        raise ValueError('基线校验失败：引擎噪声身份与冻结研究结果不一致')
