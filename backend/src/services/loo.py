"""逐站剔除影响试算的编排：分批、断点恢复、坏批回撤与发布前校验。

存储（ARTIFACT_DIR/loo/<job_id>/）：
- meta.json          任务参数与冻结输入校验和（恢复时核对，防止串任务/脏目录）
- watch.bin          watch 边端点 int32 (m,2)，memmap 读取，不占内存预算
- watch_flags.bin    watch 边基线邻域标记 uint8 (m,)
- batches/b-xxxx.json 每批完好结果（批边界 + 翻转计数/退化标记）
- journal.log        批提交/回撤的只增流水
- result.json        全部批次齐全且校验通过后的发布快照（与 DB result 同源）

中断恢复语义：
- 启动时重放 journal，只沿用“已提交且文件完好、校验和匹配”的批次；
- 损坏/不完整批次的贡献从计数中回撤并删除文件，随后重试该批，不重复计入；
- 所有批次齐全、汇总校验通过后才写 result.json 与 DB.result，研究页面方可读到排名。
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from pathlib import Path

from ..config.settings import (ARTIFACT_DIR, LOO_BATCH_SIZE, LOO_MEMORY_LIMIT,
                               LOO_QUEUE)
from ..models.loo import LooJob
from ..models.run import Run
from ..utils.serialization import now
from . import loo_engine as eng
from .audit import log
from .runs import queue

logger = logging.getLogger('loo')


def job_dir(job_id: int) -> Path:
    return ARTIFACT_DIR / 'loo' / str(job_id)


def _checksum_records(records: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True).encode()).hexdigest()


def _write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False),
                   encoding='utf-8')
    tmp.replace(path)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def create_job(db, user, run_id: int, batch_size: int | None = None) -> dict:
    r = db.get(Run, run_id)
    if not r:
        from fastapi import HTTPException
        raise HTTPException(404, '未找到分析任务')
    if r.state != 'completed' or not r.result:
        from fastapi import HTTPException
        raise HTTPException(409, '仅已完成且结果完好的冻结研究可发起逐站剔除试算')
    bs = int(batch_size or LOO_BATCH_SIZE)
    if not 1 <= bs <= 5000:
        from fastapi import HTTPException
        raise HTTPException(422, '每批站数需在 1–5000 之间')
    existing = (db.query(LooJob)
                .filter(LooJob.run_id == run_id,
                        LooJob.state.in_(('queued', 'running')))
                .one_or_none())
    if existing:
        from fastapi import HTTPException
        raise HTTPException(409, '该研究已有进行中的逐站剔除试算，请等待完成或查看失败原因')
    job = LooJob(run_id=run_id, state='queued', batch_size=bs,
                 total=len(r.snapshot['records']), finished=0, degenerate=0,
                 created_at=now(), updated_at=now())
    db.add(job)
    db.flush()
    d = job_dir(job.id)
    d.mkdir(parents=True, exist_ok=True)
    _write_json(d / 'meta.json', {
        'job_id': job.id,
        'run_id': run_id,
        'batch_size': bs,
        'total': job.total,
        'parameters': r.parameters,
        'records_checksum': _checksum_records(r.snapshot['records']),
        'created_at': job.created_at,
    })
    (d / 'journal.log').write_text(
        f"{now()} created job={job.id} run={run_id} batches="
        f"{math.ceil(job.total / bs)}\n", encoding='utf-8')
    log(db, user, 'loo.queued', job.id, f'run={run_id} n={job.total}')
    db.commit()
    try:
        queue.rpush(LOO_QUEUE, job.id)
    except Exception:
        job.state = 'failed'
        job.error = '队列暂不可用，请重新发起试算'
        db.commit()
        from fastapi import HTTPException
        raise HTTPException(503, job.error)
    return _job_row(job)


def _job_row(job: LooJob) -> dict:
    return {
        'id': job.id, 'run_id': job.run_id, 'state': job.state,
        'batch_size': job.batch_size, 'total': job.total,
        'finished': job.finished, 'degenerate': job.degenerate,
        'error': job.error, 'created_at': job.created_at,
        'updated_at': job.updated_at,
        'progress': (job.finished / job.total) if job.total else 0.0,
        'result': job.result,
    }


def list_jobs(db, run_id: int | None = None):
    q = db.query(LooJob)
    if run_id is not None:
        q = q.filter_by(run_id=run_id)
    return [_job_row(j) for j in q.order_by(LooJob.id.desc())]


def get_job(db, job_id: int) -> dict:
    job = db.get(LooJob, job_id)
    if not job:
        from fastapi import HTTPException
        raise HTTPException(404, '未找到逐站剔除试算任务')
    return _job_row(job)


# ----------------------------- worker 侧执行 -----------------------------

def _append_journal(d: Path, line: str) -> None:
    with open(d / 'journal.log', 'a', encoding='utf-8') as f:
        f.write(f'{now()} {line}\n')
        f.flush()


def _batch_path(d: Path, bidx: int) -> Path:
    return d / 'batches' / f'b-{bidx:05d}.json'


def _recover_batches(d: Path, total: int, bs: int):
    """重放批文件，返回 (完好批集合, 汇总字典)。

    坏批（文件缺失/JSON 损坏/校验和不符/越界/重复索引）一律回撤贡献并删除。
    """
    bdir = d / 'batches'
    bdir.mkdir(exist_ok=True)
    good: set[int] = set()
    agg = {'finished': 0, 'degenerate': 0, 'flips': {},
           'noise_before': {}, 'noise_after': {}}
    expected_batches = math.ceil(total / bs)
    for bidx in range(expected_batches):
        p = _batch_path(d, bidx)
        if not p.exists():
            continue
        try:
            payload = _read_json(p)
            assert payload['batch_index'] == bidx
            lo, hi = bidx * bs, min(total, (bidx + 1) * bs)
            assert payload['range'] == [lo, hi]
            trials = payload['trials']
            assert len(trials) == hi - lo
            idxs = [t['removed_index'] for t in trials]
            assert idxs == list(range(lo, hi))
            assert len(set(idxs)) == len(idxs)
            # 单批校验和：覆盖全部试算条目
            digest = hashlib.sha256(
                json.dumps(trials, sort_keys=True).encode()).hexdigest()
            assert digest == payload['checksum']
        except Exception as e:
            _append_journal(d, f'rollback batch={bidx} reason={type(e).__name__}')
            p.unlink(missing_ok=True)
            continue
        good.add(bidx)
        for t in trials:
            agg['finished'] += 1
            if t['degenerate']:
                agg['degenerate'] += 1
            else:
                agg['flips'][str(t['removed_index'])] = t['noise_flips']
                agg['noise_before'][str(t['removed_index'])] = t['noise_before']
                agg['noise_after'][str(t['removed_index'])] = t['noise_after']
    return good, agg


def _build_publish_payload(job: LooJob, run: Run, b: eng.Baseline,
                           agg: dict, budget: eng.MemoryBudget) -> dict:
    """全部批次齐全后的汇总与发布前校验；任何不一致都抛错，任务保持 failed。"""
    records = run.snapshot['records']
    if agg['finished'] != job.total:
        raise ValueError(f'试算不齐全：{agg["finished"]}/{job.total}')
    # 1) 基线与冻结研究等价
    eng.validate_against_run(b, run.result)
    # 2) 每个非退化站都有翻转计数，退化站清单与引擎解析预判一致
    deg_set = {int(i) for i in b.degenerate_indices.tolist()}
    got_deg = set()
    for bidx_path in sorted((job_dir(job.id) / 'batches').glob('b-*.json')):
        for t in _read_json(bidx_path)['trials']:
            if t['degenerate']:
                got_deg.add(t['removed_index'])
    if got_deg != deg_set:
        raise ValueError('退化试算清单与解析预判不一致')
    flips = agg['flips']
    if set(int(k) for k in flips) != (set(range(job.total)) - deg_set):
        raise ValueError('非退化试算翻转计数覆盖不完整')
    # 3) 翻转数有界性：不超过剩余样本数
    if any(not (0 <= v < job.total) for v in flips.values()):
        raise ValueError('翻转计数越界')

    references = [r['reference'] for r in records]
    ranking = sorted(
        ({'reference': references[int(k)], 'removed_index': int(k),
          'noise_flips': v,
          'noise_before': agg['noise_before'][k],
          'noise_after': agg['noise_after'][k]}
         for k, v in flips.items()),
        key=lambda item: (-item['noise_flips'], item['removed_index']))
    degenerate_list = [
        {'reference': references[k], 'removed_index': k,
         'reason': '剔除后剩余样本出现常量列，标准化尺度退化，无法判定密度邻域'}
        for k in sorted(deg_set)]
    baseline_noise = int((b.labels == -1).sum())
    payload = {
        'run_id': run.id,
        'run_name': run.name,
        'parameters': run.parameters,
        'total_stations': job.total,
        'evaluated_stations': job.total - len(deg_set),
        'degenerate_stations': len(deg_set),
        'baseline_noise_count': baseline_noise,
        'ranking': ranking,
        'degenerate_trials': degenerate_list,
        'method': ('逐站剔除后对剩余样本重新标准化，eps/min_samples 沿用冻结研究；'
                   '排名按其他站噪声身份翻转数降序；尺度退化试算单列不参评'),
        'records_checksum': _checksum_records(records),
        'memory': budget.snapshot(),
        'watch_edges': b.watch_count,
        'published_at': now(),
    }
    return payload


def execute_job(job_id: int, Session) -> None:
    """worker 调用：含恢复、分批、回撤、校验与发布。全程进度落库可见。"""
    with Session() as db:
        job = db.get(LooJob, job_id)
        if not job or job.state not in ('queued', 'running'):
            return
        run = db.get(Run, job.run_id)
        if not run or run.state != 'completed':
            job.state = 'failed'
            job.error = '冻结研究不可用，无法试算'
            job.updated_at = now()
            db.commit()
            return
        d = job_dir(job.id)
        try:
            meta = _read_json(d / 'meta.json')
            records = run.snapshot['records']
            if meta['records_checksum'] != _checksum_records(records):
                raise ValueError('冻结输入校验和与任务目录不一致，拒绝恢复')
            if meta['parameters'] != run.parameters:
                raise ValueError('邻域参数与冻结研究不一致，拒绝恢复')
            bs, total = job.batch_size, job.total
            good, agg = _recover_batches(d, total, bs)
            job.state = 'running'
            job.finished = agg['finished']
            job.degenerate = agg['degenerate']
            job.updated_at = now()
            db.commit()

            budget = eng.MemoryBudget(LOO_MEMORY_LIMIT)
            x = eng.records_to_matrix(records)
            baseline = eng.build_baseline(
                x, run.parameters['eps'], run.parameters['min_samples'],
                d, budget)
            # 基线等价校验提前做，失败则整任务失败，不浪费批次计算
            eng.validate_against_run(baseline, run.result)

            expected_batches = math.ceil(total / bs)
            for bidx in range(expected_batches):
                if bidx in good:
                    continue  # 只沿用完好批次，不重复计入
                lo, hi = bidx * bs, min(total, (bidx + 1) * bs)
                trials = []
                for k in range(lo, hi):
                    trials.append(eng.evaluate_trial(baseline, k))
                checksum = hashlib.sha256(
                    json.dumps(trials, sort_keys=True).encode()).hexdigest()
                payload = {'batch_index': bidx, 'range': [lo, hi],
                           'trials': trials, 'checksum': checksum,
                           'computed_at': now()}
                # 原子落盘后再记 journal，崩溃在中途则恢复时按坏批回撤重试
                _write_json(_batch_path(d, bidx), payload)
                _append_journal(d, f'committed batch={bidx} range={lo}:{hi}')
                good.add(bidx)
                for t in trials:
                    agg['finished'] += 1
                    if t['degenerate']:
                        agg['degenerate'] += 1
                    else:
                        agg['flips'][str(t['removed_index'])] = t['noise_flips']
                        agg['noise_before'][str(t['removed_index'])] = t['noise_before']
                        agg['noise_after'][str(t['removed_index'])] = t['noise_after']
                job.finished = agg['finished']
                job.degenerate = agg['degenerate']
                job.updated_at = now()
                # 长任务分批期间持续刷新 worker 心跳，避免健康检查误判
                (ARTIFACT_DIR / 'heartbeat').write_text(str(time.time()))
                db.commit()

            # 全部齐全后统一校验，通过才发布（页面此前读不到 result）
            result = _build_publish_payload(job, run, baseline, agg, budget)
            _write_json(d / 'result.json', result)
            job.result = result
            job.state = 'completed'
            job.error = ''
            job.updated_at = now()
            log(db, 'worker', 'loo.completed', job.id,
                f"ranked={result['evaluated_stations']} "
                f"degenerate={result['degenerate_stations']} "
                f"peak={budget.peak}")
            db.commit()
        except Exception as e:
            logger.exception('loo job %s failed', job_id)
            job.state = 'failed'
            job.error = str(e)[:2000]
            job.updated_at = now()
            try:
                _append_journal(d, f'failed reason={type(e).__name__}: {e}'[:500])
            except Exception:
                pass
            log(db, 'worker', 'loo.failed', job.id, job.error)
            db.commit()
