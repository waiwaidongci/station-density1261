"""负向：基线校验失败不发布；真实链路中尺度退化站单列不参评且不影响其他排名。"""
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, '/tmp/pylibs')
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'backend'))

ART = Path('/tmp/loo-neg-artifacts')
DB = Path('/tmp/loo-neg.db')
shutil.rmtree(ART, ignore_errors=True)
DB.unlink(missing_ok=True)
os.environ.update(ARTIFACT_DIR=str(ART), DATABASE_URL=f'sqlite:///{DB}',
                  LOO_BATCH_SIZE='10', SEED_DEMO='false')

import numpy as np  # noqa: E402
from sklearn.cluster import DBSCAN  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.config.database import Base, engine, Session  # noqa: E402
from src.models import user, source, record, note, audit  # noqa: E402,F401
from src.models.run import Run  # noqa: E402
from src.models.loo import LooJob  # noqa: E402
from src.services.algorithm import calculate  # noqa: E402
from src.services.loo import create_job, execute_job  # noqa: E402
import src.services.loo as loo_mod  # noqa: E402

FEATURES = ['temperature', 'humidity', 'particles']
Base.metadata.create_all(engine)


class FakeQueue:
    def rpush(self, *a, **k):
        pass


loo_mod.queue = FakeQueue()


class Actor:
    email = 'analyst@station-density.demo'


def make_run(records, params, name, tamper=False):
    result = calculate(records, params)
    if tamper:
        # 把第一行噪声身份翻转，制造与重算基线不一致
        first = result['rows'][0]
        first['kind'] = '核心' if first['kind'] == '噪声' else '噪声'
    with Session() as db:
        run = Run(source_id=1, name=name, state='completed', parameters=params,
                  snapshot={'source': {'filename': 'neg.csv'}, 'records': records},
                  result=result, created_at='2026-11-19T00:00:00+00:00')
        db.add(run)
        db.commit()
        return run.id


def main():
    params = {'eps': 0.55, 'min_samples': 5}

    # 1) 基线被篡改 -> 任务失败、无发布
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, (40, 3))
    x[:12] += 3
    records = [{'reference': f'HL-N-{i:04d}',
                'values': {f: float(v) for f, v in zip(FEATURES, row)}}
               for i, row in enumerate(x)]
    rid = make_run(records, params, '被篡改研究', tamper=True)
    with Session() as db:
        jid = create_job(db, Actor(), rid)['id']
    execute_job(jid, Session)
    with Session() as db:
        job = db.get(LooJob, jid)
        assert job.state == 'failed', job.state
        assert job.result is None
        assert '基线校验失败' in job.error, job.error
    print('基线篡改 -> failed 且无排名发布：通过')

    # 2) 确定性退化：温度列仅两个取值，异类点唯一 -> 剔除它后常量
    x2 = rng.normal(0, 1, (24, 3))
    x2[:, 0] = 5.0
    x2[7, 0] = 9.0
    rec2 = [{'reference': f'HL-D-{i:04d}',
             'values': {f: float(v) for f, v in zip(FEATURES, row)}}
            for i, row in enumerate(x2)]
    rid2 = make_run(rec2, params, '退化研究')
    with Session() as db:
        jid2 = create_job(db, Actor(), rid2)['id']
    execute_job(jid2, Session)
    with Session() as db:
        job2 = db.get(LooJob, jid2)
        assert job2.state == 'completed', job2.error
        res = job2.result
    deg = {t['removed_index'] for t in res['degenerate_trials']}
    assert deg == {7}, deg
    assert res['degenerate_stations'] == 1
    assert res['evaluated_stations'] == 23
    ranked = {t['removed_index'] for t in res['ranking']}
    assert 7 not in ranked and len(ranked) == 23

    # 其余 23 站与 sklearn oracle 翻转数一致
    rank_map = {t['removed_index']: t['noise_flips'] for t in res['ranking']}
    base_noise = DBSCAN(eps=0.55, min_samples=5).fit(
        StandardScaler().fit_transform(x2)).labels_ == -1
    for k in range(24):
        if k == 7:
            continue
        xm = np.delete(x2, k, axis=0)
        sc = StandardScaler().fit(xm)
        assert np.all(sc.var_ > 0)
        lab = DBSCAN(eps=0.55, min_samples=5).fit(sc.transform(xm)).labels_ == -1
        keep = np.ones(24, bool)
        keep[k] = False
        assert rank_map[k] == int(np.count_nonzero(base_noise[keep] != lab)), k
    print('尺度退化单列 + 其余站 oracle 一致：通过')


if __name__ == '__main__':
    main()
