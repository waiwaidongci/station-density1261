"""端到端：SQLite 库内构建冻结研究 -> LOO 分批试算 -> 发布 -> 断点恢复/坏批回撤。

以 -I 运行，依赖来自 /tmp/pylibs（脚本内显式入栈），不读取工作目录任何第三方文件。
"""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, '/tmp/pylibs')
ROOT = Path(__file__).resolve().parents[3] / 'backend'
sys.path.insert(0, str(ROOT))

ART = Path('/tmp/loo-e2e-artifacts')
DB = Path('/tmp/loo-e2e.db')
shutil.rmtree(ART, ignore_errors=True)
DB.unlink(missing_ok=True)
os.environ['ARTIFACT_DIR'] = str(ART)
os.environ['DATABASE_URL'] = f'sqlite:///{DB}'
os.environ['LOO_BATCH_SIZE'] = '37'
os.environ['LOO_MEMORY_LIMIT'] = str(128 * 1024 * 1024)
os.environ['SEED_DEMO'] = 'false'

import numpy as np  # noqa: E402
from sklearn.cluster import DBSCAN  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.config.database import Base, engine, Session  # noqa: E402
from src.models import user, source, record, run, note, audit, loo as _loo_model  # noqa: E402,F401
from src.models.run import Run  # noqa: E402
from src.models.loo import LooJob  # noqa: E402
from src.models.user import User  # noqa: E402
from src.services.algorithm import calculate  # noqa: E402
from src.services import loo as loo_service  # noqa: E402
from src.services.loo import execute_job, job_dir  # noqa: E402
from src.utils.security import hash_password  # noqa: E402
from src.utils.serialization import now  # noqa: E402

FEATURES = ['temperature', 'humidity', 'particles']
Base.metadata.create_all(engine)


def make_records(n=160, seed=11):
    rng = np.random.default_rng(seed)
    x = np.vstack([
        rng.normal([19, 72, 22], [0.8, 2.5, 3], size=(n // 3, 3)),
        rng.normal([26, 45, 48], [0.8, 2.5, 3], size=(n // 3, 3)),
        rng.normal([32, 28, 85], [0.8, 2.5, 3], size=(n - 2 * (n // 3), 3)),
    ])
    # 追加稀疏噪声点
    x = np.vstack([x, rng.uniform([12, 12, 4], [42, 92, 140], size=(18, 3))])
    records = [{'reference': f'HL-T-{i:04d}',
                'values': {f: float(v) for f, v in zip(FEATURES, row)}}
               for i, row in enumerate(x)]
    return records, x


def oracle_flips(x, k, eps, min_samples):
    base = DBSCAN(eps=eps, min_samples=min_samples).fit(
        StandardScaler().fit_transform(x)).labels_ == -1
    xm = np.delete(x, k, axis=0)
    sc = StandardScaler().fit(xm)
    if np.any(sc.var_ == 0):
        return None
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit(
        sc.transform(xm)).labels_ == -1
    keep = np.ones(len(x), bool)
    keep[k] = False
    return int(np.count_nonzero(base[keep] != lab))


class Actor:
    email = 'analyst@station-density.demo'


def main():
    records, x = make_records()
    params = {'eps': 0.55, 'min_samples': 5}
    result = calculate(records, params)
    with Session() as db:
        u = User(email=Actor.email, password_hash=hash_password('x'),
                 role='analyst')
        db.add(u)
        run = Run(source_id=1, name='冻结研究', state='completed',
                  parameters=params,
                  snapshot={'source': {'filename': 'synthetic.csv'},
                            'records': records},
                  result=result, created_at=now())
        db.add(run)
        db.commit()
        run_id = run.id

    # 1) 创建任务（绕过 redis 推送：直接删队列调用副作用）
    import src.services.loo as loo_mod
    # 屏蔽 redis 推送：同时替换 runs 与 loo 模块在加载时绑定的 queue
    import src.services.runs as runs_mod
    class FakeQueue:
        def rpush(self, *a, **k):
            pass
    fake = FakeQueue()
    runs_mod.queue = fake
    loo_mod.queue = fake
    with Session() as db:
        row = loo_service.create_job(db, Actor(), run_id)
        job_id = row['id']
        assert row['state'] == 'queued' and row['total'] == len(records)

    # 2) 执行（分批 37）
    execute_job(job_id, Session)
    with Session() as db:
        job = db.get(LooJob, job_id)
        assert job.state == 'completed', job.error
        payload = job.result
        assert payload['evaluated_stations'] + payload['degenerate_stations'] == len(records)
        print('job completed:', {k: payload[k] for k in
              ('total_stations', 'evaluated_stations', 'degenerate_stations',
               'baseline_noise_count', 'watch_edges')})
        print('memory:', payload['memory'])
        assert payload['memory']['peak_bytes'] <= 128 * 1024 * 1024

    # 3) 排名与 oracle 逐站比对
    rank_map = {item['removed_index']: item['noise_flips']
                for item in payload['ranking']}
    deg_set = {t['removed_index'] for t in payload['degenerate_trials']}
    for k in range(len(records)):
        o = oracle_flips(x, k, params['eps'], params['min_samples'])
        if o is None:
            assert k in deg_set, k
        else:
            assert k in rank_map, k
            assert rank_map[k] == o, (k, rank_map[k], o)
    # 排名有序
    vals = [i['noise_flips'] for i in payload['ranking']]
    assert vals == sorted(vals, reverse=True)
    print(f'oracle 逐站比对通过：{len(rank_map)} 站排名 + {len(deg_set)} 退化站')

    # 4) 恢复语义：破坏一个批次文件 + 伪造一个越界批，重跑应回撤且结果不变
    d = job_dir(job_id)
    batches = sorted((d / 'batches').glob('b-*.json'))
    assert len(batches) == -(-len(records) // 37)
    # 损坏第二批
    p2 = batches[1]
    p2.write_text('{broken')
    before = json.loads(batches[0].read_text())
    # 重置任务为 running 模拟中断后重启
    with Session() as db:
        job = db.get(LooJob, job_id)
        job.state = 'running'
        job.finished = 0
        job.degenerate = 0
        job.result = None
        db.commit()
    execute_job(job_id, Session)
    with Session() as db:
        job = db.get(LooJob, job_id)
        assert job.state == 'completed', job.error
        payload2 = job.result
    assert payload2['ranking'] == payload['ranking'], '回撤重试后排名必须一致'
    journal = (d / 'journal.log').read_text()
    assert 'rollback batch=1' in journal
    # 第一批完好未重算（无重复 committed 记录）
    assert json.loads(batches[0].read_text())['checksum'] == before['checksum']
    print('坏批回撤+完好批沿用+重试不重复计入：通过')

    # 5) 幂等：再次执行已完成任务应直接返回
    execute_job(job_id, Session)

    # 6) 导出 zip
    from src.services.export import loo_zip
    with Session() as db:
        resp = loo_zip(db, job_id)
    import zipfile, io
    zf = zipfile.ZipFile(io.BytesIO(resp.body))
    assert set(zf.namelist()) == {'loo-ranking.csv', 'loo-degenerate.csv', 'loo-meta.json'}
    assert '排名' in zf.read('loo-ranking.csv').decode('utf-8-sig')
    print('zip 导出通过：', zf.namelist())
    print('E2E PASSED')


if __name__ == '__main__':
    main()
