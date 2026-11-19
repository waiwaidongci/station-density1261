"""中断恢复专项：残留 tmp、journal 提交但文件缺失、坏批回撤，重试不重复计入。"""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, '/tmp/pylibs')
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'backend'))

ART = Path('/tmp/loo-crash-artifacts')
DB = Path('/tmp/loo-crash.db')
shutil.rmtree(ART, ignore_errors=True)
DB.unlink(missing_ok=True)
os.environ.update(ARTIFACT_DIR=str(ART), DATABASE_URL=f'sqlite:///{DB}',
                  LOO_BATCH_SIZE='20', SEED_DEMO='false')

import numpy as np  # noqa: E402
from src.config.database import Base, engine, Session  # noqa: E402
from src.models import user as _u, source as _s, record as _r, run as _run, note as _n, audit as _a  # noqa: E402,F401
from src.models.run import Run  # noqa: E402
from src.models.loo import LooJob  # noqa: E402
from src.services.algorithm import calculate  # noqa: E402
from src.services.loo import execute_job, job_dir, create_job  # noqa: E402
from src.utils.serialization import now  # noqa: E402
import src.services.runs as runs_mod  # noqa: E402
import src.services.loo as loo_mod  # noqa: E402

FEATURES = ['temperature', 'humidity', 'particles']
Base.metadata.create_all(engine)


class FakeQueue:
    def rpush(self, *a, **k):
        pass


runs_mod.queue = FakeQueue()
loo_mod.queue = runs_mod.queue


class Actor:
    email = 'analyst@station-density.demo'


def main():
    rng = np.random.default_rng(31)
    x = np.vstack([rng.normal(mu, 0.9, (30, 3)) for mu in
                   ([-3, -3, -3], [3, -3, 3], [-3, 3, 3])]
                  + [rng.uniform(-7, 7, (8, 3))])
    records = [{'reference': f'HL-C-{i:04d}',
                'values': {f: float(v) for f, v in zip(FEATURES, row)}}
               for i, row in enumerate(x)]
    params = {'eps': 0.55, 'min_samples': 5}
    with Session() as db:
        run = Run(source_id=1, name='崩溃恢复研究', state='completed',
                  parameters=params,
                  snapshot={'source': {'filename': 'c.csv'}, 'records': records},
                  result=calculate(records, params), created_at=now())
        db.add(run)
        db.commit()
        rid = run.id
        jid = create_job(db, Actor(), rid)['id']

    execute_job(jid, Session)
    d = job_dir(jid)
    with Session() as db:
        good_result = db.get(LooJob, jid).result
    batches = sorted((d / 'batches').glob('b-*.json'))
    assert len(batches) == -(-len(records) // 20)

    # 场景 A：原子写残留 tmp（应被忽略，不影响恢复）；选不会重算的批次名避免与原子写撞名
    (d / 'batches' / 'b-00004.json.tmp').write_text('partial')
    # 场景 B：journal 有提交记录但批文件丢失（应重算该批）
    lost = batches[3]
    lost_payload = json.loads(lost.read_text())
    lost.unlink()
    # 场景 C：另一批内容损坏（应回撤重算）
    bad = batches[2]
    bad.write_text('{corrupt')
    # 场景 D：校验和被篡改
    tampered = batches[1]
    tp = json.loads(tampered.read_text())
    tp['trials'][0]['noise_flips'] = 99999
    tampered.write_text(json.dumps(tp))

    with Session() as db:
        job = db.get(LooJob, jid)
        job.state = 'running'
        job.finished = 0
        job.degenerate = 0
        job.result = None
        db.commit()

    execute_job(jid, Session)
    with Session() as db:
        job = db.get(LooJob, jid)
        assert job.state == 'completed', job.error
        assert job.result['ranking'] == good_result['ranking']
        assert job.finished == len(records)

    journal = (d / 'journal.log').read_text()
    assert journal.count('rollback batch=2') >= 1
    assert journal.count('rollback batch=1') >= 1
    # 重算后的丢批次与原内容一致（不重复计入）
    assert json.loads(lost.read_text())['checksum'] == lost_payload['checksum']
    # 残留 tmp 仍在但未被当作批次
    assert (d / 'batches' / 'b-00004.json.tmp').exists()
    recovered = sorted((d / 'batches').glob('b-*.json'))
    assert len(recovered) == len(batches)
    # 汇总无重复：finished 恰好等于总站数
    print('中断恢复四场景通过：tmp 忽略 / 丢批重算 / 坏批回撤 / 篡改拒认，'
          f'finished={job.finished}/{len(records)}，排名与首次一致')


if __name__ == '__main__':
    main()
