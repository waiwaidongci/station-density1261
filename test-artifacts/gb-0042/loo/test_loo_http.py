"""HTTP 级：TestClient + 鉴权 + LOO 路由 + 权限 + 校验后发布。"""
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, '/tmp/pylibs')
ROOT = Path(__file__).resolve().parents[3] / 'backend'
sys.path.insert(0, str(ROOT))

ART = Path('/tmp/loo-http-artifacts')
DB = Path('/tmp/loo-http.db')
shutil.rmtree(ART, ignore_errors=True)
DB.unlink(missing_ok=True)
os.environ['ARTIFACT_DIR'] = str(ART)
os.environ['DATABASE_URL'] = f'sqlite:///{DB}'
os.environ['LOO_BATCH_SIZE'] = '25'
os.environ['SEED_DEMO'] = 'false'

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.config.database import Base, engine, Session  # noqa: E402
from src.models.run import Run  # noqa: E402
from src.models.user import User  # noqa: E402
from src.services.algorithm import calculate  # noqa: E402
from src.services.loo import execute_job  # noqa: E402
from src.utils.security import hash_password  # noqa: E402
from src.utils.serialization import now  # noqa: E402
from src.main import app  # noqa: E402
import src.services.runs as runs_mod  # noqa: E402
import src.services.loo as loo_mod  # noqa: E402

FEATURES = ['temperature', 'humidity', 'particles']
Base.metadata.create_all(engine)


class FakeRedis:
    def rpush(self, *a, **k):
        pass

    def ping(self):
        return True


runs_mod.queue = FakeRedis()
loo_mod.queue = runs_mod.queue


def seed():
    rng = np.random.default_rng(21)
    x = np.vstack([
        rng.normal([19, 72, 22], [0.8, 2.5, 3], (40, 3)),
        rng.normal([26, 45, 48], [0.8, 2.5, 3], (40, 3)),
        rng.uniform([12, 12, 4], [42, 92, 140], (14, 3)),
    ])
    records = [{'reference': f'HL-H-{i:04d}',
                'values': {f: float(v) for f, v in zip(FEATURES, row)}}
               for i, row in enumerate(x)]
    with Session() as db:
        db.add_all([
            User(email='admin@station-density.demo',
                 password_hash=hash_password('pw'), role='admin'),
            User(email='analyst@station-density.demo',
                 password_hash=hash_password('pw'), role='analyst'),
            User(email='viewer@station-density.demo',
                 password_hash=hash_password('pw'), role='viewer'),
        ])
        run = Run(source_id=1, name='HTTP 冻结研究', state='completed',
                  parameters={'eps': 0.55, 'min_samples': 5},
                  snapshot={'source': {'filename': 'http.csv'},
                            'records': records},
                  result=calculate(records, {'eps': 0.55, 'min_samples': 5}),
                  created_at=now())
        db.add(run)
        # 未完成任务，用于 409 校验
        run2 = Run(source_id=1, name='未完成', state='queued',
                   parameters={'eps': 0.55, 'min_samples': 5},
                   snapshot={'source': {'filename': 'x.csv'},
                            'records': records},
                   created_at=now())
        db.add(run2)
        db.commit()
        return run.id, run2.id


def login(client, email):
    r = client.post('/auth/login', json={'email': email, 'password': 'pw'})
    assert r.status_code == 200, r.text
    return r.json()['token']


def main():
    run_id, run2_id = seed()
    with TestClient(app) as client:
        anon = client.get('/loo-jobs')
        assert anon.status_code == 401, anon.text
        viewer = login(client, 'viewer@station-density.demo')
        analyst = login(client, 'analyst@station-density.demo')

        # viewer 只读
        r = client.post('/loo-jobs',
                        json={'run_id': run_id},
                        headers={'Authorization': 'Bearer ' + viewer})
        assert r.status_code == 403, r.text

        # 未完成研究不得发起
        r = client.post('/loo-jobs', json={'run_id': run2_id},
                        headers={'Authorization': 'Bearer ' + analyst})
        assert r.status_code == 409, r.text

        # 非法批次
        r = client.post('/loo-jobs',
                        json={'run_id': run_id, 'batch_size': 99999},
                        headers={'Authorization': 'Bearer ' + analyst})
        assert r.status_code == 422, r.text

        # 正常发起
        r = client.post('/loo-jobs', json={'run_id': run_id},
                        headers={'Authorization': 'Bearer ' + analyst})
        assert r.status_code == 200, r.text
        job = r.json()
        jid = job['id']
        assert job['state'] == 'queued' and job['result'] is None

        # 完成前列表不暴露排名
        r = client.get('/loo-jobs?run_id=' + str(run_id),
                       headers={'Authorization': 'Bearer ' + viewer})
        assert r.json()[0]['result'] is None

        # worker 执行
        execute_job(jid, Session)

        r = client.get(f'/loo-jobs/{jid}',
                       headers={'Authorization': 'Bearer ' + viewer})
        body = r.json()
        assert body['state'] == 'completed' and body['progress'] == 1.0
        payload = body['result']
        assert payload['ranking'], '应产生排名'
        assert payload['records_checksum']
        # 重复发起进行中保护：已完成则允许再查；先验证 404
        assert client.get('/loo-jobs/99999',
                          headers={'Authorization': 'Bearer ' + viewer}
                          ).status_code == 404

        # 导出
        r = client.get(f'/loo-jobs/{jid}/export',
                       headers={'Authorization': 'Bearer ' + viewer})
        assert r.status_code == 200 and r.headers['content-type'] == 'application/zip'
        assert len(r.content) > 200

        # 对同一已完成研究再次发起（无运行中任务）应成功入队
        r = client.post('/loo-jobs', json={'run_id': run_id},
                        headers={'Authorization': 'Bearer ' + analyst})
        assert r.status_code == 200, r.text
        jid2 = r.json()['id']
        # 此时再发起第三次应 409（已有运行中）
        r = client.post('/loo-jobs', json={'run_id': run_id},
                        headers={'Authorization': 'Bearer ' + analyst})
        assert r.status_code == 409, r.text
        execute_job(jid2, Session)

    print('HTTP 鉴权/权限/校验/发布/导出/重复保护：全部通过')


if __name__ == '__main__':
    main()
