import csv,io,zipfile,json
from fastapi import HTTPException
from fastapi.responses import Response
from ..models.note import Note
from ..models.loo import LooJob
from .runs import get_run
def csv_response(result,filename):
    buf=io.StringIO();fields=result['columns'];w=csv.DictWriter(buf,fieldnames=fields);w.writeheader()
    for r in result['rows']:w.writerow({k:r.get(k,'') for k in fields})
    return Response('﻿'+buf.getvalue(),media_type='text/csv; charset=utf-8',headers={'Content-Disposition':'attachment; filename="'+filename+'"'})
def run_csv(db,id):
    r=get_run(db,id)
    if r.state!='completed':raise HTTPException(409,'结果尚未完成')
    return csv_response(r.result,'run-'+str(id)+'.csv')
def note_csv(db,id):
    n=db.get(Note,id)
    if not n:raise HTTPException(404,'未找到说明')
    if n.state!='approved':raise HTTPException(403,'仅批准后的业务说明可导出')
    return csv_response(n.frozen_result,'approved-'+str(id)+'.csv')
def loo_zip(db,job_id):
    job=db.get(LooJob,job_id)
    if not job:raise HTTPException(404,'未找到逐站剔除试算任务')
    if job.state!='completed' or not job.result:raise HTTPException(409,'试算尚未完成或未通过校验，无排名可导出')
    result=job.result
    rank_buf=io.StringIO()
    rw=csv.writer(rank_buf)
    rw.writerow(['排名','样本编号','剔除站序号','其他站噪声身份翻转数','试算前其他站噪声数','试算后其他站噪声数'])
    for rank,item in enumerate(result['ranking'],1):
        rw.writerow([rank,item['reference'],item['removed_index'],item['noise_flips'],item['noise_before'],item['noise_after']])
    deg_buf=io.StringIO()
    dw=csv.writer(deg_buf)
    dw.writerow(['样本编号','剔除站序号','单列原因'])
    for item in result['degenerate_trials']:
        dw.writerow([item['reference'],item['removed_index'],item['reason']])
    meta_buf=io.StringIO()
    json.dump({k:v for k,v in result.items() if k not in ('ranking','degenerate_trials')},meta_buf,ensure_ascii=False,indent=2)
    zbuf=io.BytesIO()
    with zipfile.ZipFile(zbuf,'w',zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('loo-ranking.csv','﻿'+rank_buf.getvalue())
        zf.writestr('loo-degenerate.csv','﻿'+deg_buf.getvalue())
        zf.writestr('loo-meta.json',meta_buf.getvalue())
    return Response(zbuf.getvalue(),media_type='application/zip',headers={'Content-Disposition':'attachment; filename="loo-job-'+str(job_id)+'.zip"'})
