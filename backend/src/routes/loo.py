from fastapi import APIRouter,Depends,Query
from pydantic import BaseModel
from ..config.database import db_session
from ..services.auth import current_user,writer
from ..services.export import loo_zip
from ..controllers.loo import LooController as C
router=APIRouter()
class LooInput(BaseModel):
    run_id:int
    batch_size:int|None=None
@router.get('/loo-jobs')
def jobs(run_id:int|None=Query(None),db=Depends(db_session),u=Depends(current_user)):
    return C.list(db,run_id)
@router.post('/loo-jobs')
def create(data:LooInput,db=Depends(db_session),u=Depends(writer)):
    return C.create(db,u,data.run_id,data.batch_size)
@router.get('/loo-jobs/{job_id}')
def detail(job_id:int,db=Depends(db_session),u=Depends(current_user)):
    return C.get(db,job_id)
@router.get('/loo-jobs/{job_id}/export')
def export(job_id:int,db=Depends(db_session),u=Depends(current_user)):
    return loo_zip(db,job_id)
