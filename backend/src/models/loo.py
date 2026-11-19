from sqlalchemy import Column,Integer,String,JSON,Text,ForeignKey
from ..config.database import Base
from ..config.settings import CONFIG
class LooJob(Base):
    """逐站剔除影响试算任务：基于一个已完成（冻结）的密度任务。"""
    __tablename__=CONFIG['loo_table']
    id=Column(Integer,primary_key=True)
    run_id=Column(Integer,ForeignKey(CONFIG['run_table']+'.id'),nullable=False)
    state=Column(String(20),default='queued')  # queued/running/completed/failed
    batch_size=Column(Integer,nullable=False)
    total=Column(Integer,nullable=False)       # 剔除试算总数（=冻结样本数）
    finished=Column(Integer,default=0)         # 已完成试算数（含退化）
    degenerate=Column(Integer,default=0)       # 尺度退化试算数
    result=Column(JSON,nullable=True)          # 校验通过后才写入并发布
    error=Column(Text,default='')
    created_at=Column(String(40),nullable=False)
    updated_at=Column(String(40),nullable=False)
