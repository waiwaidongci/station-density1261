from ..services import loo
class LooController:
    list=staticmethod(loo.list_jobs)
    create=staticmethod(loo.create_job)
    get=staticmethod(loo.get_job)
