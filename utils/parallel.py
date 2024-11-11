from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed

class ProgressParallel(Parallel):
    def __init__(self, n_total_tasks=None, **kwargs):
        super().__init__(**kwargs)
        self.n_total_tasks = n_total_tasks
    
    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)
    
    def print_progress(self):
        if self.n_total_tasks:
            self._pbar.total = self.n_total_tasks
        else:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def mod_parallel(func, workers, inputs, **kwargs):
    data = ProgressParallel(
            n_jobs=workers, 
            n_total_tasks=len(inputs)
        )(delayed(partial(func, **kwargs))(input) for input in inputs)
    return data