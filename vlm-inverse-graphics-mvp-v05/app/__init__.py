"""Local CPU-friendly graphics application; avoid excessive native thread pools."""
import os
for _var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var,'1')
