"""Run test modules in clean interpreters to isolate OpenCV/Cairo state."""
from pathlib import Path
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
TEST_MODULES=sorted((ROOT/'tests').glob('test_*.py'))


def main() -> int:
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]='1'
    failed=[]
    for file in TEST_MODULES:
        print(f'\n=== {file.name} ===',flush=True)
        try:
            proc=subprocess.run([sys.executable,'-m','pytest','-q',str(file)],
                                cwd=ROOT,env=env,timeout=100,check=False)
            if proc.returncode:
                failed.append(file.name)
        except subprocess.TimeoutExpired:
            print(f'TIMED OUT: {file.name}',flush=True)
            failed.append(file.name)
    if failed:
        print('\nFAILED:',', '.join(failed),flush=True)
        return 1
    print(f'\nAll {len(TEST_MODULES)} test modules passed in isolated Python processes.',flush=True)
    return 0


if __name__=='__main__':
    sys.exit(main())
