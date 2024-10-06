#!/bin/bash

python src/tabular.py --device='fingerprick' --features='all' --n_jobs=16
python src/tabular.py --device='eversense' --features='all' --n_jobs=16
python src/tabular.py --device='fsl' --features='all' --n_jobs=16

python src/tabular.py --device='fingerprick' --features='morning' --n_jobs=16
python src/tabular.py --device='eversense' --features='morning' --n_jobs=16
python src/tabular.py --device='fsl' --features='morning' --n_jobs=16

python src/tabular.py --device='fingerprick' --features='afternoon' --n_jobs=16
python src/tabular.py --device='eversense' --features='afternoon' --n_jobs=16
python src/tabular.py --device='fsl' --features='afternoon' --n_jobs=16

python src/tabular.py --device='fingerprick' --features='evening' --n_jobs=16
python src/tabular.py --device='eversense' --features='evening' --n_jobs=16
python src/tabular.py --device='fsl' --features='evening' --n_jobs=16

python src/tabular.py --device='fingerprick' --features='night' --n_jobs=16
python src/tabular.py --device='eversense' --features='night' --n_jobs=16
python src/tabular.py --device='fsl' --features='night' --n_jobs=16

python src/tabular.py --device='fingerprick' --features='full' --n_jobs=16
python src/tabular.py --device='eversense' --features='full' --n_jobs=16
python src/tabular.py --device='fsl' --features='full' --n_jobs=16

exit