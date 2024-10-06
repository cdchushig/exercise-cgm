Characterizing the impact of physical activity
====

## Clone and download files of repository

To dowload the source code, you can clone it from the Github repository.
```console
git clone git@github.com:cdchushig/exercise-cgm.git
```

## Installation and configuration

Before installing libraries, ensuring that a Python virtual environment is activated (using conda o virtualenv). To install Python libraries run: 

```console
pip install -r requirements.txt 
```

## Execute and run experiments

### Compute stats for glucose measurements

```console
python src/stats.py --device='fsl'
```

### Train models for identifying physical exercise

```console
python src/tabular.py --device='fsl' --features='all' --n_jobs=4
```
