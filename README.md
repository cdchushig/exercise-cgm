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

If have any issue with skrebate, please install the following modified version:
```console
pip install git+https://github.com/cdchushig/scikit-rebate.git@1efbe530a46835c86f2e50f17342541a3085be9c
```

## Download data and copy to project

A further description of the original datasets is available in the paper: "Severe Hypoglycemia in Older Adults with Type 1 Diabetes: A Study to Identify Factors Associated with the Occurrence of Severe Hypoglycemia in Older Adults with T1D".

Raw data and preprocessed data have been uploaded in Onedrive folder. The link for both raw and preprocessed datasets is:

[Link to raw and preprocessed datasets](https://doi.org/10.6084/m9.figshare.25136942.v1)

To replicate results, download datasets of preprocessed folder. Please, after downloading data, you have to put folders and files in data/preprocessed.  

## To obtain results of models using single-modality data

For results using tabular data:
```console
python src/train.py --type_data='unimodal' --type_modality='tabular'
```

For results using time series:
```console
python src/train.py --type_data='unimodal' --type_modality='time_series'
```

For preprocessing time series, run the following command:
```console
python src/preprocessing.py
```

For results using text:
```console
python src/train.py --type_data='unimodal' --type_modality='text'
```

## To obtain results of models using multi-modality data

For results with early fusion:
```console
python src/train.py --type_data='multimodal' --type_fusion='early'
```

For results with late fusion:
```console
python src/train.py --type_data='multimodal' --type_fusion='late'
```
