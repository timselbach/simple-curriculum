# Curriculum Learning with Simple Language for Transformer-Based Language Models

This repository contains the code to run a variety of curriculum learning approaches on two simple language datasets.

## About

Curriculum learning has successfully been applied to different machine learning areas. However, there have been some controversial results regarding language curriculum learning. To see whether simple language datasets help to create a curriculum and to learn more about the properties that need to be fulfilled to apply curriculum learning successfully.

## Installation

We used a virtual anaconda environment to run our experiments. To create one, you can use the following command:

```console
conda create -n simple-cl python=3.10.16
```

To install all required packages, run:

```console
conda activate simple-cl
conda config --add channels conda-forge
conda config --add channels pytorch
conda config --add channels nvidia
conda install --file requirements.txt
```

Also, those following commands need to be executed in your active environment:

```console 
python -m spacy download de_core_news_lg
python -m spacy download en_core_web_trf
```
 

## Before getting started

To run the experiments, you have to set some variables in the ```code.curriculum.config.py``` file. Set ```MODEL_SAVE_BASE_PATH``` and ```TOKENIZER_SAVE_BASE_PATH```to the directory of your system where the model and tokenizer should get saved to e.g. ```MODEL_SAVE_BASE_PATH=/home/alice/simple_curriculum/results/model```.

In the same config file you need to set the ```DATASETS_BASE_PATH``` to you datasets directory, e.g. ```DATASETS_BASE_PATH=/home/alice/simple_curriculum/datasets```

Additionally, the paths in ```code.data.all_datasets.py``` need to be set to the correct location e.g. ```/home/alice/simple_curriculum/datasets/SimpleGerman/Datasets/```

(Optional: Set the dataset paths in ```code.data.datasets.py``` if you want to work with respective functions)

## Usage

Here you find all neccessary information to run our experiments.

### Vocabulary analysis

All the needed files and functions for vocabulary or metric analysis are located in the package ```code.vocabulary```. As a demonstration of the analysis you can run the ```main.py``` file to get insights and results about the datasets' properties.

### Pre-training

A specific training scenario can be defined in ```code.curriculum/config.py```. If you want to train a new tokenizer from scratch, you have to set TRAIN_TOKENIZER=True. Otherwise, you can set the respective tokenizer paths to load the pretrained tokenizers from in the ```code.curriculum/main.py```.

The training strategy can be set in the ```TRAINING_STRATEGY``` variable. The type key can be set to incremental, competence, or sequential. In the case of competence training, be sure to set the ```DIFFICULTY_METRIC``` variable.

### Fine-tuning & Embedding properties
Furthermore, we provide the code for fine-tuning our models in the package ```code.finetune``` for the follwing tasks:
- IMDB sentiment classification
- XNLI for German
- WordSim-353
- Schm-280

Here you need to set the corresponding model path to the directory you saved the specific model. You find an example definition of that in the code.

### Evaluate levels seperately
We provide a script that lets you evaluate the models' performance on the different language levels seperately. The script can be found at ```curriculum/evaluate_by_level.py```. You have to consider that you set the seed to the same value the corresponding model was trained on.


## Tools/Libraries
All packages that are used for this project can be found in the ```requirements.txt```.
