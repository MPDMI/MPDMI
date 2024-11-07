## MPDMI
The code and dataset for our paper:Meta-Paths aware Dynamic Multi-Interest learning for sequential recommendation
## Dependencies
- Python 3.7
- torch 1.5.0
- numpy 1.15.4
- scipy 1.1.0
- sklearn 0.20.0
- torch_scatter 2.0.5
- netwrokx 2.5
## Usage
### Datasets
You need to download the datasets required for the model via the following links:

Books:http://jmcauley.ucsd.edu/data/amazon

MovieLens-1M:https://grouplens.org/datasets/movielens

Last-FM:https://grouplens.org/datasets/hetrec-2011/
### Generate data
You need to run the file data_{dataset}.py to generate the data format needed for our model.

You need to run the file generate_paths_{dataset}.py to generate metapath. You can set the data set in the file.
### Training and Testing
Then you can run the file main.py to train and test our model. 
