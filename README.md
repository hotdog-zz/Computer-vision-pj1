# Computer-vision-pj1

This is an implementation of three-layer mlp model without pytorch or tensorflow trained on fashion-mnist dataset.

## Setup

``` sh
git clone 
cd ./Computer-vision-pj1

conda create -n pj1 python=3.10
conda activate pj1
pip install .
```

## Usage

First run the following command the download the dataset and create folders. Then you can put the weight file in the weight folder.

``` sh
python main.py --train
```

To train the model with parameter search:

``` sh
python main.py --train
```

To test the model in the default path "./weight/params_best.pkl":

``` sh
python main.py --test
```