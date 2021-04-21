##  SRGAN for enhancing skin lesions 

### Create initial training data lake

####  Store and organize data:
```sh
    cd mllib/scripts
    python3 store_data.py     # download and extract data
    mv ./data ..
    mkdir ../organized_data
    python3 organize_data.py     # organize data into benign and malignant folders
    cd ../mlutils/scripts
    mkdir ../../benchmark_dataset
    python3 create_benchmark_dataset.py     # creates dataset for evaluating classifier
    python3 create_enhancer_dataset.py      # creates dataset for SRGAN
    
```

#### Train SRGAN

```sh
    cd mllib/mlutils/enhancer_srgan
    python3 train.py
```

#### After training create an enhanced dataset for evaluation

```sh
    cd mllib/mlutils/enhancer_srgan
    python3 create_enhanced_dataset.py    # use SRGAN for creating dataset in folder benchmark_dataset
    cd ../predictor_future
    python3 model_base.py   # evaluates specified data (unfortunately, need to specify the name of folder in code)
    
```