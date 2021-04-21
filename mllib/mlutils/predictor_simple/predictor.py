import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from model_base import get_model, load_model, eval_model

tf.random.set_seed(7)
np.random.seed(7)
random.seed(7)

# generate_csv("../organized_data", {"benign": 0, "malignant": 1})
df = pd.read_csv("../../mlutils/predictor_simple/data/organized_data.csv")
df_train, df_test = train_test_split(df, test_size=0.1, random_state=False)
train_ds = tf.data.Dataset.from_tensor_slices((df_train["filepath"], df_train["label"]))
test_df = tf.data.Dataset.from_tensor_slices((df_test["filepath"], df_test["label"]))
model = get_model()
model = load_model(model,
                   "../../mlutils/predictor_simple/weights/benign-vs-malignant_64_rmsprop_0.100.h5")


eval_model(model, test_df)
