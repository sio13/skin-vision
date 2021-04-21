from glob import glob

from tqdm import tqdm

from .config import cfg as CFG
from .model_base import get_model
from .predictor_base import *

random.seed(a=42)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)

GCS_PATH = f'../input//melanoma-{CFG["read_size"]}x{CFG["read_size"]}'
files_train = np.sort(np.array(tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')))
files_test = np.sort(np.array(tf.io.gfile.glob(GCS_PATH + '/test*.tfrec')))

malig_files = sorted(glob(f'../input/malignant-v2-{CFG["read_size"]}x{CFG["read_size"]}/*.tfrec'))
malig_files = malig_files[15:]

files_train = np.concatenate([files_train, malig_files])


def read_labeled_tfrecord(example):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'patient_id': tf.io.FixedLenFeature([], tf.int64),
        'sex': tf.io.FixedLenFeature([], tf.int64),
        'age_approx': tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
        'diagnosis': tf.io.FixedLenFeature([], tf.int64),
        'target': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['target'], example['image_name']


def get_dataset(files, cfg, augment=False, shuffle=False, labeled=True):
    AUTO = tf.data.experimental.AUTOTUNE
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(1024 * 8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    if labeled:
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else:
        ds = ds.map(lambda example: read_unlabeled_tfrecord(example),
                    num_parallel_calls=AUTO)
    ds = ds.map(lambda img, label, fn: (prepare_image(img, augment=augment, cfg=cfg), label, fn),
                num_parallel_calls=AUTO)
    ds = ds.batch(cfg['batch_size'])
    ds = ds.prefetch(AUTO)
    return ds


def get_opt_loss_fn():
    optimizer = tf.keras.optimizers.Adam(learning_rate=CFG["LR"])
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return optimizer, loss_object


def get_train_fn():
    @tf.function
    def train_step(images, labels, model, optimizer, loss_object):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions[:, 0])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    return train_step


def get_test_fn():
    @tf.function
    def test_step(images, labels, model, loss_object):
        predictions = model(images, training=False)
        return loss_object(labels, predictions[:, 0])

    return test_step


def get_pred_fn():
    @tf.function
    def pred_step(images, model):
        predictions = model(images, training=False)
        return predictions

    return pred_step


fold_cv_scores = []
submission_scores = []

folds = KFold(n_splits=5, shuffle=True, random_state=42)
fold_num = 0
for tr_idx, va_idx in folds.split(files_train):
    print(f"Starting fold: {fold_num}")
    no_imp = 0
    CFG['batch_size'] = 32
    checkpoint_filepath = f"checkpoint_{fold_num}.h5"

    files_train_tr = files_train[tr_idx]
    files_train_va = files_train[va_idx]

    ds_train = get_dataset(files_train_tr, CFG, augment=True, shuffle=True)
    ds_val = get_dataset(files_train_va, CFG, augment=False, shuffle=False)

    optimizer, loss_object = get_opt_loss_fn()
    model = get_model(CFG)
    train_fn = get_train_fn()
    test_fn = get_test_fn()
    pred_fn = get_pred_fn()

    bestLoss = float("inf")
    for e in range(CFG["epochs"]):
        trainLoss = 0
        tk0 = tqdm(ds_train)
        for idx, (x, y, _) in enumerate(tk0):
            loss = train_fn(x, y, model, optimizer, loss_object)
            trainLoss += loss.numpy()
            tk0.set_postfix(loss=trainLoss / (idx + 1))

        testLoss = 0
        tk0 = tqdm(ds_val)
        for idx, (x, y, _) in enumerate(tk0):
            loss = test_fn(x, y, model, loss_object)
            testLoss += loss.numpy()
            tk0.set_postfix(loss=testLoss / (idx + 1))

        testLoss /= idx
        if testLoss < bestLoss:
            no_imp = 0
            bestLoss = testLoss
            model.save_weights(checkpoint_filepath)
        else:
            no_imp += 1
            if no_imp > CFG["es_patience"]:
                print("Early stopping..")
                break

    model.load_weights(checkpoint_filepath)

    CFG['batch_size'] = 256
    ds_valAug = get_dataset(files_train_va, CFG, augment=True)
    ds_testAug = get_dataset(files_test, CFG, augment=True, labeled=False)
    for t in range(CFG['tta_steps']):
        for idx, (x, y, fn) in enumerate(ds_valAug):
            predictions = pred_fn(x, model)
            for j in range(predictions.shape[0]):
                fold_cv_scores.append([fold_num,
                                       fn[j].numpy().decode("utf-8"),
                                       predictions[j, 0].numpy(),
                                       y[j].numpy()])

        for idx, (x, y, fn) in enumerate(ds_testAug):
            predictions = pred_fn(x, model)
            for j in range(predictions.shape[0]):
                submission_scores.append([fold_num,
                                          fn[j].numpy().decode("utf-8"),
                                          predictions[j, 0].numpy()])

    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()

    fold_num += 1

df_fold = pd.DataFrame(fold_cv_scores, columns=["Fold", "Filename", "Pred", "Label"])
df_sub = pd.DataFrame(submission_scores, columns=["Fold", "Filename", "Pred"])

df_fold = df_fold.groupby(["Filename"]).mean().reset_index()
print("CV ROCAUC: ")
print(roc_auc_score(df_fold["Label"], df_fold["Pred"]))

df_sub = df_sub.groupby(["Filename"]).mean().reset_index()
df_sub = df_sub[["Filename", "Pred"]]
df_sub.columns = ["image_name", "target"]
df_sub.to_csv("submission.csv", index=False)
