import sys
import pathlib
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, Callback
import random
from pathlib import Path
import time
import argparse
import json
import submitit
import shutil
from OptimizedDataGenerator import OptimizedDataGenerator
from loss import custom_loss
from models import CreateModel
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
epochs = 200
batch_size = 128
val_batch_size = 128
train_file_size = 10
val_file_size = 4
n_filters = 5
pool_size = 3
learning_rate = 0.001
early_stopping_patience = 15
shape=(13,21,20)

# paths
data_directory_path = "/ceph/submit/data/user/a/anton100/datasets/recon3D/" # "/net/scratch/badea/dataset8/unflipped/"
labels_directory_path = "/ceph/submit/data/user/a/anton100/datasets/labels/" # "/net/scratch/badea/dataset8/unflipped/"

# create tf records directory
# stamp = '%08x' % random.randrange(16**8)
stamp = "d7414f9d"
output_directory = Path(f"./tfrecords_{stamp}").resolve()
os.makedirs(output_directory, exist_ok=True)
tfrecords_dir_train = Path(output_directory, f"tfrecords_train_{stamp}").resolve()
tfrecords_dir_validation = Path(output_directory, f"tfrecords_validation_{stamp}").resolve()
tfrecords_dir_test = Path(output_directory, f"tfrecords_test_{stamp}").resolve()


training_generator = OptimizedDataGenerator(
load_from_tfrecords_dir = f"/ceph/submit/data/user/h/haoyun22/smart_pixels_data/tfrecords_{stamp}/tfrecords_train_{stamp}/",
shuffle = True,
seed = 13,
quantize = True
)

validation_generator = OptimizedDataGenerator(
load_from_tfrecords_dir = f"/ceph/submit/data/user/h/haoyun22/smart_pixels_data/tfrecords_{stamp}/tfrecords_validation_{stamp}/",
shuffle = True,
seed = 13,
quantize = True
)

test_generator = OptimizedDataGenerator(
load_from_tfrecords_dir= f"/ceph/submit/data/user/h/haoyun22/smart_pixels_data/tfrecords_{stamp}/tfrecords_test_{stamp}/",
shuffle = True,
seed = 13,
quantize = True
)


model = CreateModel(shape=shape, n_filters=5, pool_size=3)
model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss)
model.summary()


checkpoint_directory = Path(f"./checkpoints_{stamp}").resolve()
checkpoint_directory.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
checkpoint_filepath = Path(checkpoint_directory, 'weights.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.hdf5').resolve()

es = EarlyStopping(patience=early_stopping_patience, restore_best_weights=True)

mcp = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    save_best_only=False,
)

csvlogger_directory = Path("./csvlogs").resolve()
csvlogger_directory.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

csv_logger = CSVLogger(Path(csvlogger_directory, f'training_log_{stamp}.csv').resolve(), append=True)


history = model.fit(x=training_generator,
                    validation_data=validation_generator,
                    callbacks=[es, mcp, csv_logger],
                    epochs=epochs,
                    shuffle=False, # shuffling now occurs within the data-loader
                    verbose=1)

minval = 1e-9

# --- Select the most recent weights file ---
weights_path = sorted(checkpoint_directory.glob('weights.*.hdf5'))[-1]
print(f"Using weights: {weights_path.name}")

# --- Create output directory for this stamp ---
outfile_directory = Path(f"./outfile_{stamp}").resolve()
outfile_directory.mkdir(parents=True, exist_ok=True)  # make folder if needed

# --- Name the CSV file after the weights file, with evaluation prefix ---
outfile_name = f"evaluation_results_{weights_path.stem}.csv"
outfile_path = outfile_directory / outfile_name



n_filters = 5
pool_size = 3
shape=(13,21,20)


# --- Handle job ID substitution (optional) ---
try:
    job_env = submitit.JobEnvironment()
    outfile_path = Path(str(outfile_path).replace("%j", str(job_env.job_id)))
except:
    outfile_path = Path(str(outfile_path).replace("%j", "%08x" % random.randrange(16**8)))

# --- Make sure output directory exists ---
os.makedirs(outfile_directory, exist_ok=True)
print(f"Output file will be saved to: {outfile_path}")

# ------------------------------------------------------------
#  Model evaluation
# ------------------------------------------------------------

model = CreateModel(shape=shape, n_filters=n_filters, pool_size=pool_size)
model.load_weights(weights_path)
p_test = model.predict(test_generator)

# Collect all true labels
complete_truth = None
for _, y in test_generator:
    complete_truth = y if complete_truth is None else np.concatenate((complete_truth, y), axis=0)

# Build DataFrame
df = pd.DataFrame(
    p_test,
    columns=['x','M11','y','M22','cotA','M33','cotB','M44','M21','M31','M32','M41','M42','M43']
)

# Add true labels
df['xtrue'], df['ytrue'], df['cotAtrue'], df['cotBtrue'] = complete_truth.T

# Clamp diagonal matrix elements to >= minval
for m in ['M11','M22','M33','M44']:
    df[m] = minval + tf.math.maximum(df[m], 0)

# Compute residuals
df['residual_x']  = df['xtrue']    - df['x']
df['residual_y']  = df['ytrue']    - df['y']
df['residual_A']  = df['cotAtrue'] - df['cotA']
df['residual_B']  = df['cotBtrue'] - df['cotB']

# Save results
df.to_csv(outfile_path, header=True, index=False)
print(f"✅ Evaluation results saved to: {outfile_path}")