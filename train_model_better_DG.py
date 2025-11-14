import os
import pathlib
import random
import json
import submitit
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

from OptimizedDataGenerator_v2 import OptimizedDataGenerator
from models import CreateModel
from loss import custom_loss


# --------------------------------------------------------
#  HYPERPARAMETERS
# --------------------------------------------------------
epochs = 200
batch_size = 5000
learning_rate = 0.001
early_stopping_patience = 50

# NEW SHAPE: 20 timestamps, 13x21 pixel plane → after transpose becomes (13,21,20)
shape = (13, 21, 2)

stamp = "2ts5000"
base_path = f"/ceph/submit/data/user/a/anton100/tfrecords_{stamp}"

tfrecords_train = f"{base_path}/train"
tfrecords_val   = f"{base_path}/val"
tfrecords_test  = f"{base_path}/test"


# --------------------------------------------------------
#  LOAD DATA WITH OPTIMIZED DATA GENERATOR V2
# --------------------------------------------------------
training_generator = OptimizedDataGenerator(
    load_from_tfrecords_dir = tfrecords_train,
    shuffle = True,
    seed = 13,
    quantize = False
)

validation_generator = OptimizedDataGenerator(
    load_from_tfrecords_dir = tfrecords_val,
    shuffle = True,
    seed = 13,
    quantize = False
)


# --------------------------------------------------------
#  BUILD MODEL
# --------------------------------------------------------
model = CreateModel(shape=shape, n_filters=5, pool_size=3)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss=custom_loss)
model.summary()


# --------------------------------------------------------
#  CHECKPOINTS + LOGGING
# --------------------------------------------------------
checkpoint_directory = Path(f"./checkpoints_{stamp}")
checkpoint_directory.mkdir(parents=True, exist_ok=True)

checkpoint_filepath = checkpoint_directory / "weights.{epoch:03d}-t{loss:.3f}-v{val_loss:.3f}.hdf5"

mcp = ModelCheckpoint(
    filepath = str(checkpoint_filepath),
    save_weights_only = True,
    monitor = "val_loss",
    save_best_only = False,
)

csvlogger_directory = Path("./csvlogs")
csvlogger_directory.mkdir(parents=True, exist_ok=True)

csv_logger = CSVLogger(str(csvlogger_directory / f"training_log_{stamp}.csv"), append=True)

es = EarlyStopping(patience=early_stopping_patience, restore_best_weights=True)


# --------------------------------------------------------
#  TRAIN
# --------------------------------------------------------
history = model.fit(
    x = training_generator,
    validation_data = validation_generator,
    epochs = epochs,
    shuffle = False,       # internal shuffling already done by generator
    callbacks = [es, mcp, csv_logger],
    verbose = 1,
)

print("✓ Training complete.")
