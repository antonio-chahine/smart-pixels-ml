## Directory
```bash
SMART-PIXELX-ML/
├── datasets/
│   ├── labels/
|       ├── labels parquet files
│   └── recon3D/
|       ├── recon3D parquet files
├── test_train_model.ipynb
├── tfrecords_3e778b82/
|   ├── tfrecords_test_3e778b82/
|   ├── tfrecords_train_3e778b82/
|   └── tfrecords_validation_3e778b82/
├── checkpoints_3e778b82/
|   └── weights.hdf5
├── src/
│   ├── model.py
│   └── utils.py
└── README.md
```

## Define the path
```python
# paths
data_directory_path = "./datasets/recon3D/"
labels_directory_path = "./datasets/labels/"

# create tf records directory
# stamp = '%08x' % random.randrange(16**8)
stamp = "3e778b82" # Use same stamp over the code
output_directory = Path(f"./tfrecords_{stamp}").resolve()
os.makedirs(output_directory, exist_ok=True)
tfrecords_dir_train = Path(output_directory, f"tfrecords_train_{stamp}").resolve()
tfrecords_dir_validation = Path(output_directory, f"tfrecords_validation_{stamp}").resolve()
tfrecords_dir_test = Path(output_directory, f"tfrecords_test_{stamp}").resolve()
```

## tfrecords_3e778b82
Generate train, validation and test tfrecords file only **once** with the `OptimizedGenerator()`.


Generate tfrecords
```python
# Trainning Generator
start_time = time.time()
training_generator = OptimizedDataGenerator(
    data_directory_path = data_directory_path,
    labels_directory_path = labels_directory_path,
    is_directory_recursive = False,
    file_type = "parquet",
    data_format = "3D",
    batch_size = batch_size,
    file_count = train_file_size,
    to_standardize= True,
    include_y_local= False,
    labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
    input_shape = (2,13,21), # (20,13,21),
    transpose = (0,2,3,1),
    save=True,
    use_time_stamps = [0,19],
    tfrecords_dir = tfrecords_dir_train,
)
print(f"Training data generator created in {time.time() - start_time} seconds.")

# Same for the validation generator.
```

Loading the tfrecords everytime, so you can save storage by avoinding generate tfrecords everytime.

```python
training_generator = OptimizedDataGenerator(
load_from_tfrecords_dir = f"/home/hep/hl2822/smart-pixels-ml/tfrecords_{stamp}/tfrecords_train_{stamp}/",
shuffle = True,
seed = 13,
quantize = True
)

# Same for the validation generator.
```

## Checkpoints_3e778b82 folders
This can save model for every epoch and can resume training from the latest epoch even if you interupt the code running or SSH disconnect. In this code, I stop training after epoch 117, if you wish to continue, you can run the following code and remember to change your directory.

```python
# Find the latest checkpoint file
import glob

checkpoint_directory = Path(f"./checkpoints_{stamp}").resolve()
checkpoint_files = sorted(glob.glob(str(checkpoint_directory / "weights.*.hdf5")))
checkpoint_files_directory = "/home/hep/hl2822/smart-pixels-ml/checkpoints_3e778b82/weights.117-t-1220.46-v-1137.35.hdf5"

checkpoint_files = sorted(glob.glob(str(checkpoint_files_directory)))

if checkpoint_files:
    latest_checkpoint = checkpoint_files[-1]
    print(f"Loading weights from: {latest_checkpoint}")
    model.load_weights(latest_checkpoint)
    
    # Extract epoch number from filename to resume training
    import re
    match = re.search(r'weights\.(\d+)-', latest_checkpoint)
    if match:
        initial_epoch = int(match.group(1)) + 1
        print(f"Resuming from epoch {initial_epoch}")
    else:
        initial_epoch = 0
else:
    print("No checkpoint files found, starting from scratch")
    initial_epoch = 0

# Now train with initial_epoch parameter
history = model.fit(x=training_generator,
                    validation_data=validation_generator,
                    callbacks=[es, mcp, csv_logger],
                    epochs=epochs,
                    initial_epoch=initial_epoch,  # Add this parameter
                    shuffle=False,
                    verbose=1)
```

## Model evaluation
Check the ipynb file.
