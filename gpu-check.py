import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ TensorFlow sees {len(gpus)} GPU(s):")
    for gpu in gpus:
        print("  •", gpu)
else:
    print("No GPUs detected by TensorFlow.")
