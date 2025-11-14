# --- TensorFlow & GPU info ---
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")

    # CUDA + GPU detection
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs detected: {[gpu.name for gpu in gpus]}")
        print(f"CUDA available to TensorFlow: {tf.test.is_gpu_available()}")
    else:
        print("No GPU detected by TensorFlow.")

except ImportError:
    print("TensorFlow not installed.")

# --- QKeras version ---
try:
    import qkeras
    print(f"QKeras version: {qkeras.__version__}")
except ImportError:
    print("QKeras not installed.")
