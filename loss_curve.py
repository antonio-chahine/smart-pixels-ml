import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Plot training and validation loss against epochs
csv_logger_path = "/work/submit/anton100/msci-project/smart-pixels-ml/csvlogs/training_log_2ts5000.csv"

data = pd.read_csv(csv_logger_path)
plt.figure(figsize=(10, 6))
plt.plot(data['epoch'], data['loss'], label='Training Loss')
plt.plot(data['epoch'], data['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid()
plt.savefig("./figures/loss_curve_2ts5000.png")
plt.show()