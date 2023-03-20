
#! Variables to play with
epochs = 75
batch_size = 20
dropout_rate = 0.5
validation_split = 0.2
loss = "mae"
#! Variables to play with
val_loss = f"val_{loss}"
#! CONFIG THINGS
import numpy as np
from os.path import join
all_names = [
    "O2b",
    "O2a",
    "Hd",
    "Hc",
    "Hb",
    "O3b",
    "O3a",
    "N2b",
    "Ha",
    "N2a",
    "S2b",
    "S2a",
]
num_spectra = 20
z_min = 0.00370
z_max = 0.14970
all_z_values = np.round(np.linspace(z_min, z_max, 146), 5)
prepend_path = "/uufs/astro.utah.edu/common/home/u6031907/casiglo/final_code/"
training_data_folder = join(prepend_path, "1_data_generation/training_data/data")
checkpoint_folder = join(prepend_path, "2_training/models/best.ckpt")
#! CONFIG THINGS

from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import (
    Dense,
    Activation,
    Dropout,
    Flatten,
    Conv1D,
    BatchNormalization,
    MaxPool1D,
)
from keras.metrics import MeanAbsoluteError, Precision, KLDivergence
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#NOTE: KL Divergence ~~ amount of information lost when model is used to approximate reality
metrics = [MeanAbsoluteError(name="mae", dtype=None), Precision(name="precision", dtype=None), KLDivergence(name="kl")]

class SaveMetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        np.save("models/mae.npy", np.append(np.load("models/mae.npy"), logs["mae"]))
        np.save("models/val_mae.npy", np.append(np.load("models/val_mae.npy"), logs["val_mae"]))

        np.save("models/precision.npy", np.append(np.load("models/precision.npy"), logs["precision"]))
        np.save("models/val_precision.npy", np.append(np.load("models/val_precision.npy"), logs["val_precision"]))

        np.save("models/kl.npy", np.append(np.load("models/kl.npy"), logs["kl"]))
        np.save("models/val_kl.npy", np.append(np.load("models/val_kl.npy"), logs["val_kl"]))

metric_callback = SaveMetricsCallback()

early_stopping_callback = EarlyStopping(
    monitor=val_loss,
    min_delta=0.001,
    patience=epochs//2,
    verbose=False,
    mode="min",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_folder,
    save_weights_only=True,
    monitor=val_loss,
    mode="min",
    verbose=0,
    save_best_only=False,
    save_freq=1,
)

def create_smaller_model(metrics: list) -> Sequential:

    model = Sequential()
    model.add(Conv1D(input_shape=(4563, 1), filters=13, kernel_size=13))  # 4563 -> (4563 - 13 + 1) = 4551
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2))  # 4551 -> 4551 // 2 = 2275
    model.add(Activation("relu"))
    # (2275, 13)
    model.add(Conv1D(filters=13, kernel_size=130))  # 2275 --> (2275 - 130 + 1) = 2146
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=4)) # 2146 --> 2146 // 4 = 536
    model.add(Activation("relu"))
    # (536, 13)
    model.add(Flatten()) 
    # (536,13) --> 536 * 13 = (6968, 1)
    model.add(Dropout(dropout_rate))

    # model.add(Dense(256))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))
    # model.add(Dropout(dropout_rate))

    model.add(Dense(13))

    model.compile(loss="mae", optimizer="adam", metrics=metrics)
    return model


def create_model(metrics: list) -> Sequential:

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=13, input_shape=(4563, 1)))  # 4563 -> (4563 - 13 + 1) = 4551
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2))  # 4551 -> (4551 - 2 + 1) = 4550
    model.add(Activation("relu"))

    model.add(Conv1D(filters=16, kernel_size=13))  # 4550 -> (4550 - 13 + 1) = 4538
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2))  # 4538 -> (4538 - 2 + 1) = 4537
    model.add(Activation("relu"))

    model.add(Conv1D(filters=8, kernel_size=13))  # 4537 -> (4537 - 13 + 1) = 4525
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2))  # 4525 -> (4525 - 2 + 1) = 4521
    model.add(Activation("relu"))

    model.add(Conv1D(filters=4, kernel_size=33))  # 4521 -> (4521 - 33 + 1) = 4489
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2))  # 4489 -> (4489 - 2 + 1) = 4488
    model.add(Activation("relu"))

    model.add(Flatten())  # 4488

    model.add(Dense(2048))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))

    model.add(Dense(13))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


