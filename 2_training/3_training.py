from tensorflow.compat.v1.logging import set_verbosity, ERROR
from tensorflow.train import latest_checkpoint
from sklearn.preprocessing import normalize
from random import shuffle
from random import sample
from glob import glob
import pandas as pd
import numpy as np
import model

set_verbosity(ERROR)

def train(function, test_or_data: str, files: list, restart: str, callbacks: list):
    nn = function(model.metrics)
    if restart == "y" or "Y":
        np.save("models/mae.npy", [])
        np.save("models/val_mae.npy", [])

        np.save("models/precision.npy", [])
        np.save("models/val_precision.npy", [])
        
        np.save("models/kl.npy", [])
        np.save("models/val_kl.npy", [])
    else:
        checkpoint = latest_checkpoint("./models")
        nn.load_weights(checkpoint)

    for batch in range(len(files) // model.batch_size):
        batch_files = files[batch * model.batch_size : (batch + 1) * model.batch_size]
        spectra = []
        labels = []
        for file in batch_files:
            folder = file.split("/")[3]
            if folder == f"{test_or_data}":
                df = pd.read_parquet(file)
                spectra.append([a.tolist() for a in df.spectra.values])
                labels.append(list(df.labels.values))

            elif folder == "backgrounds":
                with open(file) as f:
                    contents = sample(f.readlines(), model.num_spectra)
                    for line in contents:
                        spectra.append([float(i) for i in line.split(",")])
                    labels.append([0 for _ in range(13)])

        if len(spectra) != len(labels):
            print("Spectra / Label Error")
            continue
        elif np.isnan(spectra).any():
            print("NaN Error")
            continue

        spectra = np.array(spectra).reshape(model.batch_size * model.num_spectra, 4563)
        labels = np.array(labels).reshape(model.batch_size * model.num_spectra, 13)
        nn.fit(
            normalize(spectra),
            labels,
            validation_split=model.validation_split,
            epochs=model.epochs,
            verbose=True,
            callbacks=callbacks,
        )

test_or_data = "data"

data_files = glob(f"../1_data_generation/training_data/{test_or_data}/*/*")
background_files = glob("../1_data_generation/training_data/backgrounds/*")

# Put data_files and background files together
files = data_files #!+ background_files
shuffle(files)

print(f"Number of Files: {len(files)}")
print(f"Number of Batches: {len(files)//model.batch_size}")

restart = input("Do you want to restart? [y]es or [n]o:\t")
# train(model.create_model, test_or_data, files, restart, [metric_callback, checkpoint_callback])
train(
    # model.create_smaller_model,
    model.create_smaller_model,
    test_or_data,
    files,
    restart,
    [model.metric_callback, model.checkpoint_callback, model.early_stopping_callback],
)
