import synthetic_data_gen as syn
import numpy as np
import config 
import sys


if len(sys.argv[1:]) > 1:
    raise ValueError(f"Only 1 argument should be passed.")
elif int(sys.argv[1]) > 63 or int(sys.argv[1]) < 0:
    raise ValueError(f"Task number should be between 0-{config.n_tasks}")
else:    
    task_number = int(sys.argv[1])

z_values = np.array_split(config.all_z_values, config.n_tasks)[task_number]

def run_dataset(redshift: float):
    
    dataset_generator = syn.DatasetCreation(redshift)
    for line_names in config.all_dataset_names:
        dataset_generator.make_dataset(line_names, z, "validation")



for z in z_values:
    run_dataset(z)