from itertools import combinations
from scipy.stats import moyal
from os.path import join
from os import environ
import numpy as np

# Functions
num_spectra = 20
def get_snr_combos(n: int) -> list:
    lst = []
    for _ in range(num_spectra):
        lst.append(moyal.rvs(loc=4, scale=1.5, size=n))
        
    return np.round(lst, 5).tolist()

# Paths
join_str = "-"
prepend_path = "/uufs/astro.utah.edu/common/home/u6031907/casiglo/final_code/"
backgrounds_folder = join(prepend_path, f"1_data_generation/training_data/backgrounds")
training_data_folder = join(prepend_path, "1_data_generation/training_data/data")
checkpoint_folder = join(prepend_path, "2_training/models/best.ckpt")

try:
    manga_directory = environ["MANGA_SPECTRO_REDUX"]
except:
    manga_directory = None


if manga_directory:

    id_file_path = join(manga_directory, "../specz/v3_1_1/1.0.1/speczall.fits")
    specz_folder_path = join(manga_directory, "../specz/v3_1_1/1.0.1")
    rss_folder_path = join(manga_directory, "v3_1_1")

else:
    pass

# Files
manga_wave = np.load(join(prepend_path, "manga_wave.npy"))  # len = 4563

# Values
dz = 0.001
z_min = 0.00370
z_max = 0.14970
n_tasks = 64
all_z_values = np.round(np.linspace(z_min, z_max, 146), 5)

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

all_dataset_names = []
for l in range(1, len(all_names) + 1):
    for subset in combinations(all_names, l):
        all_dataset_names.append(subset)

all_dataset_names.sort(key=len)
