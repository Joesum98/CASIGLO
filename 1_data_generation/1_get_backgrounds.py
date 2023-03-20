from astropy.io import fits
from os.path import join
from tqdm import tqdm
import numpy as np
import config
import time

backgrounds_path = join(
    config.prepend_path, "1_data_generation/training_data/backgrounds/"
)

galaxy_distribution = np.zeros_like(config.all_z_values)

def get_backgrounds(plate: int, ifu: int):
    """Save all background spectra for a given plate-ifu combination into the appropriate text file by combining information from SDSS's specz file and rss file.

    Args:
        plate (int): Plate number
        ifu (int): IFU Design
    """

    specz_filename = join(
        config.specz_folder_path, f"{plate}/specz-{plate}-{ifu}-LOGRSS.fits"
    )
    rss_filename = join(
        config.rss_folder_path, f"{plate}/stack/manga-{plate}-{ifu}-LOGRSS.fits.gz"
    )

    specz_file = fits.open(specz_filename)
    rss_file = fits.open(rss_filename)

    models = specz_file["MODEL"].data
    redshifts = specz_file["REDSHIFTS"].data["SPECZ"]
    index_row = specz_file["REDSHIFTS"].data["INDEX_ROW"]
    fluxs = rss_file["FLUX"].data
    # ivars = rss_file["IVAR"].data
    bin_numbers = np.digitize(redshifts, config.all_z_values) - 1

    for i, foreground in enumerate(models):
        flux = fluxs[index_row[i]]
        background = flux - foreground
        z = config.all_z_values[bin_numbers[i]]
        galaxy_distribution[bin_numbers[i]] = galaxy_distribution[bin_numbers[i]] + 1
        with open(join(backgrounds_path, f"z_{z}.txt"), "a") as file:
            file.write(",".join(map(str, background)))
            file.write("\n")

# Clear old text files from previous runs.
for z in config.all_z_values:
    f = open(join(backgrounds_path, f"z_{z}.txt"), "w")
    f.close()

start_time = time.time()

# Get all backgrounds and save to json files
with fits.open(config.id_file_path) as ids_file:
    plate_ids = ids_file["SUMMARY"].data["PLATE"]  # type: ignore
    ifu_designs = ids_file["SUMMARY"].data["IFU"]  # type: ignore

    print(f"Plates: {len(plate_ids)}\nIFU's: {len(ifu_designs)}")

    for i in tqdm(range(len(plate_ids))):
        plate = plate_ids[i]
        ifu = ifu_designs[i]
        try:
            get_backgrounds(plate, ifu)
        except:
            continue

print("--- %s seconds ---" % (time.time() - start_time))
np.save("../docs/galaxy_distribution.npy", galaxy_distribution)