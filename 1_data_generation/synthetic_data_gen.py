from numpy.random import uniform, randint
from scipy.stats import norm
from numpy import ndarray
import pandas as pd
import numpy as np
import warnings
import config
import os

warnings.simplefilter("ignore")

emission_line_data = [
    ["O2b", 3727.09, 1.78],
    ["O2a", 3729.88, 1.78],
    ["Hd", 4102.89, 1.52],
    ["Hc", 4341.68, 1.38],
    ["Hb", 4862.68, 1.13],
    ["O3b", 4960.30, 1.09],
    ["O3a", 5008.24, 1.07],
    ["N2b", 6549.86, 0.58],
    ["Ha", 6564.61, 0.58],
    ["N2a", 6585.27, 0.57],
    ["S2b", 6718.29, 0.54],
    ["S2a", 6732.68, 0.54],
]
emission_lines = pd.DataFrame(
    emission_line_data, columns=["names", "wavelength", "zmax"]
).set_index("names")


class DataGeneration:
    """
    Used to create synthetic spectra for the training and validation of the CASIGLO Neural Net.
    """

    def __init__(self, redshift: float, output_path=config.prepend_path):
        """Initialize an object for creating data.

        Args:
            redshift (float): The redshift of synthetic data that will be created.
            wavelengths (ndarray): The wavelengths spectra are observed over (x-axis).
        """
        self.output_path = output_path

        if redshift > config.z_max:
            with open(self.output_path) as f:
                f.write("Redshift (z) value exceeds maximum z measured by MaNGA.")
            raise ValueError(f"Redshift range is {config.z_min}-{config.z_max}")

        self.redshift = redshift

    def remove_peaks(
        self, background: ndarray, size=200, std_multiplier=3, passes=2
    ) -> ndarray:
        """Uses a rolling window to remove peak features from input background spectra.
        Windows do not overlap

        Args:
            background (ndarray): Specrtra to remove peaks from
            size (int, optional): Size of roling window. Defaults to 200.
            passes (int, optional): Number of times to run the peak removal. Defaults to 2.
            snr (int, optional): SNR limit to determine when a peak is detected. Defaults to 3.

        Returns:
            ndarray: background spectra with peaks removed.
        """
        new_background = pd.Series([a for a in background])
        for _ in range(passes):
            for i, window in enumerate(new_background.rolling(size)):
                # No overlapping windows
                if ((i + 1) % size) == 0:
                    indecies = list(window.keys())
                    fluxs = list(window.values)
                # Shorten last window
                elif i >= len(new_background) - 1:
                    leftover_size = len(new_background) % size
                    indecies = list(window.keys())[-leftover_size:]
                    fluxs = list(window.values)[-leftover_size:]
                else:
                    continue
                avg = np.mean(fluxs)
                std = np.std(fluxs)
                for j, flux in enumerate(fluxs):
                    if abs(flux) > abs(std * std_multiplier):
                        new_background[indecies[j]] = avg + std_multiplier * uniform(
                            -std, std
                        )

        return np.array(new_background)

    def get_random_background(self, backgrounds_path) -> ndarray:
        """Gets a random background for the appropriate redshift bin from input backgrounds file.

        Args:
            backgrounds_file (str): Hdf5 file to grab backgrounds from.

        Returns:
            ndarray: Background to be used for spectra.
        """
        with open(backgrounds_path) as f:
            contents = f.readlines()
            random_idx = randint(len(contents))
            line = contents[random_idx]

        return self.remove_peaks(np.asarray([float(i) for i in line.split(",")]))

    def get_single_line(self, line_name: str, z: float) -> ndarray:
        """Returns a gaussian of height 1 and at the location of the speicified emission line.

        Args:
            line_name (str): Name of synthetic emission line to create. (Names chosen from emission_lines DataFrame).

        Returns:
            ndarray: PDF of a gaussian around the emission line location for specified line_name.
        """
        if not config.all_names.__contains__(line_name):
            raise Exception("Invallid Line Name")
        else:
            loc = (1 + z) * emission_lines[
                emission_lines.index == line_name
            ].wavelength.values
            width = (np.random.choice([0.4, 1.7], p=[6 / 7, 1 / 7], size=1))[0]
            gaussian = (
                np.sqrt(2 * np.pi) * width * norm.pdf(config.manga_wave, loc, width)
            )
            return gaussian

    def generate_spectra(
        self, lines_and_snrs: dict, backgrounds_path: str, z: float
    ) -> list:
        """Generates synthetic spectra with the desired emission lines at their specified signal to noise ratios and adds them onto a real background from backgrounds_path.

        Args:
            lines_and_snrs (dict): Dictionary containing line names and their corresponding SNRs (e.g. {"O2a": 4}).
            backgrounds_path (str): Path to appropriate backgrounds file.
        Returns:
            list: Syntheitc spectra of a lensed galaxy
        """
        line_names = list(lines_and_snrs.keys())
        SNRs = list(lines_and_snrs.values())

        if len(line_names) != len(SNRs):
            raise ValueError(
                "Length of line_names and SNRs does not match. Empty Spectra returned"
            )

        background = self.get_random_background(backgrounds_path)
        spectra = np.zeros(len(background))
        for i, name in enumerate(line_names):
            line = self.get_single_line(name, z)
            noise_power = float(np.average(background**2))
            amplitude = SNRs[i] * noise_power
            spectra += amplitude * line

        return np.asarray(spectra+background).astype(np.float32)
    


class DatasetCreation:
    """
    Used to create full datasets for the training and validation of the CASIGLO Neural Net.
    """

    def __init__(self, redshift: float, output_path=config.prepend_path) -> None:
        self.redshift = redshift
        self.background_path = os.path.join(
            config.backgrounds_folder, f"z_{redshift}.txt"
        )
        self.output_path = output_path

    def get_background_redshift(self) -> float:
        """Returns a redshift value 'behind' the backgrounds redshift.
        Used to generate varrying z values for each background redshift bin."""
        factor = uniform(low=1, high=10, size=1)[0]
        return np.round(
            float(uniform(low=factor * self.redshift, high=config.z_max, size=1)), 5
        )

    def get_labels(self, line_names: list, snr_set: list, z: float) -> list:
        """Given the input line names and SNR set, returns the training labels for the CASIGLO
        Deep Learning Neural Network

        Args:
            line_names (list): Emission lines present in spectra.
            snr_set (list): SNR of corresponding emission lines.
            z (float): Background redshift of source galaxy.

        Returns:
            list: Array (length 13) with SNR for each emission line and redshift (last element).
        """
        labels = {name: 0.0 for name in config.all_names}
        for i, name in enumerate(line_names):
            labels[name] = np.round(snr_set[i], 5)

        labels["z"] = z

        return np.array(list(labels.values())).astype(np.float32)

    def make_dataset(
        self,
        line_names: list,
        z: float,
        training_or_validation: str,
    ) -> None:
        """_summary_

        Args:
            line_names (list): Emission lines to generate .
            z (float): Redshift of spectra.
            training_or_validation (str): Whether dataset should be saved under training or validation. Defaults to training.

        Raises:
            ValueError: If training or validation is not "training" or "validation"
        """
        if training_or_validation not in ["training", "validation"]:
            raise ValueError("Must pass either 'training' or 'validation'")

        dc = DataGeneration(redshift=z, output_path=self.output_path)
        snr_combos = config.get_snr_combos(len(line_names))
        df = pd.DataFrame(columns=["spectra", "labels"])

        for snr_set in snr_combos:
            background_z = self.get_background_redshift()
            temp_dict = {name: snr for name, snr in zip(line_names, snr_set)}
            spectra = dc.generate_spectra(temp_dict, self.background_path, background_z)
            labels = self.get_labels(line_names, snr_set, background_z)
            df = df.append({"spectra": spectra, "labels": labels}, ignore_index=True)

        path = os.path.join(
            config.prepend_path,
            f"1_data_generation/{training_or_validation}_data/data/z_{dc.redshift}",
        )
        name = f"{config.join_str.join(line_names)}.parquet"
        try:
            df.to_parquet(os.path.join(path, name), engine="pyarrow")
        except OSError:
            os.makedirs(path, exist_ok=True)
            df.to_parquet(os.path.join(path, name), engine="pyarrow")
