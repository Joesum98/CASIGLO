# Project Title

 Computer Assisted Spectroscopic Inspection of Gravitational Lensing Objects

## Core Hours Requested

 Fall     2022: 30,000
 Winter     2023: 30,000
 Spring    2023: 30,000
 Summer     2023: 30,000

## Proposal

### Project Abstract

#### Abstract

 Searching for gravitationally lensed objects in galactic spectra has always been a very tedious process: manually clicking through thousands of identified strong-lens candidates in order to label features related to the presence of emission-lines in the background of each galaxy. This manual process is labor intensive, and requires many hours of training to recognize specific features, and likely introduces bias to the final catalogs that may either include false positives or miss real lenses, which adds expense to followup oservation by space telescopes, such as Hubble and James Webb.

    For example, the Spectroscopic Identification of Lensing Objects (SILO) project is the largest search for strong lenses using the ~2 million galaxy spectra contained within the extended Baryon Oscillation Spectroscopic Survey (eBOSS) from the sixteenth data release (DR16) of the Sloan Digital Sky Survey (SDSS), where 1,511 possible strong lens candidates were found by Talbot et al.[2] in 2021, after manually inspecting tens of thousands of spectra. This survey extended the initial BOSS survey by Brownstein et al.[1] from 2011 which found 45 candidates of which 36 were confirmed by Hubble. Initial MaNGA searches for lensing objects from the ~10,000 galaxy targets found only 8 likely, 17 probable, and 69 possible strong galaxy-galaxy gravitational lens candidates found, as by Talbot et al.[4] in 2022, but similarly required "months" of manual inspection in order to grade each one.

    However, with machine learning frameworks (PyTorch, Tensorflow, Keras, etc.) recently becoming much more accessible, this inspection process could be automated, and sped up drastically by training a neural network to recognize the signs of gravitationally lensed objects in their spectra. This would not only accelerate the process of finding lenses for researchers, but also eventually be more reliable, reproducable, and less error-prone than human inspection. Training a neural network to detect lenses could upgrade previously downgraded lenses and lead to more discoveries about dark matter, and remove human selection bias in grading gravitational lens candidates.

    Machine Learning is a vibrant scientific working group within SDSS, including a collaborative mailing list data-science@sdss.org; and most recently, new work within SDSS-V such as Straumit et al.[5] demonstrate attempts to automate spectroscopic analysis for the Milky Way Mapper (MWM) program. Other astro surveys have also had machine learning algorithms implemented to help improve their searches for particular objects as well, such as He & Li[6] who created a random forrest algorithm for quasar identification in the 9th data release of the DESI Survey.

    Using these data science frameworks, we aim to train a custom neural network to recognize lensed objects in MaNGA spectra. For training data, we utilize synthetic spectra, as Ramachandra et al.[3] found in their 2021 paper, "physical modeling of synthetic data can outperform purely observation-based training". To ensure that training data is representative of actual MaNGA spectra, synthetic spectra are generated using real foreground-subtracted spectra from MaNGA, as well as Gaussian fits of emission-lines from the same data. To ensure our model is being trained properly to modern data science standards, we are co-investigating with Prof. Jeff Phillips: director of the Utah Center for Data Science. After the initial training, which recovers human inspection at >90% without refinement, we intend to add refinement to the detection algorithms to minimize false positives and reduce the deep learning technical debt by implementing physics and human knowledge through a correction function applied to the model’s output.

### Calculation

 We are requesting core hours to generate synthetic data and train our model.
 The following is an outline of how the calculations for both data creation and training will be run.

 1.) Data creation:

  for each line_names in all_dataset_names:
   for each z in all_z_values:
    create random list of 20 SNR combinations [each of len(line_names)]
    create 20 synthetic spectra using those SNR combinations
    write dataset consisting of those 20 spectra

  generate ~1,000 spectra with all emission-lines at random SNRs to use for validation

 2.) Training:

  for each folder in data_location:
   for each dataset in folder:
    get spectra and labels from dataset
    model.fit(spectra, labels)
    save/overwrite model

## Sources of Funding

 This project uses SDSS-IV data, which was fully funded at Utah but is no longer current due to the beginning of SDSS-V. This project is not an SDSS-V project and is not eligible to use SDSS-V funded resources.

## Publications and etc

1.) Brownstein et al. (2011) The BOSS Emission-Line Lens Survey (BELLS). I. A large spectroscopically selected sample of Lens Galaxies at redshift ~ 0.5 (<https://ui.adsabs.harvard.edu/abs/2012ApJ...744...41B/abstract>)

2.) Talbot et al. (2021) The completed SDSS-IV extended Baryon Oscillation Spectroscopic Survey: a catalogue of strong galaxy-galaxy lens candidates (<https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.4617T/abstract>)

3.) Ramachandra et al. (2021): Machine learning synthetic spectra for probabilistic redshift estimation: SYTH-Z (<https://ui.adsabs.harvard.edu/abs/2022MNRAS.515.1927R/abstract>)

4.) Talbot et al. (2022) SDSS-IV MaNGA: A Catalogue of Spectroscopically Detected Strong Galaxy-Galaxy Lens Candidates (<https://ui.adsabs.harvard.edu/abs/2022MNRAS.515.4953T/abstract>)

5.) Straumit et al. (2022) Zeta-Payne: A Fully Automated Spectrum Analysis Algorithm for the Milky Way Mapper Program of the SDSS-V Survey (<https://ui.adsabs.harvard.edu/abs/2022AJ....163..236S/abstract>)

6.) He & Li (2022) The Quasar Candidates Catalogs of DESI Legacy Imaging Survey Data Release 9 (<https://ui.adsabs.harvard.edu/abs/2022arXiv220706792H/abstract>)
