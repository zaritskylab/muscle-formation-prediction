# Machine learning inference of continuous single-cell state transitions during myoblast differentiation and fusion

<!-- [![DOI](https://zenodo.org/badge/300036005.svg)](https://zenodo.org/badge/latestdoi/300036005) -->
Cells dynamically change their internal organization via continuous cell state transitions to 
mediate a plethora of physiological processes. Understanding such continuous processes is 
severely limited due to a lack of tools to measure the holistic physiological state of single cells 
undergoing a transition. We combined live-cell imaging and machine learning to quantitatively 
monitor skeletal muscle precursor cell (myoblast) differentiation during multinucleated muscle 
fiber formation. Our machine learning model predicted the continuous differentiation state of 
single primary murine myoblasts over time and revealed that inhibiting ERK1/2 leads to a 
gradual transition from an undifferentiated to a terminally differentiated state 7.5-14.5 hours 
post inhibition. Myoblast fusion occurred ~3 hours after predicted terminal differentiation. 
Moreover, we showed that our model could predict that cells have reached terminal 
differentiation under conditions where fusion was stalled, demonstrating potential applications 
in screening. This method can be adapted to other biological processes to reveal connections 
between the dynamic single-cell state and virtually any other functional readout.


## Data collection, processing and model training procedures

![our workflow](6.paper_figures/figures/supplementary/figureS3_workflowpipeline.png)

> Semi-manual single-cell tracking: Time-lapse images were converted to XML/hdf5 format, and Mastodon's FIJI plufing was used for 
> single-cell tracking and manual correction. Cells that fused into multinucleated fibers and cells that did not fuse within the experimental timeframe were included.

Preprocessing trajectories: Image registration using OpenCV's CalcOpticalFlowFarneback was performed to correct erroneous offsets in the tracked cells' trajectories.

Models training: The training pipeline involved the following steps:

a. Determining labels for training: ERKi-treated cells were labeled as "differentiated" in a specific time segment before the first fusion event was observed, while DMSO-treated cells were labeled as "undifferentiated" in specific time segments based on their differentiation timeline.

b. Partitioning single-cell trajectories to temporal segments: Trajectories of DMSO- and ERKi-treated cells were divided into overlapping temporal segments of equal lengths. The specific time segment where ERKi-treated cells were considered "differentiated" was used for training.

c. Extracting motility and actin features: Single-cell motility and actin intensity time series were extracted from each temporal segment.

d. Extracting hundreds of single-cell time series features: Features encoding properties of the temporal segments were extracted using the "tsfresh" Python package.

e. Training classifiers: Random forest classifiers were trained to distinguish between differentiated and undifferentiated cells. Grid search and cross-validation were used for hyperparameter tuning.

Evaluating classifier performance: The trained classifiers' performance was evaluated on an independent experiment. Time series were partitioned into temporal segments, and the corresponding trained models were used to assess discrimination performance.
> 
> 
> 
> 
> 
> 
> Data collection and data processing workflows, related to Figure 1.
(a) We cultured A549 lung cancer cells and exposed them to 1,327 different compound perturbations in about six doses per compound.
We plated these cells in 384 well plates, and, using the same plate layout, measured gene expression (using the L1000 assay) and morphology (using the Cell Painting assay) in compound-perturbed A549 cells.
(b) Our image-based profiling pipeline we used to process the Cell Painting images.
We used pycytominer to process the single cell profiles.
All processing code and profile data are available at https://github.com/broadinstitute/lincs-cell-painting. Image data available at Image Data Resource (accession: idr0125).

## Findings summary

We find that each assay provides complementary information for mapping cell state.
Cell Painting has generally higher reproducibility, but suffers from more technical artifacts.
L1000 has a more diverse feature space, but contains less diverse samples.
In general Cell Painting captures more MOAs by an unsupervised analysis, but L1000 performs better in deep learning predictions.

**Importantly, each assay captures complementary cell states.**
By combining each data type, one can capture more mechanisms than either alone.

See our paper (linked below) for more details and extensive resources.

## Reproduce computational environment

We use a combination of conda and pip to manage the proper python packages for data assessment and model predictions.
To reproduce our environment run the following:

```bash
# In the top folder of the directory
conda env create --force --file environment.yml && conda activate lincs-complimentarity && cd 2.MOA-prediction/ && python setup.py && cd ..
```

We also need to setup custom computational environments for tensorflow and pytorch for the MOA prediction analysis.

```bash
# Navigate into the MOA prediction folder
cd 2.MOA-prediction

# Step 1 - Tensorflow
# Initialize a virtual environment
python3 -m venv tensorflow_env

# Activate the environment
source tensorflow_env/bin/activate

# Upgrade pip if necessary
# python3 -m pip install --upgrade pip

# Install tensorflow requirements
python3 -m pip install -r tensorflow_requirements.txt

# Step 2 - Pytorch
python3 -m venv pytorch_env
source pytorch_env/bin/activate
python3 -m pip install -r pytorch_requirements.txt && python3 setup.py
```

## Citation

For a complete discussion of our findings please view our preprint:

> Morphology and gene expression profiling provide complementary information for mapping cell state.
Gregory P. Way, Ted Natoli, Adeniyi Adeboye, Lev Litichevskiy, Andrew Yang, Xiaodong Lu, Juan C. Caicedo, Beth A. Cimini, Kyle Karhohs, David J. Logan, Mohammad Rohban, Maria Kost-Alimova, Kate Hartland, Michael Bornholdt, Niranj Chandrasekaran, Marzieh Haghighi, Shantanu Singh, Aravind Subramanian, Anne E. Carpenter. bioRxiv 2021.10.21.465335; doi: https://doi.org/10.1101/2021.10.21.465335 
