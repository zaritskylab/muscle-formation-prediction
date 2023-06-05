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

<p align="center">
<img src="figures/figure_2A.png" width=60%>
</p>

> **Semi-manual single-cell tracking:** Time-lapse images were converted to XML/hdf5 format, and Mastodon's FIJI plugin was used for single-cell tracking and manual correction. Cells that fused into multinucleated fibers and cells that did not fuse within the experimental timeframe were included.

> **Preprocessing trajectories:** Image registration using OpenCV's CalcOpticalFlowFarneback was performed to correct erroneous offsets in the tracked cells' trajectories.

> **Models training:** The training pipeline involved the following steps:
>   a. Determining labels for training: ERKi-treated cells were labeled as "differentiated" in a specific time segment before the first fusion event was observed, while DMSO-treated cells were labeled as "undifferentiated".
>   b. Partitioning single-cell trajectories to temporal segments: Trajectories of DMSO- and ERKi-treated cells were divided into overlapping temporal segments of equal lengths.
>   c. Extracting motility and actin features: Single-cell motility and actin intensity time series were extracted from each temporal segment.
>   d. Extracting hundreds of single-cell time series features: Features encoding properties of the temporal segments were extracted using the "tsfresh" Python package.
>   e. Training classifiers: Random forest classifiers were trained to distinguish between differentiated and undifferentiated cells.

## Findings summary

We used live cell imaging and machine learning to track the differentiation state of muscle cells during muscle fiber formation. Our findings include identifying the time frame of myoblast differentiation and its link to fusion events. We also validated that inhibiting fusion did not significantly affect the differentiation process. This approach has potential applications in identifying new factors and screening compounds for muscle regeneration. Our study highlights the importance of supervised machine learning in accurately inferring cell state and its broader applicability to studying other dynamic cellular processes.

See our paper (linked below) for more details and extensive resources.

<!-- ## Reproduce computational environment

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
python3 -m pip install -r pytorch_requirements.txt && python3 setup.py -->
```

## Citation

For a complete discussion of our findings please view our preprint:

> Machine learning inference of continuous single-cell state transitions during myoblast differentiation and fusion
Amit Shakarchy, Giulia Zarfati, Adi Hazak, Reut Mealem, Karina Huk, Ori Avinoam, Assaf Zaritsky
bioRxiv 2023.02.19.529100; doi: https://doi.org/10.1101/2023.02.19.529100

