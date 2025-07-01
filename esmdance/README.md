# Fine-Tuning ESMDance with NMA for Binding Prediction

## 1. Project Overview

This project implements an advanced machine learning pipeline to predict protein binding affinity (e.g., enrichment scores) by leveraging and fine-tuning the biophysics-aware protein language model, **ESMDance**.

Standard approaches often rely on evolutionary information from models like ESM-2 or use a single set of biophysical predictions. This project takes a more sophisticated approach by creating a custom model that learns from three distinct sources of information simultaneously:

1. **Raw Evolutionary Data:** From the foundational ESM-2 transformer model.
2. **General Biophysical Dynamics:** From the pre-trained ESMDance model, which predicts features like SASA, RMSF, and various interaction energies based on large-scale MD simulations.
3. **Custom Mutant Dynamics:** From a version of ESMDance that has been specifically fine-tuned to predict the results of **Normal Mode Analysis (NMA)** on a custom-generated library of protein mutants.

The final model (approach 2 in esmdance_fine-tuning_on_yeast_data.ipynb) uses an attention-based architecture to intelligently weigh these different feature sets and make a final prediction, allowing the gradients from the binding data to update and specialize the model's understanding of protein dynamics. This model is compared to several baseline approaches without NMA fine-tuning in baselines.ipynb.

## 2. Pipeline Workflow

The project is executed in three main stages:

### Stage 1: Data Generation

A computational pipeline for generating the necessary structural and dynamic data for a protein of interest.

1. **Mutant Library Generation:** Starting with a single wild-type PDB structure, a large library of mutant PDB files is generated using **PyRosetta**. This process introduces a random number of mutations (e.g., 1-5) at specified positions. The script is parallelized across multiple CPU cores for efficiency and includes robust features like incremental runs and pre-checking for duplicates to avoid wasted computation.
2. **NMA Feature Extraction:** For each relaxed mutant PDB file, **Normal Mode Analysis (NMA)** is performed using the **ProDy** library. This calculates key dynamic properties from the static structure, specifically:
   - **Residue Fluctuations (MSF):** How much each residue is predicted to move (from GNM).
   - **Pairwise Correlations:** How pairs of residues are predicted to move in relation to each other (from ANM). These features are saved as `.npz` files, one for each mutant.

### Stage 2: NMA-Specific Model Fine-Tuning

The goal of this stage is to create an "NMA expert" model.

1. **Load Base ESMDance:** The standard ESMDance model (with its 50/13 output heads) is loaded.
2. **Reconfigure Heads:** The model architecture is reconfigured to have smaller prediction heads that match the NMA data format (3 residue features, 3 pair features).
3. **Load Weights:** The pre-trained weights from the original ESMDance model are loaded, skipping the final layers where the dimensions do not match.
4. **Fine-Tuning:** This new `3/3` model is fine-tuned on the NMA data generated in Stage 1. The result is a new set of model weights for a model that is specialized in predicting NMA dynamics from a protein sequence.

### Stage 3: Final Binding Model Training

This is the final stage where all information is integrated to predict the experimental binding data. This details specifically approach 2 in esmdance_fine-tuning_on_yeast_data.ipynb

1. **Model Definition:** A new, end-to-end model is defined. This model contains:
   - An instance of the NMA-tuned ESMDance model from Stage 2.
   - An advanced `AttentionBindingHead` that processes per-residue and per-pair features.
2. **Selective Unfreezing:** The model is configured such that only specific layers are trainable:
   - The **ESMDance prediction heads** (for both residue and pair features).
   - The **final two layers** of the underlying ESM-2 transformer.
   - The new `AttentionBindingHead`. All other layers remain frozen to preserve their powerful pre-trained knowledge.
3. **End-to-End Training:** This final, integrated model is trained on the experimental dataset (e.g., a CSV with `aa_sequence` and `enrichment_score` columns). The loss from the binding prediction task backpropagates through all trainable components, allowing the model to adapt its understanding of dynamics to be most predictive of binding.
