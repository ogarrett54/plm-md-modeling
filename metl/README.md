# Training a biophysics-aware protein language model

## Objectives

- Use biophysics-aware, generalized or specialized, to learn and predict Pikh1 HMA-AVR-PikC interactions.
- Compare models in terms of performance and data efficiency.

## Resources

- [Biophysics-based protein language models for protein engineering](https://doi.org/10.1101/2024.03.15.585128)
- [metl | Github](https://github.com/gitter-lab/metl)
- [metl-sim | Github](https://github.com/gitter-lab/metl-sim)
- [metl-pretrained | Github](https://github.com/gitter-lab/metl-pretrained)

## Approach

### Overview

1. Fine-tune Gelman et al., 2025 global model on Pikh1-AVR-PikC data
2. Fine-tune ESM2 to predict system-specific Rosetta scores, then use this as a base model to fine-tune on Pikh1-AVR-PikC binding data
3. Recreate Gelman et al., 2025 local model for Pikh1 HMA-AVR-PikC, then fine-tune on Pikh1-AVR-PikC yeast binding data.
4. For each approach, fine-tune on varied amounts of binding data. Compare data efficiency with the equivalents using ESM2 only.

### Fine-tune Gelman et al., 2025 global model

1. Clone GitHub repository metl and metl-pretrained.
2. Follow the finetuning notebook, using the METL-Global 20M 3D model.
3. Fine-tune on dataset with example sized of 80, 320, 1600, and the full dataset. Compare model spearman results to ESM2 8M full-finetuning and ESM2 35M partial fine-tuning with equivalent datasets.

### Pre-training ESM2 with biophysics data

1. Clone GitHub repository metl-sim.
2. Deploy their pipeline to generate Rosetta scores on 300 mutants. Use both the standard and binding scores.
3. "Pre-fine-tune" ESM2 on standard only, binding only, and standard+binding scores.
4. Use this new model to fine-tune on yeast binding data, replacing the original output layer with a randomized linear prediction head (as done in Gelman et al., 2025)
5. Fine-tune on dataset with example sized of 80, 320, 1600, and the full dataset. Compare model spearman results to ESM2 8M full-finetuning and ESM2 35M partial fine-tuning with equivalent datasets.

### Pre-training a biophysics-aware protein language model from scratch

1. Use Gelman et al.'s simulation pipeline to simulate 64,000 mutants, generating both standard and binding scores.
2. Follow Gelman et al.'s work to pretrain a local METL-Bind model on Pikh1-AVR-PikC.
3. Use this model to fine-tune on yeast binding data.
4. Fine-tune on dataset with example sized of 80, 320, 1600, and the full dataset. Compare model spearman results to ESM2 8M full-finetuning and ESM2 35M partial fine-tuning with equivalent datasets.
