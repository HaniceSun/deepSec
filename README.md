<p align="left">
<img src="assets/logo.png" alt="deepSec logo" width="250"/>
</p>

# Deep Learning Prediction of Selenocysteine Sites (deepSec)

## Overview

deepSec is a deep learning-based tool designed to predict selenocysteine (Sec) incorporation sites. Selenocysteine, known as the 21st amino acid, is incorporated into proteins via a unique mechanism that involves the recoding of the UGA stop codon. Selenocysteine is an evolutionarily conserved amino acid found in all three domains of life. It has a lower reduction potential than cysteine, which makes it particularly effective in enzymes involved in antioxidant activity. Dietary selenium is incorporated into selenoproteins that support essential functions throughout the body. Accurate prediction of Sec sites - either through the discovery of novel Sec-containing proteins or the annotation of Sec-inducing mutations - is essential for understanding Sec-related protein function and diseases.

## Installation

- using conda

```
git clone git@github.com:HaniceSun/deepSec.git
cd deepSec
conda env create -f environment.yml
conda activate deepSec
```

# Quick Start

```
ds download-training-data
ds preprocess --down_sampling_fold 100
ds torch-dataset --split_by_species=True

train_file=Ensembl_Protein2Transcript_Upos_FeatureSeq_DownSampling100_dataset_train.pt
val_file=Ensembl_Protein2Transcript_Upos_FeatureSeq_DownSampling100_dataset_val.pt
test_file=Ensembl_Protein2Transcript_Upos_FeatureSeq_DownSampling100_dataset_test.pt
seq=TCAAGAGCCTCAACAACAGGGTGGTTCTGGTCGTTAACGTTGCAAGCAAATGAGGCCTGACTGCCGCGAACTACAAGGAGTTCGCGACTCTGCTTGGCAA

ds train --model_name GPT  --config_file config.yaml --train_file $train_file --val_file $val_file
ds train --model_name SpliceAI  --config_file config.yaml --train_file $train_file --val_file $val_file
ds train --model_name ResNet  --config_file config.yaml --train_file $train_file --val_file $val_file
ds train --model_name CNN  --config_file config.yaml --train_file $train_file --val_file $val_file

ds test --test_file $test_file --model_name SpliceAI --epoch 9

ds predict --pred_seq $seq --model_name SpliceAI --epoch 9 --out_file predicted_SpliceAI_ckpt9.txt

ds saliency --sal_seq $seq --model_name SpliceAI --epoch 9 --out_file saliency_SpliceAI_ckpt9.txt
```

## Author and License

**Author:** Han Sun

**Email:** hansun@stanford.edu

**License:** [MIT License](LICENSE)
