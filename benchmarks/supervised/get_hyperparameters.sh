#!/bin/bash

model_names=( "google/bigbird-pegasus-large-pubmed" \
              "monologg/biobert_v1.1_pubmed" \
              "raynardj/ner-gene-dna-rna-jnlpba-pubmed" \
              "mse30/bart-base-finetuned-pubmed" \
              "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL" \
              "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" \
              "kamalkraj/bioelectra-base-discriminator-pubmed" \
              "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" )

TUNE_DIR="/scratch/taw2/.cache/raytune/"
log_file="output/optimal_hyperparameters.txt"

for model in "${model_names[@]}"
do
  write_file="/scratch/taw2/biosift/benchmarks/supervised/output/hyperparameters/${model//\//_}.json"
  TARGET_DIR="${TUNE_DIR}${model//\//_}_supervised_pbt"
  find $TARGET_DIR -name ".wandb" -prune -o -type d | xargs -n 1 -I {} find {} -type f -name result.json | xargs cat | jq -s 'sort_by(.objective)' | jq '.[-1]' > $write_file
done