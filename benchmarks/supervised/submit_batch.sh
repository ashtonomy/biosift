#!/bin/bash
#
# Submit Jobs
#   Submit one per model. pass in target at command line
#   Usage
#       e.g. `source submit_hpo_jobs.sh run_hpo.sh`


model_names=( "google/bigbird-pegasus-large-pubmed" \
              "monologg/biobert_v1.1_pubmed" \
              "raynardj/ner-gene-dna-rna-jnlpba-pubmed" \
              "mse30/bart-base-finetuned-pubmed" \
              "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL" \
              "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" \
              "kamalkraj/bioelectra-base-discriminator-pubmed" \
              "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" )
target=$1 

for model in "${model_names[@]}"
do
  qsub -v MODEL_NAME=$model $target
done