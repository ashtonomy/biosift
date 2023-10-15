"""
compile
Takes csv with label and PMID columns and 
generates a huggingface dataset by querying 
PubMed for the abstract text. Mapping 
with num_workers > 1 not suggested due to
API limits.

"""
from utils import PubFetcher
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_dataset(filepath):
    dataset = load_dataset("csv", data_files="my_file.csv")
    return label_df

if __name__=='__main__':
    logger.warning("Not implemented.")