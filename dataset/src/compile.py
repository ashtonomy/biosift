from utils import PubFetcher
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_dataset(filepath):
    label_df = pd.read_csv(filepath)

if __name__=='__main__':
    logger.warning("Not implemented.")