"""
PubFetcher

Fetch abstracts by PMID using EFetch API
https://www.nlm.nih.gov/dataguide/eutilities/how_eutilities_works.html
"""
import requests
import typing
from typing import List, Union, Optional
from functools import partial
import logging
import time

logger = logging.getLogger(__name__)

# Type aliases for id arguments. Iterab
PMID = Union[int, str]
PMIDList = List[PMID]

class PubFetcher:
    def __init__(
        self,
        base_url: str='http://eutils.ncbi.nlm.nih.gov/entrez/eutils/', 
        util: str='efetch.fcgi',
        db: str='pubmed',
        ids: Optional[PMIDList]=None,
        step_size: int=1,
        timeout: float=5.0,
        attempts: int=3,
        **kwargs
    ):
        """
        Initialize with API path specifications. If ids are passed at initialization, 
        PubFetcher can be used as an iterable.

        Arguments
            base_url: str base url for endpoint
            util: str e-utility to use. Defaults to 'efetch.fcgi
            db: Database to search. Default is pubmed
            ids: Optional list of pubmed ids. If passed, can be called as an iterable that 
                 returns request content. Useful for large batches.
            if ids is not None:
                step_size: Number of ids to process per iteration
                timeout: Timeout between failed requests. Total possible wait time 
                     for failed requests is <attempts> * wait time
                attempts: Number of attempts to try in case of failed request
                iter_args: Kwargs for fetch and split, when using iterator
                        WARNING: These will override fetch arguments if passed
        
        """
        # Build endpoint base
        self.base_url = base_url
        self.util = util
        self.db = db
        self.base = f'{base_url}{util}?db={self.db}'
        
        # For use as an iterable
        self.ids = ids
        self.step_size = step_size
        self.timeout = timeout
        self.attempts = attempts
        self.iter_args = kwargs
        
        # Sanity checks/usage warnings
        if self.step_size == 1:
            logger.debug("Step size is set to 1 by default. Consider increasing to process larger pmid batches.")
        if self.ids is None and self.iter_args != {}:
            logger.warning("Passing iterator arguments for use outside of iteration is not supported.")

    def __iter__(self):
        self.iteration = 0

        # If iter args are passed, override default arguments
        if self.iter_args is not None:
            retmode = iter_args["retmode"] if "retmode" in self.iter_args else "text"
            rettype = iter_args["rettype"] if "rettype" in self.iter_args else "abstract"
            retstart = iter_args["retstart"] if "retstart" in self.iter_args else None
            retmax = iter_args["retmax"] if "retmax" in self.iter_args else None
            self._iter_fetch = partial(self.fetch, 
                                       retmode = retmode, 
                                       rettype = rettype, 
                                       retstart = rettype, 
                                       retmax = retmax)
        else:
            self._iter_fetch = self.fetch
        
        return self

    def __next__(self):
        start_idx = self.iteration * self.step_size

        if start_idx >= len(self.ids):
            raise StopIteration

        end_idx = start_idx + self.step_size

        tries = 0
        while tries < self.attempts:
            logging.debug(f"Requesting pmids {start_idx} to {end_idx}")
            records = self._iter_fetch(self.ids[start_idx:end_idx])
            
            # Handle failed request
            if isinstance(records, int):
                logger.warning("Unable to process pmids {start_idx} to {end_idx} due to {records} response.")
                time.sleep(self.timeout)
                tries += 1
            else:
                self.iteration += 1
                return records
        
        logger.warning("Exceeded max request attempts for pmids {start_idx} to {end_idx}. Returning None")
        self.iteration += 1
        return None
    
    @staticmethod
    def split_records(
        records: str,
        separator: str="\n\n\n"
    ) -> list:
        record_list = [record.strip() for record in records.split(separator)]
        return record_list

    def fetch(
        self,
        ids: Union[PMID, PMIDList],
        retmode: str='text',
        rettype: str='abstract',
        retstart: Optional[Union[int, str]]=None,
        retmax: Optional[Union[int, str]]=None,
        split: bool=True,
        separator: str="\n\n\n",
        **kwargs
    ) -> str:
        """
        Build and process a GET request with specified parameters returning
        decoded content if possible. Uses base url, e-utility and database in self.
        More info on arguments at https://www.ncbi.nlm.nih.gov/books/NBK25499/

        Arguments
            ids: List of uids/pmids to fetch
            retmode: typically "xml" or "text", for more info check docs
            rettype: field to return (defaults to abstract), for more info check docs
            retstart: start index for returned items ( int in [1,10000] )
            retmax: num of returned items ( int in [1,10000] )
            split: whether to return 
            separator: string on which to split
            kwargs: additional query params for request 
                    where k, v in kwargs becomes '&{k}={v}'
        
        Returns
            str if request and decode is successful and split is false
            list if request and decode is successful and split is True
            integer status code if request is not successful

        """
        
        if isinstance(ids, List):
            if isinstance(ids[0], int):
                ids = [str(i) for i in ids]
            ids = ",".join(ids)
        elif isinstance(ids, int):
            ids = str(ids)
            split = False
        else:
            split=False
        
        # Build endpoint with parameters
        req = self.base
        req += f'&id={ids}'
        req += f'&retmode={retmode}'
        req += f'&rettype={rettype}'
        req += f'&retstart={retstart}' if retstart is not None else ''
        req += f'&retmax={retmax}' if retmax is not None else ''
        if kwargs is not None:
            for k, v in kwargs:
                req += f'&{k}={v}'

        response = requests.get(req)
        if response.status_code != 200:
            logger.warning(f'Unable to process GET request due to status code {response.status_code}.')
            logger.debug(req)
            return response.status_code

        content = response.content.decode()

        if split:
            return self.split_records(content, separator)
        
        return content

