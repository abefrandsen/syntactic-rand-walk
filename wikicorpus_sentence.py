"""
This file contains rewrites of certain classes and functions in gensim related to
iterating through the wikipedia corpus. The changes made allow you to iterate
through each article sentence by sentence, rather than word by word. The main
class here is WikiCorpusBySentence, which has the same interface as WikiCorpus in
gensim.corpora.wikicorpus. 
"""

import gensim
import nltk
import bz2
import logging
import multiprocessing
import re
import signal
from xml.etree.cElementTree import \
    iterparse  # LXML isn't faster, so let's go with the built-in solution
from nltk import tokenize as nltk_tokenize
from gensim import utils
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora.wikicorpus import *

def process_article_by_sentence(args, tokenizer_func=tokenize, token_min_len=TOKEN_MIN_LEN,
                    token_max_len=TOKEN_MAX_LEN, lower=True):
    """Parse a wikipedia article sentence by sentence, extract all tokens.

    Notes
    -----
    Set `tokenizer_func` (defaults is :func:`~gensim.corpora.wikicorpus.tokenize`) parameter for languages
    like japanese or thai to perform better tokenization.
    The `tokenizer_func` needs to take 4 parameters: (text: str, token_min_len: int, token_max_len: int, lower: bool).

    Parameters
    ----------
    args : (str, bool, str, int)
        Article text, lemmatize flag (if True, :func:`~gensim.utils.lemmatize` will be used), article title,
        page identificator.
    tokenizer_func : function
        Function for tokenization (defaults is :func:`~gensim.corpora.wikicorpus.tokenize`).
        Needs to have interface:
        tokenizer_func(text: str, token_min_len: int, token_max_len: int, lower: bool) -> list of str.
    token_min_len : int
        Minimal token length.
    token_max_len : int
        Maximal token length.
    lower : bool
         If True - convert article text to lower case.
    sent : bool
         If True - return list of list of str, where the tokens are grouped by sentence

    Returns
    -------
    (list of str, str, int)
        List of tokens from article, title and page id.

    """
    text, lemmatize, title, pageid = args
    text = filter_wiki(text)
    if lemmatize:
        result = [utils.lemmatize(s) for s in nltk_tokenize.sent_tokenize(text)]
    else:
        result = [tokenizer_func(s,token_min_len,token_max_len,lower) for s in nltk_tokenize.sent_tokenize(text)]
    return result, title, pageid

def process_article_by_sentence_helper(args):
    """Same as :func:`process_article_by_sentence`, but with args in list format.

    Parameters
    ----------
    args : [(str, bool, str, int), (function, int, int, bool)]
        First element - same as `args` from :func:`~gensim.corpora.wikicorpus.process_article`,
        second element is tokenizer function, token minimal length, token maximal length, lowercase flag.

    Returns
    -------
    (list of str, str, int)
        List of tokens from article, title and page id.

    Warnings
    --------
    Should not be called explicitly. Use :func:`~gensim.corpora.wikicorpus.process_article` instead.

    """
    tokenizer_func, token_min_len, token_max_len, lower = args[-1]
    args = args[:-1]

    return process_article_by_sentence(
        args, tokenizer_func=tokenizer_func, token_min_len=token_min_len,
        token_max_len=token_max_len, lower=lower
    )


class WikiCorpusBySentence(TextCorpus):
    """Treat a wikipedia articles dump as a **read-only** corpus, iterate through it article by article,
    with each article being returned as a list of sentences.

    Supported dump formats:

    * <LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2
    * <LANG>wiki-latest-pages-articles.xml.bz2

    The documents are extracted on-the-fly, so that the whole (massive) dump can stay compressed on disk.

    Notes
    -----
    Dumps for English wikipedia can be founded `here <https://dumps.wikimedia.org/enwiki/>`_.

    Attributes
    ----------
    metadata : bool
        Whether to write articles titles to serialized corpus.

    Warnings
    --------
    "Multistream" archives are *not* supported in Python 2 due to `limitations in the core bz2 library
    <https://docs.python.org/2/library/bz2.html#de-compression-of-files>`_.

    Examples
    --------
    >>> from gensim.corpora import WikiCorpus, MmCorpus
    >>>
    >>> wiki = WikiCorpus('enwiki-20100622-pages-articles.xml.bz2') # create word->word_id mapping, takes almost 8h
    >>> MmCorpus.serialize('wiki_en_vocab200k.mm', wiki) # another 8h, creates a file in MatrixMarket format and mapping

    """

    def __init__(self, fname, processes=None, lemmatize=utils.has_pattern(), dictionary=None,
                 filter_namespaces=('0',), tokenizer_func=tokenize, article_min_tokens=ARTICLE_MIN_WORDS,
                 token_min_len=TOKEN_MIN_LEN, token_max_len=TOKEN_MAX_LEN, lower=True):
        """Initialize the corpus.

        Unless a dictionary is provided, this scans the corpus once,
        to determine its vocabulary.

        Parameters
        ----------
        fname : str
            Path to file with wikipedia dump.
        processes : int, optional
            Number of processes to run, defaults to **number of cpu - 1**.
        lemmatize : bool
            Whether to use lemmatization instead of simple regexp tokenization.
            Defaults to `True` if *pattern* package installed.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Dictionary, if not provided,  this scans the corpus once, to determine its vocabulary
            (this needs **really long time**).
        filter_namespaces : tuple of str
            Namespaces to consider.
        tokenizer_func : function, optional
            Function that will be used for tokenization. By default, use :func:`~gensim.corpora.wikicorpus.tokenize`.
            Need to support interface:
            tokenizer_func(text: str, token_min_len: int, token_max_len: int, lower: bool) -> list of str.
        article_min_tokens : int, optional
            Minimum tokens in article. Article will be ignored if number of tokens is less.
        token_min_len : int, optional
            Minimal token length.
        token_max_len : int, optional
            Maximal token length.
        lower : bool, optional
             If True - convert all text to lower case.
        
        """
        self.fname = fname
        self.filter_namespaces = filter_namespaces
        self.metadata = False
        if processes is None:
            processes = max(1, multiprocessing.cpu_count() - 1)
        self.processes = processes
        self.lemmatize = lemmatize
        self.tokenizer_func = tokenizer_func
        self.article_min_tokens = article_min_tokens
        self.token_min_len = token_min_len
        self.token_max_len = token_max_len
        self.lower = lower
        self.dictionary = dictionary 
        #self.dictionary = dictionary or Dictionary(self.get_texts())

    def get_texts(self):
        """Iterate over the dump, yielding list of tokens for each article.

        Notes
        -----
        This iterates over the **texts**. If you want vectors, just use the standard corpus interface
        instead of this method:

        >>> for vec in wiki_corpus:
        >>>     print(vec)

        Yields
        ------
        list of str
            If `metadata` is False, yield only list of token extracted from the article.
        (list of str, (int, str))
            List of tokens (extracted from the article), page id and article title otherwise.

        """

        articles, articles_all = 0, 0
        positions, positions_all = 0, 0

        tokenization_params = (self.tokenizer_func, self.token_min_len, self.token_max_len, self.lower)
        texts = \
            ((text, self.lemmatize, title, pageid, tokenization_params)
             for title, text, pageid
             in extract_pages(bz2.BZ2File(self.fname), self.filter_namespaces))
        pool = multiprocessing.Pool(self.processes, init_to_ignore_interrupt)

        try:
            # process the corpus in smaller chunks of docs, because multiprocessing.Pool
            # is dumb and would load the entire input into RAM at once...
            for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
                for tokens, title, pageid in pool.imap(process_article_by_sentence_helper, group):
                    articles_all += 1
                    positions_all += len(tokens)
                    # article redirects and short stubs are pruned here
                    tok_len = sum([len(s) for s in tokens])
                    if tok_len < self.article_min_tokens or \
                            any(title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):
                        continue
                    articles += 1
                    positions += tok_len 
                    if self.metadata:
                        yield (tokens, (pageid, title))
                    else:
                        yield tokens

        except KeyboardInterrupt:
            logger.warn(
                "user terminated iteration over Wikipedia corpus after %i documents with %i positions "
                "(total %i articles, %i positions before pruning articles shorter than %i words)",
                articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS
            )
        else:
            logger.info(
                "finished iterating over Wikipedia corpus of %i documents with %i positions "
                "(total %i articles, %i positions before pruning articles shorter than %i words)",
                articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS
            )
            self.length = articles  # cache corpus length
        finally:
            pool.terminate()
