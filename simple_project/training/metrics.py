import os
import math
import copy
import pickle
import string
import spacy
import textstat
import numpy as np
from tqdm import tqdm
from typing import Union, List
from functools import partial
from collections import Counter

from torchtext.datasets import WikiText2
#from spacy_syllables import SpacySyllables
from torchtext.data.utils import get_tokenizer
from concurrent.futures import ProcessPoolExecutor
from torchtext.vocab import build_vocab_from_iterator

ALL_DATASETS = ["OSE", "SimpleWiki","SimpleGerman"]
PATHS = {"OSE": "/home/iailab34/selbacht0/data/OneStopEnglishCorpus/data",
         "Simple": "/home/iailab34/selbacht0/data/SimpleWikipedia/data"}


class SentenceEvaluator:
    """ Class that implements all metric from the D. Campos' "Curriculum Learning for Language Models" (2021) and more
    to evaluate the difficulty of a sentence. Clear text input is needed, because spacy uses their own tokenizers.
    The metric are:
        * sentence length
        * uni-, bi- and trigram probabilities
        * word rarity
        * diversity: unique part of speech tags
        * complexity: depth of a dependency parse tree
    As well as the following reading scores:
        * Lix formula
        * SMOG grade
        * Gunning Fog Index
        * Coleman-Liau Index
        * Automated Readibility Index
        * Flesch-Kincaid-Readibility Test

        Args:
            raw_corpus (): raw corpus, not tokenized or lemmatized
            max_ngrams (default 3): used in _create_corpus_counts()
    """

    def __init__(self, raw_corpus: list = None, lang: str = "en", specials: list =["<unk>"], used_metrics: bool = True,
                 more_trigrams: bool = False):
        """
        Args:
            raw_corpus (list of strings): containing one entry per word
            specials (list of strings): list of special words added to the raw_corpus when calculating the stats
            more_trigrams (bool): whether to create trigrams à la (word, <unk>, word) or not
        Returns:
            eval_results (dict): collected results for each metric
        """
        self.specials = specials
        self.more_trigrams = more_trigrams
        self.has_corpus = False if not raw_corpus else True
        self.lang = lang
        textstat.set_lang(lang)

        self.all_metrics = {"length": self._length,
                            "uni_entropy": partial(self._ngram_entropy, 1),
                            "bi_entropy": partial(self._ngram_entropy, 2),
                            "tri_entropy": partial(self._ngram_entropy, 3),
                            "word_rarity": self._word_rarity,
                            "diversity": self._diversity,
                            "complexity": self._complexity,
                            # "bert": self._bert,
                            "lix": self._lix_formula,
                            # "smog": self._smog_grade,
                            "gunning_fog": self._gunning_fog_index,
                            "coleman_liau": self._coleman_liau_index,
                            "autom_reading": self._automated_readibility_index,
                            "flesch_kincaid": self._flesch_reading_ease,
                            }

        if isinstance(used_metrics, list):
            self.metrics = {d: v for d, v in self.all_metrics.items() if d in used_metrics}
        elif isinstance(used_metrics, bool) and used_metrics is True:
            self.metrics = self.all_metrics
        else:
            print(f"{used_metrics} not in {list(self.all_metrics.keys())}. Provide a list of those metrics or none "
                  f"(defaults to all metrics).")

        if "bert" in self.metrics.keys():
            # init bert model according to lang
            pass

        if self.has_corpus:
            # TODO: add to every n-gram evaluation method clause if has_corpus == False
            self.uni_counts = self._count_ngrams(raw_corpus, 1)
            self.bi_counts = self._count_ngrams(raw_corpus, 2)
            self.tri_counts = self._count_ngrams(raw_corpus, 3)
            self.corpus_counts = {**self.uni_counts, **self.bi_counts, **self.tri_counts}
        else:
            print("No raw corpus, evaluator won't be able to evaluate any metrics based on n-gram probabilities")

        self.nlp = spacy.load("en_core_web_trf")
        self.nlp.add_pipe("syllables", after="tagger")

        self.sentence_count = len([])

    def _calculate_metrics(self, sentence: str) -> dict:
        results = {}
        sentence = sentence.lower().split(" ")
        for m in self.metrics:
            results[m] = self.metrics[m](sentence)
        return results

    def _complete_dict(self, datapoint: dict):
        """
        Complete a given datapoint by calculating all metrics for the input sentence and adds them as dictionary
        entries.
        Args:
            datapoint [dict]: contains two entries 'sentence' and 'level'

        Returns:
            results [dict]: copy of datapoint with an added entry 'metrics' containing a dict with values for
                            each metric.
        """
        metric_results = {}
        sentence = [s.lower() for s in datapoint["sentence"]]
        for m in self.metrics:
            metric_results[m] = self.metrics[m](sentence)

        # should not be needed
        results = copy.deepcopy(datapoint)
        results["metrics"] = metric_results
        return results

    # TODO: divide functionality - eval sent/ eval list of sent/ eval sents in dict
    def evaluate(self, sentences: Union[str, list], max_workers: Union[None, int] = None, chunksize: int = 20) -> Union[
        dict, List[dict]]:
        """
        Function to evaluate a sentence (str) or a list of sentences; uses statistics from the corpus used at
        initialization of the SentenceEvaluator for some metrics.
        Args:
            sentences (string): string of words
            max_workers (int): parameter for multiprocessing; default None for no multiprocessing
            chunksize (int): parameter for map() function, multiprocessing

        Returns:
            eval_results (dict): results for different metric
        """
        if isinstance(sentences, str):
            sentences = sentences.strip().translate(str.maketrans('', '', string.punctuation))
            sentences = " ".join([i for i in sentences.split(' ') if i != ''])
            return self._calculate_metrics(sentences)
        elif isinstance(sentences[0], list):
            return [self._calculate_metrics(s) for s in sentences]

        elif isinstance(sentences[0], dict) and max_workers:
            print("Start metric calculations in parallel.")
            # TODO: take a look at chunksize of map()
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self._complete_dict, sentences, chunksize=chunksize))

        elif isinstance(sentences[0], dict) and not max_workers:
            print("Start metric calculations.")
            results = []
            for ix, s in enumerate(tqdm(sentences)):
                results.append(self._complete_dict(s))
                if ix % 500 == 0:
                    with open(os.path.join(PATHS["Simple"], "int_simple.pkl"), "wb") as f:
                        pickle.dump(results, f)

        else:
            raise TypeError(f"Evaluate expects either a sentence (string) or a list of sentences or a list of dicts "
                            f"containing the sentences and other information, but received {type(sentences)}")

        return results

    def _length(self, sentence: list):
        return len(sentence)

    def _word_rarity(self, sentence: list):
        """
        Word (unigram) rarity according to Platanios et al. 2019:
        For each word, rarity is defined as -log(p(word)) where
        p(word) = (frequency of word in corpus) / (total number of words in corpus).
        Returns:
            Average negative log probability (i.e., surprisal per word) for the sentence.
        """
        if not self.has_corpus:
            print("Missing corpus; can't calculate")
            return None

        total_unigrams = sum(self.uni_counts.values())
        rarity = 0.0
        for w in sentence:
            if w in self.uni_counts and self.uni_counts[w] > 0:
                p = self.uni_counts[w] / total_unigrams
                rarity += -math.log(p)
            else:
                print(f"Word {w} not in corpus or frequency zero.")
        # Normalize by the sentence length (number of words)
        normalized_rarity = rarity / len(sentence) if len(sentence) > 0 else None
        return np.round(normalized_rarity, 3)

    def _ngram_entropy(self, n, sentence):
        """
        "Sentence entropy" according to Campos 2021 for uni-, bi- and trigrams:
        For each ngram, entropy contribution is -log(p(ngram)) where
        p(ngram) = (frequency of the ngram in corpus) / (total frequency of all ngrams of order n).
        Returns:
            Sum of negative log probabilities of all ngrams in the sentence.
        """
        # Create a list of ngram occurrences from the sentence.
        ngrams = [tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)]

        # Compute total number of ngrams (of order n) in the corpus.
        total_ngrams = sum(count for ngram, count in self.corpus_counts.items() if len(ngram) == n)

        entropy = 0.0
        for ngram in ngrams:
            try:
                entropy += -math.log(self.corpus_counts[ngram] / total_ngrams)
            except KeyError:
                if n == 3 and self.more_trigrams:
                    # Try to use a fallback: replace the middle token with <unk>.
                    ngram_list = list(ngram)
                    ngram_list[1] = "<unk>"
                    fallback_ngram = tuple(ngram_list)
                    try:
                        p = self.corpus_counts[fallback_ngram] / total_ngrams
                    except KeyError:
                        print(f"ngram {ngram} (even with fallback {fallback_ngram}) not in corpus")
                        continue
                else:
                    print(f"ngram {ngram} not in corpus")
                    continue
        return np.round(entropy, 3)

    def _count_ngrams(self, corpus: list, n: int):
        counts = dict()
        for idx in range(len(corpus)):
            try:
                chunk = corpus[idx: idx + n]
                # discard last chunk of words, where len(chunk) < n
                if len(chunk) < n:
                    break
                chunk = [c.lower() for c in chunk]
                counts[" ".join(chunk)] += 1
            # thrown when word not yet in the counts dict
            except KeyError:
                # works, because exception will be thrown after initializing chunk
                counts[" ".join(chunk)] = 1

            # extra entries for trigrams à la (word, <unk>, word)
            if self.more_trigrams and len(chunk) == 3:
                try:
                    masked_chunk = chunk
                    masked_chunk[1] = "<unk>"
                    counts[" ".join(masked_chunk)] += 1
                except KeyError:
                    counts[" ".join(masked_chunk)] = 1
        return counts

    def _diversity(self, sentence: list):
        """
        The higher the diversity of part-of-speech tags, the harder the sentence.
        Returns:

        """
        doc = self.nlp(" ".join(sentence))
        pos_tags = set()
        for token in doc:
            pos_tags.add(token.pos_)
        return len(set(pos_tags))

    def _complexity(self, sentence: list):
        """
        The higher the parse tree complexity, the harder the sentence.
        Returns:
            depth (int): depth of the parse tree
        """
        doc = self.nlp(" ".join(sentence))
        # apparently SimpleWikipedia has a data point where spacy can't find the sentence's head
        try:
            depth = self._get_depth([token for token in doc if token.head == token][0], 1)
        except IndexError:
            return 0
        return depth

    def _get_depth(self, node, depth):
        if node.n_lefts + node.n_rights > 0:
            return max([self._get_depth(child, depth + 1) for child in node.children])
        else:
            return depth

    def _lix_formula(self, sentence: list):
        """
        20 corresponds to "very easy", 60 corresponds to "very hard". Same for English and German.
        Args:
            sentence:

        Returns:

        """
        sentence = " ".join(sentence)
        return textstat.lix(sentence)
        # old implementation
        # # words longer than six chars:
        # long_words = [w for w in sentence if len(w) > 6]
        # try:
        #     return np.round(len(sentence) + 100 * len(long_words) / len(sentence), 3)
        # except ZeroDivisionError:
        #     print("Empty sentence evaluated")
        #     return 0

    def _smog_grade(self, sentence: list):
        """
        the higher the more difficult
        Args:
            sentence:

        Returns:

        """
        sentence = " ".join(sentence)
        return textstat.smog_index(sentence)
        # # polysyllables have more than two syllables
        # polys = [token for token in self.nlp(" ".join(sentence)) if
        #          token._.syllables_count is not None and token._.syllables_count >= 3]
        # return np.round(1.0430 * np.sqrt(len(polys) * (30 / 1) + 3.1291), 3)

    def _gunning_fog_index(self, sentence: list):
        """
        Index to estimate the years of formal education needed to understand a given text. 17 corresponds to college
        graduate, 6 corresponds to sixth grade. Originally only created for English.
        Args:
            sentence:

        Returns:

        """
        sentence = " ".join(sentence)
        return textstat.gunning_fog(sentence)
        # old implementation
        # # complex words have more than two syllables, disregarding common suffixes as syllables
        # # ignores compound nouns for now
        # common_suffixes = ["ing", "ed", "es"]
        # complex_words = []
        # for token in self.nlp(" ".join(sentence)):
        #     # skip proper nouns for complex word count
        #     if token.pos_ == "PROPN":
        #         continue
        #
        #     sylls = token._.syllables
        #     # if token has no syllables, sylls == [], skip token
        #     if not sylls:
        #         continue
        #     # ignore "common (verb) suffixes" for syllable count
        #     if token.pos_ == "VERB" and sylls[-1] in common_suffixes:
        #         sylls = sylls[:-1]
        #
        #     if len(sylls) >= 3:
        #         complex_words.append(token)
        # try:
        #     return np.round(0.4 * ((len(sentence) / 1) + 100 * (len(complex_words) / len(sentence))), 3)
        # except ZeroDivisionError:
        #     return 0

    def _coleman_liau_index(self, sentence: list):
        """
        The higher the more difficult the text. No German equivalent.
        Args:
            sentence:

        Returns:

        """
        sentence = " ".join(sentence)
        return textstat.coleman_liau_index(sentence)

        # old implementation
        # chars = sum([len(w) for w in sentence])
        # try:
        #     return np.round(0.0588 * (chars / len(sentence) * 100) - 0.296 * 1 * 100 - 15.8, 3)
        # except ZeroDivisionError:
        #     return 0

    def _automated_readibility_index(self, sentence: list):
        """
        1 corresponds to Kindergarten level, 14 corresponds to college student
        Args:
            sentence:

        Returns:

        """
        sentence = " ".join(sentence)
        return textstat.automated_readability_index(sentence)
        # old implementation
        # chars = sum([len(w) for w in sentence])
        # try:
        #     return np.round(4.71 * (chars / len(sentence)) + 0.5 * (len(sentence) / 1) - 21.43, 3)
        # except ZeroDivisionError:
        #     return 0

    # def flesch_kincaid_score(self, sentence: list):
    #     """
    #     FLESCH READING EASE SCORE
    #     Originally from Flesch, R. (1948). A new readability yardstick. Journal of applied psychology, 32(3), 221.,
    #     adapted to only consider sentences and for German language according to Amstad, T. (1978). Wie verständlich sind
    #     unsere Zeitungen?. Studenten-Schreib-Service. Garais, E. G. (2011). Web Applications Readability. Journal of
    #     Information Systems & Operations Management, 5(1), 114-120.)
    #
    #     100-90 corresponds to 5th grade, 10-0 corresponds to professional/ university graduates
    #     Args:
    #         sentence:
    #
    #     Returns:
    #
    #     """
    #     # syllable count
    #     sylls = [token._.syllables_count if token._.syllables_count is not None else 0 for token in
    #              self.nlp(" ".join(sentence))]
    #     try:
    #         if self.lang == "en":
    #             fks = np.round(206.835 - 1.015 * (len(sentence) / 1) - 84.6 * (sum(sylls) / len(sentence)), 3)
    #         elif self.lang == "de":
    #             fks = np.round(180 - (len(sentence) / 1) - 58.5 * (sum(sylls) / len(sentence)), 3)
    #         else:
    #             raise AttributeError(f"Choose lang from ['en', 'de'], not {self.lang}")
    #         return fks, sylls
    #     except ZeroDivisionError:
    #         return 0

    def _flesch_reading_ease(self, sentence: list):
        """
        FLESCH READING EASE SCORE
        Originally from Flesch, R. (1948). A new readability yardstick. Journal of applied psychology, 32(3), 221.,
        adapted to only consider sentences and for German language according to Amstad, T. (1978). Wie verständlich sind
        unsere Zeitungen?. Studenten-Schreib-Service. Garais, E. G. (2011). Web Applications Readability. Journal of
        Information Systems & Operations Management, 5(1), 114-120.)

        100-90 corresponds to 5th grade, 10-0 corresponds to professional/ university graduates
        Args:
            sentence:

        Returns:

        """
        sentence = " ".join(sentence)
        return textstat.flesch_reading_ease(sentence)
        # old implementation
        # syllable count
        # sylls = [token._.syllables_count if token._.syllables_count is not None else 0 for token in
        #          self.nlp(" ".join(sentence))]
        # try:
        #     if self.lang == "en":
        #         fks = np.round(206.835 - 1.015 * (len(sentence) / 1) - 84.6 * (sum(sylls) / len(sentence)), 3)
        #     elif self.lang == "de":
        #         fks = np.round(180 - (len(sentence) / 1) - 58.5 * (sum(sylls) / len(sentence)), 3)
        #     else:
        #         raise AttributeError(f"Choose lang from ['en', 'de'], not {self.lang}")
        #     return fks
        # except ZeroDivisionError:
        #     return 0


if __name__ == "__main__":
    # changed api? WikiText2 needs torchdata, now
    data = WikiText2(split="train")
    # table = str.maketrans("", "", string.punctuation)
    table = str.maketrans("", "", string.punctuation.replace("<", "").replace(">", ""))
    raw_text = []
    for snippet in data:
        words = snippet.split()
        words = [w.translate(table) for w in words]
        raw_text += list(filter(None, words))

    easy = "He had a guest starring role on the television series The Bill in 2000."
    difficult = "This was followed by a starring role in the play Herons written by Simon Stephens \
                 which was performed in 2001 at the Royal Court Theatre."

    evaluator = SentenceEvaluator(raw_text)
    # res_easy_b = evaluator_b.evaluate(easy)
    res_easy = evaluator.evaluate(easy)
    res_difficult = evaluator.evaluate(difficult)

    print("-" * 15)

    print(f"easy2 b: {res_easy}")
    print(f"difficult b: {res_difficult}")
