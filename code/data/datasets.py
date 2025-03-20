import os
import time

import sklearn.svm
import spacy
import string
import pickle
from pathlib import Path
from tqdm import tqdm
from typing import Union

import random
import unidecode
import numpy as np
from collections import Counter

import torch
from torch import Tensor
from torch.utils import data
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


from training.metrics import SentenceEvaluator
from data.load_data import get_articles_with_level, read_articles, prep_text

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

ALL_DATASETS = ["OSE", "SimpleWiki", "SimpleGerman"]
PATHS = {
    "OSE": "/home/iailab34/selbacht0/Sync/datasets/OneStopEnglishCorpus/data/",
    "Simple": "/home/iailab34/selbacht0/Sync/datasets/SimpleWikipedia/data/",
    "SimpleGerman": "/home/iailab34/selbacht0/Sync/datasets/SimpleGerman/Datasets/"
}


FILES = {"OSE": "ready_ose.pkl",
         "Simple": "ready_simple.pkl",
         "SimpleGerman": "ready_sgerman.pkl"}


class BaseDataset(data.Dataset):
    def __init__(self, split: str = "train", split_ratio: tuple = (0.8, 0.1, 0.1), seed: int = 42,
                 parallel: bool = False, debugging: bool = False):
        """

        Args:
            split: either train, val, dev
            split_ratio: train, val, dev ratio of entire dataset
            seed: random seed for data split
            parallel: start parallel creation of dataset or not
            dev: for debugging only - create dataset of 20 samples
        """
        self.split = split
        self.train_split = split_ratio[0]
        self.val_split = split_ratio[1]
        self.dev_split = split_ratio[2]
        self.seed = seed
        self.parallel = parallel
        self.debugging = debugging

        # init in subclass, but needed in parent methods -> better way to do this?
        self.data = None
        self.all_sentences = None
        self.diff_levels = None
        self.level = None

        self.nlp = spacy.load("en_core_web_trf")
        self.nlp.add_pipe('sentencizer')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item) -> dict:
        return self.data[item]

    def _get_sentences(self, candidates: list):
        all_sents = []
        doc = self.nlp(candidates, disable=['parser', 'tagger', 'ner', 'attribute_ruler'])
        if len(doc):
            sents = [str(s) for s in doc.sents]
            for s in sents:
                line = basic_preprocessing(s)
                line = line.translate(str.maketrans('', '', string.punctuation))
                line = [i for i in line.split(' ') if i != '']
                all_sents.append(line)
        return all_sents

    def _make_raw_corpus(self) -> list:
        stream = []
        for entry in self.all_sentences:
            stream += entry["sentence"]
        return stream

    def _filter_by_level(self, sentences: list) -> list:
        """
        Function to filter sentences of the specified reading level; returns again a list of dictionaries but with
        only sentences that match the reading level
        """
        filtered = []
        for s in sentences:
            if s["level"] == self.diff_levels[self.level]:
                filtered.append(s)
        return filtered


class SimpleWikiDataset(BaseDataset):
    def __init__(self, path: Union[str, None] = None, level: str = None, *args, **kwargs):
        """
        Simple Wikipedia dataset; either loaded from file or created during init. List of dictionaries per data sample
        with entries for sentence, difficulty level and metric names.
        Args:
            path:
        """
        super().__init__(*args, **kwargs)
        self.path = Path(PATHS["Simple"], FILES["Simple"]) if not path else path
        self.diff_levels = {"simple": "simple", "normal": "normal"}
        self.name = "SimpleWiki"
        try:
            with open(self.path, "rb") as f:
                self.all_sentences = pickle.load(f)
        except FileNotFoundError:
            self.all_sentences = self._create_simple_wiki_dataset(parallel=self.parallel)

        random.Random(self.seed).shuffle(self.all_sentences)

        train_index = int(len(self.all_sentences) * self.train_split)
        val_index = int(len(self.all_sentences) * self.val_split)

        train_set = self.all_sentences[:train_index]
        val_set = self.all_sentences[train_index:train_index + val_index]
        dev_set = self.all_sentences[train_index + val_index:]

        if self.split == "train":
            self.data = train_set if not self.level else self._filter_by_level(train_set)
        elif self.split == "val":
            self.data = val_set if not self.level else self._filter_by_level(val_set)
        elif self.split == "dev":
            self.data = dev_set if not self.level else self._filter_by_level(dev_set)
        else:
            print(f"Split should be either ['test', 'val', 'dev'], but is {self.split}")

    def _create_simple_wiki_dataset(self, parallel: bool) -> list:
        tic = time.perf_counter()

        try:
            with open(Path(PATHS["Simple"], "only_sents_simple.pkl"), "rb") as f:
                sentences = pickle.load(f)
            print(f"{len(sentences)} sentences loaded from {Path(PATHS['Simple'], 'only_sents_simple.pkl')}.")

        except FileNotFoundError:
            simple, normal = [], []
            with open(Path(PATHS["Simple"], "simple.aligned"), "r") as fs, open(Path(PATHS["Simple"], "normal.aligned"), "r") as fn:
                for ls, ln in zip(fs, fn):
                    simple.append(ls.strip().split("\t")[-1])
                    normal.append(ln.strip().split("\t")[-1])

            if self.debugging:
                simple = simple[:10]
                normal = normal[:10]

            sentences = []
            # Python 3.8 assignment expression
            for entry in (pbar := tqdm(simple)):
                pbar.set_description("Going over all simple sentences.")
                sents = self._get_sentences(entry)
                # might be multiple sentences:
                for s in sents:
                    sentences.append({"sentence": s, "level": "simple"})

            for entry in (pbar := tqdm(normal)):
                pbar.set_description("Going over all normal sentences.")
                sents = self._get_sentences(entry)
                # might be multiple sentences
                for s in sents:
                    sentences.append({"sentence": s, "level": "normal"})

            with open(Path(PATHS["Simple"], "only_sents_simple.pkl"), "wb") as f:
                pickle.dump(sentences, f)

        # create raw corpus for init of Evaluator
        raw_corpus = []
        for entry in sentences:
            raw_corpus += entry["sentence"]

        # use Evaluator to add "metric_names" entry to each dict with the results of the different sentence metric_names
        evaluator = SentenceEvaluator(raw_corpus)
        random.Random(self.seed).shuffle(sentences)
        if self.debugging:
            sentences = sentences[:20]

        parallel = 20 if parallel else None
        # big work happens here:
        results = evaluator.evaluate(sentences, max_workers=parallel)
        toc = time.perf_counter()
        print(f"Overall time for metric calculation {(toc - tic) / 60:0.4f}")

        with open(Path(PATHS["Simple"], "ready_simple.pkl"), "wb") as f:
            pickle.dump(results, f)

        return results


class OSEDataset(BaseDataset):
    def __init__(self, path: Union[str, None] = None, level: str = None, *args, **kwargs):
        """
        One Stop English dataset; either loaded from file or created during initialisation. List of dictionaries per
        data sample with entries for sentence, difficulty level and metric_names
        Args:
            path: absolute path to the dataset on disk
        """
        super().__init__(*args, **kwargs)
        self.path = Path(PATHS["OSE"], FILES["OSE"]) if not path else path
        self.name = "OSE"
        self.folder_name = {"ele": "Ele-Txt", "int": "Int-Txt", "adv": "Adv-Txt"}
        self.diff_levels = {"ele": "Elementary", "int": "Intermediate", "adv": "Advanced"}

        try:
            with open(self.path, "rb") as f:
                self.all_sentences = pickle.load(f)
        except FileNotFoundError:
            print("No prepared dataset found.")
            self.all_sentences = self._create_ose_dataset()

        random.Random(self.seed).shuffle(self.all_sentences)

        train_index = int(len(self.all_sentences) * self.train_split)
        val_index = int(len(self.all_sentences) * self.val_split)

        train_set = self.all_sentences[:train_index]
        val_set = self.all_sentences[train_index:train_index+val_index]
        dev_set = self.all_sentences[train_index+val_index:]

        if self.split == "train":
            self.data = train_set if not self.level else self._filter_by_level(train_set)
        elif self.split == "val":
            self.data = val_set if not self.level else self._filter_by_level(val_set)
        elif self.split == "dev":
            self.data = dev_set if not self.level else self._filter_by_level(dev_set)
        else:
            print(f"Split should be either ['test', 'val', 'dev'], but is {self.split}")

    # def _evaluate(self):
    #     raw_corpus = self._make_raw_corpus()
    #     evaluator = SentenceEvaluator(raw_corpus)
    #     for entry in tqdm(self.all_sentences):
    #         metrics = evaluator.evaluate(entry["sentence"])
    #         entry["metric_names"] = metrics

    def _create_ose_dataset(self) -> list:
        """
        Reads sentences from file and saves them to a list of dictionaries with the sentence and its respective
        difficulty level.
        """
        tic = time.perf_counter()
        ose_paths = self._get_paths()

        try:
            with open(Path(PATHS["OSE"], "only_sents_ose.pkl"), "rb") as f:
                sentences = pickle.load(f)
            print(f"{len(sentences)} sentences loaded from {Path(PATHS['OSE'], 'only_sents_ose.pkl')}.")

        except FileNotFoundError:
            # create initial list of data points (dicts) entries "sentence" and (difficulty-)"level"
            sentences = []
            if self.debugging:
                ose_paths = ose_paths[:2]
            for file_path in (pbar := tqdm(ose_paths)):
                pbar.set_description("Going through each file.")
                temp = []
                with open(file_path, "r") as f:
                    all_lines = []
                    for line in f:
                        all_lines.append(line.strip())

                    # spacy sentencizer is quite slow; this is supposed to slim down the pipeline
                    for doc in self.nlp.pipe(all_lines, disable=['parser', 'tagger', 'ner', 'attribute_ruler']):
                        # some lines might be empty
                        if len(doc) > 0:
                            sents = [str(s) for s in doc.sents]
                            for s in sents:
                                line = basic_preprocessing(s)
                                line = line.translate(str.maketrans('', '', string.punctuation))
                                line = [i for i in line.split(' ') if i != '']
                                temp.append(line)
                    # first line indicates difficulty level
                    for s in temp[1:]:
                        sentences.append({"sentence": s, "level": temp[0][0]})

            with open(Path(PATHS["OSE"], "only_sents_ose.pkl"), "wb") as f:
                pickle.dump(sentences, f)

        # create raw corpus for init of Evaluator
        raw_corpus = []
        for entry in sentences:
            raw_corpus += entry["sentence"]

        # use Evaluator to add "metric_names" entry to each dict with the results of the different sentence metric_names
        evaluator = SentenceEvaluator(raw_corpus)
        random.Random(self.seed).shuffle(sentences)
        if self.debugging:
            sentences = sentences[:20]

        parallel = 20 if self.parallel else None
        # big work happens here:
        results = evaluator.evaluate(sentences, max_workers=parallel)

        with open(Path(PATHS["OSE"], "ready_ose.pkl"), "wb") as f:
            pickle.dump(results, f)

        toc = time.perf_counter()
        print(f"Overall time for dataset creation {(toc - tic) / 60:0.4f}")

        return results

    def _get_paths(self) -> list:
        ose_paths = []
        for root, dirs, files in os.walk(os.path.join(PATHS["OSE"], "Texts-SeparatedByReadingLevel")):
            for f in files:
                if f.endswith("txt"):
                    ose_paths.append(os.path.join(root, f))

        return ose_paths
    #
    # def get_stream(self) -> str:
    #     # what's the difference to _make_raw_corpus()?
    #     """
    #     Returns a stream of space seperated words.
    #     """
    #     ose_paths = self._get_paths()
    #     if self.level:
    #         ose_paths = [p for p in ose_paths if self.folder_name[self.level] in p]
    #
    #     stream = ""
    #     for file_path in ose_paths:
    #         with open(file_path, "r") as f:
    #             for line in f:
    #                 line = line.strip().translate(str.maketrans('', '', string.punctuation))
    #                 if len(line) > 0:
    #                     line = basic_preprocessing(line)
    #                     stream += line + " "
    #
    #     return stream


class SimpleGermanDataset(BaseDataset):
    def __init__(self, path: Union[str, None] = None, level: str = None, split: str = "train", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = Path(PATHS["SimpleGerman"], FILES["SimpleGerman"]) if not path else path
        self.name = "SimpleGerman"
        self.diff_levels = {"AS": "AS", "ES": "ES", "LS": "LS"}
        self.split = split

        try:

            with open(self.path, "rb") as f:

                self.all_sentences = pickle.load(f)
        except FileNotFoundError:
            print("No prepared dataset found, starting dataset creation.")
            self.all_sentences = self._create_simple_german_dataset()

        random.Random(self.seed).shuffle(self.all_sentences)

        train_index = int(len(self.all_sentences) * self.train_split)
        val_index = int(len(self.all_sentences) * self.val_split)

        train_set = self.all_sentences[:train_index]
        val_set = self.all_sentences[train_index:train_index + val_index]
        dev_set = self.all_sentences[train_index + val_index:]

        if self.split == "train":
            self.data = train_set if not self.level else self._filter_by_level(train_set)
        elif self.split == "val":
            self.data = val_set if not self.level else self._filter_by_level(val_set)
        elif self.split == "dev":
            self.data = dev_set if not self.level else self._filter_by_level(dev_set)
        else:
            print(f"Split should be either ['test', 'val', 'dev'], but is {self.split}")

    def _create_simple_german_dataset(self):
        tic = time.perf_counter()
        try:
            with open(Path(PATHS["SimpleGerman"], FILES["SimpleGerman"]), "rb") as f:
                sentences = pickle.load(f)
            print(f"{len(sentences)} sentences loaded from {Path(PATHS['SimpleGerman'], FILES['SimpleGerman'])}.")

        except FileNotFoundError:
            sentences = []
            # get tuples of (article, level)
            all_article_paths = get_articles_with_level(PATHS['SimpleGerman'])
            for path, level in tqdm(all_article_paths):
                with open(path, "r", encoding="utf-8") as f:
                    # read article and get its sentences
                    sents = prep_text(f.read())
                    # iterate over all sentences and put the into dicts
                    for s in sents:
                        new_s = basic_preprocessing(s)
                        new_s = new_s.translate(str.maketrans('', '', string.punctuation))
                        new_s = [i for i in new_s.split(' ') if i != '']
                        # all sentences here have the level of the article
                        sentences.append({"sentence": new_s, "level": level})

        with open(Path(PATHS["SimpleGerman"], "only_sents_sgerman.pkl"), "wb") as f:
            pickle.dump(sentences, f)

        toc = time.perf_counter()
        print(f"Overall time for sentence parsing {(toc - tic) / 60:0.4f}")

        # create raw corpus for init of Evaluator
        raw_corpus = []
        for entry in sentences:
            raw_corpus += entry["sentence"]

        # use Evaluator to add "metric_names" entry to each dict with the results of the different sentence metric_names
        evaluator = SentenceEvaluator(raw_corpus, lang="de")
        random.Random(self.seed).shuffle(sentences)
        if self.debugging:
            sentences = sentences[:20]

        parallel = 20 if self.parallel else None
        # big work happens here:
        results = evaluator.evaluate(sentences, max_workers=parallel)

        with open(Path(PATHS["SimpleGerman"], FILES["SimpleGerman"]), "wb") as f:
            pickle.dump(results, f)

        toc = time.perf_counter()
        print(f"Overall time for dataset creation {(toc - tic) / 60:0.4f}")

        return results


class CurriculumDataset(data.Dataset):
    def __init__(self, data_samples: list, split: str, input_length: int, seed: int = 42,
                 split_dist: tuple = (0.75, 0.15, 0.05)):
        raise NotImplementedError
        # enumerate data_samples with id as key
        samples = {num: items for num, items in enumerate(data_samples)}

        # set random seed for torch and/ or numpy
        np.seed(seed)
        rng = np.random.default_rng()

        # do dataset split here
        num_datapoints = len(samples)

        train_size, val_size = int(num_datapoints*split_dist[0]), int(num_datapoints*split_dist[1])

        indices_samples = list(range(len(samples)))
        rng.shuffle(indices_samples)
        training_indices = indices_samples[:train_size]
        validation_indices = indices_samples[train_size:val_size]
        test_indices = indices_samples[val_size:]

        # get stream of tokens

        # create Evaluator
        # evaluate each sentence

        # do text preprocessing here
        # since I do have single sentences now: add start and end markers?

        # padding of sentences to equal length

        # do text to token preprocessing here

        pass

    def __len__(self):
        # return length of the dataset after split
        pass

    def __getitem__(self, item):
        # return item using sample id as key in dict
        pass

    def get_stream(self):
        # return stream of words as list to generate corpus for n-grams
        pass


# TODO: split into sentences and apply padding according to input length
# TODO: create Datasets in such a way that __getitem__ returns (x0,y0)
class WikiDataset(data.Dataset):
    def __init__(self, split: str, tokenizer: str = "basic_english"):
        split_iter = WikiText2(split=split)
        tokenizer = get_tokenizer(tokenizer)
        self.vocab = build_vocab_from_iterator(map(tokenizer, split_iter), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])

        raw_text_iter = WikiText2(split=split)
        dataset = [torch.tensor(self.vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        # at this point just a long tensor of continuous tokens
        self.final_dataset = torch.cat(tuple(filter(lambda t: t.numel() > 0, dataset)))

    def __len__(self):
        """
        Returns: the number of words in the dataset
        """
        return self.final_dataset.size()[0]

    def __getitem__(self, idx):
        """
        For later: when dealing with bigger datasets, then use this method to load the data sample from file. Assumes
        that IDs for training and test samples are provided.
        Returns: the token of a single word in the dataset
        """
        return self.final_dataset[idx]

    def to_text(self, data):
        # TODO: maybe move this to single function so that it can be used in any class
        if isinstance(data, int):
            try:
                return self.vocab.lookup_token(data)
            except RuntimeError:
                print(f"Index {data} is not part of the vocabulary (length: {len(self.vocab)})")
        elif isinstance(data, list):
            try:
                return self.vocab.lookup_tokens(data)
            except RuntimeError:
                print(f"Some index in the list {data} is not part of the vocabulary (length: {len(self.vocab)}.")
        else:
            print(f"{data} is neither a single token (int) nor a list of tokens.")


def basic_preprocessing(stream: str):
    # replacing multiple whitespaces by single whitespace DONE BY TOKENIZER; if "basic_english"
    # pattern = re.compile(r'\s+')
    # text = re.sub(pattern, ' ', stream)

    # replacing accented characters with ascii characters
    text = unidecode.unidecode(stream)

    return text


def predict(X, m, c):
    return np.append([1, X]) @ np.append(m, c)


def get_classification_xy(dataset, balanced: bool = True, level: list = None, return_sents: bool = False):
    """
    Returns features and labels for sentence classification.
    Args:
        dataset:
        balanced: naive implementation to create class balance (discarding samples from overrepresented classes)
        level: choose level from the according dataset to get only samples from them
        return_sents: return list of sentences at corresponding index

    Returns:

    """
    if dataset.name == 'OSE':
        level_to_label = {"Elementary": 1, "Intermediate": 2, "Advanced": 3}
    elif dataset.name == "SimpleWiki":
        level_to_label = {"simple": 1, "normal": 3}
    elif dataset.name == "SimpleGerman":
        level_to_label = {"AS": 3, "LS": 2, "ES": 1}
    else:
        raise NotImplementedError

    level = level if level else list(level_to_label.values())
    X = []
    y = []
    sentences = []
    for entry in dataset:
        if level_to_label[entry["level"]] in level:
            # data = [v for v in entry["metrics"].values()]

            data = list(entry["metrics"].values())
            X.append(data)
            y.append(level_to_label[entry["level"]])
            if return_sents:
                sentences.append(entry["sentence"])
        else:
            continue
    X, y = np.asarray(X), np.asarray(y)

    if balanced:
        min_number_samples = min(Counter(y).values())
        balanced_X = []
        balanced_y = []
        balanced_sentences = []
        for l in level:
            balanced_X.append(X[y == l][:min_number_samples])
            balanced_y.append(y[y == l][:min_number_samples])
            if return_sents:
                balanced_sentences += [sentences[int(i)] for i in np.asarray(y == l).nonzero()[0]]

        if balanced and return_sents:
            sentences = balanced_sentences
        X, y = np.concatenate(np.asarray(balanced_X)), np.concatenate(np.asarray(balanced_y))
        assert len(X) == len(
            y), f"len(balanced_X) = {len(X)} and len(balanced_y) = {len(y)}"
        assert len(X) == min_number_samples * len(
            level), f"{len(X)} is not {min_number_samples * len(level)}"
        shuffle = np.random.permutation(len(X))
        X, y = X[shuffle], y[shuffle]
        if return_sents:
            shuffled_sentences = [sentences[i] for i in shuffle]
            return X, y, shuffled_sentences
        else:
            return X, y
    else:
        if return_sents:
            return X, y,sentences


def quick_fix(dataset):
    """
    For datasets where the last entry (flesch_kincaid) accidentally also includes the list of syllables
    Args:
        dataset:

    Returns:

    """
    fixed_dataset = []
    for entry in dataset:
        if isinstance(entry[-1], tuple):
            new_entry = list(entry[:-1])
            new_entry.append(entry[-1][0])
            fixed_dataset.append(np.array(new_entry))
        elif isinstance(entry[-1], int) or isinstance(entry[-1], float):
            fixed_dataset.append(entry)
        else:
            print("Something's gone awry")
    return np.array(fixed_dataset)


def create_new_labels(ocsvm: sklearn.svm.OneClassSVM, samples: list, inlier: int, outlier: int):
    """
    Method that uses a fitted OneClassSVM to discern between sentences of label inlier and outlier
    Args:
        ocsvm: OneClassSVM
        samples: list of data samples (already transformed)
        inlier: label of classes that was used to fit the OneClassSVM
        outlier: label of the class that should be discerned from class fitted_on

    Returns:
        np.array of labels
    """
    new_labels = ocsvm.predict(samples)
    new_labels[new_labels == 1] = inlier
    new_labels[new_labels == -1] = outlier
    return new_labels


if __name__ == "__main__":
    simple_dev = SimpleWikiDataset(split="dev")
    ose_dev = OSEDataset(path="/home/iailab32/toborek/datasets/OneStopEnglishCorpus/data/ready_ose_updated_2023-02-07-18:01:43.pkl", split="dev")
    ose_val = OSEDataset(path="/home/iailab32/toborek/datasets/OneStopEnglishCorpus/data/ready_ose_updated_2023-02-07-18:01:43.pkl", split="val")
