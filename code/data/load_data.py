import os
import re
import json
import spacy
from spacy.lang.en import English

DATASETS = ["OSE", "SimpleWiki", "SimpleGerman"]
nlp = spacy.load("de_core_news_lg")


def load_samples(sample_paths: list, dataset: str, level: str):
    """

    Args:
        sample_paths (list): contains strings to all files
        dataset (str): referring to either "OSE" (OneStopEnglish) or "SimpleWiki" dataset
        level (str): for loading of OSE dataset; filtering for specific kind of reading level

    Returns:
        a list of dictionaries with all loaded samples
    """
    all_samples = []
    if dataset == "OSE":
        loading_function = read_OSE_file
    elif dataset == "SimpleWiki":
        loading_function = read_SW_file
    else:
        print(f"{dataset} not supported. Choose from {DATASETS}")
        return

    if type(sample_paths) == str:
        sample_paths = [sample_paths]

    for path in sample_paths:
        all_samples += loading_function(path, level)

    return all_samples


def read_OSE_file(path: str, level=None):
    """
    Function that reads the content of a text file from the One Stop English Corpus into the target format.
    Args
        path (str): path to article, .txt file
        level (str): choose from {advanced, intermediate, elementary}

    returns:
        list of dicts {path, source, sentence, level}
    """

    with open(path, "r") as f:
        content = [line.strip() for line in f]

    article_name = os.path.split(path)[-1]

    nlp = English()
    nlp.add_pipe('sentencizer')

    data_points = []

    # first entry always specifies the reading level, which is also encoded in the filename
    for entry in content[1:]:
        level = content[0].lower()
        doc = nlp(entry)
        result = [sent.text.strip() for sent in doc.sents]
        # if one entry contains more than one sentence, save each sentence as data point with the same meta data
        if len(result) > 1:
            for s in result:
                sample = {"path": path, "source": article_name, "sentence": s, "level": level}
                data_points.append(sample)
        elif len(result) == 1:
            sample = {"path": path, "sentence": result[0], "source": article_name, "level": level}
            data_points.append(sample)
        else:
            continue

    return data_points


def read_SW_file(path: str, level: str):
    """
    path (str): path to article, .aligned file

    returns:
        list of dicts {source, score, all_scores, sentence}
    """

    with open(path, "r") as f:
        content = list(line.strip().split('\t') for line in f)

    nlp = English()
    nlp.add_pipe('sentencizer')

    data_points = []

    for entry in content:
        doc = nlp(entry[2])
        result = [sent.text.strip() for sent in doc.sents]
        # if one line contains more than one sentence, save each sentence as data point with the same meta data
        if len(result) > 1:
            for s in result:
                sample = {"path": path, "source": entry[0], "sentence": s}
                data_points.append(sample)
        else:
            sample = {"path": path, "source": entry[0], "sentence": result[0]}
            data_points.append(sample)

    return data_points


def load_simple_german_dataset(path: str):
    pass


def get_articles_with_level(root_dir: str, source: str | list = None, type: str = None) -> list[
    tuple[str, str]]:
    """ Returns a list of tuples in the form of (article_path, article_level) in the specified directory
    Args:
        root_dir (str, optional): Directory in which to find the articles, potentially nested. Info needs to be given in parsed_header.json files
    Returns:
        list[tuple[str,str]]: list of tuples in the form of (article_path, article_level) in the specified directory
    """
    article_list = []
    source = [source] if isinstance(source, str) else source
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            if name == 'parsed_header.json':
                # if you want to filter by source(s) and the current file is not from that source
                if source and not any(name in root for name in source):
                    continue

                with open(os.path.join(root, name), 'r', encoding="utf-8") as fp:
                    data = json.load(fp)

                for fname in data:
                    if type and data[fname]['type'] != type:
                        continue
                    article_list.append((os.path.join(root, 'parsed/' + fname + '.txt'), data[fname]["type"]))

    return article_list


def read_articles(article_list: list[tuple[str, str]], sentence_split: bool = True) -> list[list[list[str], list[str]]]:
    """
    Takes a list of tuples with (path_to_simple_article, path_to_everyday_language_article) reads the articles'
    contents, performs some additional sentence splitting and returns the texts.
    Args:
        article_list (list[tuple[str,str]]): list of tuples in the form of (easy_article, everyday_article)
    Returns:
        list[list[list[str], list[str]]]: list of article pairs, where the entry for each article pair consists either
        of a list of sentences (if sentence_split==True) or a string of text.
    """
    articles = []
    for simple_path, everyday_path in article_list:
        with open(simple_path, "r", encoding="utf-8") as fs, open(everyday_path, "r", encoding="utf-8") as fe:

            if sentence_split:
                articles.append([prep_text(fs.read()), prep_text(fe.read())])
            else:
                articles.append([fs.read(), fe.read()])

    return articles


def prep_text(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    sents = [str(sent) for sent in nlp(text).sents]
    return sents