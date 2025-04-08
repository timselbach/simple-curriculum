
import pandas as pd
import numpy as np
from collections import Counter



def calculate_vocabulary_overlap_with_percentages(vocab_dict, method='jaccard'):
    """
    Calculates the vocabulary overlap between multiple vocabularies

    Args:
        vocab_dict: dictionary with each class and the corresponding set of words in the vocabulary
        method : method to calculate the overlap with 'jaccard' and 'relative_to_cls1' as possible values

    Returns:
        overlap_count_matrix: matrix that contains the number of overlapping words between the classes
        overlap_percentage_matrix: matrix that contains the percentage of overlapping between the classes
    """
    classes = list(vocab_dict.keys())
    overlap_count_matrix = pd.DataFrame(index=classes, columns=classes, dtype=int)
    overlap_percentage_matrix = pd.DataFrame(index=classes, columns=classes, dtype=float)

    for cls1 in classes:
        for cls2 in classes:
            if cls1 == cls2:
                overlap_count_matrix.loc[cls1, cls2] = len(vocab_dict[cls1])
                overlap_percentage_matrix.loc[cls1, cls2] = 100.0  # 100% Selbstüberlappung
            else:
                intersection = vocab_dict[cls1].intersection(vocab_dict[cls2])
                overlap_count = len(intersection)
                overlap_count_matrix.loc[cls1, cls2] = overlap_count

                if method == 'jaccard':
                    union = vocab_dict[cls1].union(vocab_dict[cls2])
                    overlap_percentage = (overlap_count / len(union)) * 100 if len(union) > 0 else 0.0
                elif method == 'relative_to_cls1':
                    overlap_percentage = (overlap_count / len(vocab_dict[cls1])) * 100 if len(vocab_dict[cls1]) > 0 else 0.0
                else:
                    raise ValueError("Ungültige Methode. Verwende 'jaccard' oder 'relative_to_cls1'.")

                overlap_percentage_matrix.loc[cls1, cls2] = overlap_percentage

    return overlap_count_matrix, overlap_percentage_matrix


def count_sentences_per_class(y, label_mapping=None):
    """
    Prints the number of sentences per class

    Args:
        y : Labels array of length n_samples
        label_mapping :  Dictionary mapping label values to class names

    Returns:
        None
    """
    # Create a DataFrame from your labels
    df_labels = pd.DataFrame({'complexity': y})


    df_labels['complexity_label'] = df_labels['complexity'].map(label_mapping)

    # count the number of sentences per complexity category
    counts = df_labels['complexity_label'].value_counts()

    print("Number of sentences per complexity category:")
    for complexity, count in counts.items():
        print(f"{complexity}: {count} sentences")


def calculate_vocabulary_per_class(y, sents, label_mapping=None):
    """
    Calculates the vocabulary size for each class

    Args:
        y : Labels array of length n_samples
        sents : List of sentences corresponding to the data points
        label_mapping :  Dictionary mapping label values to class names

    Returns:
        tuple:
            - vocab_size:dict: dictionary with each class and the corresponding vocabulary size
            - vocab_dict: dictionary with each class and the corresponding set of words in the vocabulary
    """
    # Set label mapping
    if label_mapping:
        y_mapped = np.array([label_mapping.get(label, label) for label in y])
    else:
        y_mapped = y

    # create vocab dictionary for each class, use set as entires to eliminate duplicates
    vocab_dict = {}
    for label in np.unique(y_mapped):
        vocab_dict[label] = set()

    # iterate over words and add them to respective set
    for label, sent in zip(y_mapped, sents):
        words = sent
        vocab_dict[label].update(word.lower() for word in words)

    # calculate vocabulary size
    vocab_size_dict = {label: len(vocab) for label, vocab in vocab_dict.items()}

    return vocab_size_dict, vocab_dict



def print_n_most_common_words_between_classes(y, sents, class1, class2, n, label_mapping=None):
    """
    Prints the n most common words that are shared between a group of classes (class1) and another class (class2).

    Args:
        y : Labels array of length n_samples
        sents : List of sentences corresponding to the data points
        class1: Identifier or list of identifiers for the first set of classes
        class2: Identifier for the second class
        n: Number of most common shared words to print
        label_mapping :  Dictionary mapping label values to class names

    Returns:
        None
    """
    # Make class1 a list even if it just has one element
    if not isinstance(class1, list):
        class1 = [class1]

    # Map labels
    if label_mapping:
        y_mapped = [label_mapping.get(label, label) for label in y]
    else:
        y_mapped = y

    # Initialize counters
    counter_group = Counter()
    counter2 = Counter()

    # Iterate over sentences and count words for each specified class
    for label, sent in zip(y_mapped, sents):
        words = sent

        # Convert words to lowercase for comparison
        words = [word.lower() for word in words]

        # Update the counter if the label is in the group of classes
        if label in class1:
            counter_group.update(words)
        elif label == class2:
            counter2.update(words)

    # Find the common words between the group and class2
    common_words = set(counter_group.keys()).intersection(set(counter2.keys()))

    # Combine frequencies from both groups for the common words
    combined_freq = {word: counter_group[word] + counter2[word] for word in common_words}

    # Sort the common words by their combined frequency in descending order
    sorted_common_words = sorted(combined_freq.items(), key=lambda x: x[1], reverse=True)

    print(f"The {n} most common words shared between classes {class1} and '{class2}' are:")
    for word, freq in sorted_common_words[:n]:
        print(f"{word}: {freq}")