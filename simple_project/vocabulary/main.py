from simple_project.vocabulary.utils import count_sentences_per_class, print_n_most_common_words_between_classes
from simple_project.data.all_datasets import SimpleWikiDataset, SimpleGermanDataset, get_classification_xy
from simple_project.vocabulary.plots import plot_metric_distribution_by_complexity, plot_metric_correlation_heatmap
from simple_project.vocabulary.preprocess import filter_by_sentence_length
from simple_project.vocabulary.utils import calculate_vocabulary_per_class, calculate_vocabulary_overlap_with_percentages

# load datasets
swiki_all = SimpleWikiDataset(split="all")
X_swiki_all, y_swiki_all, sents_swiki_all = get_classification_xy(swiki_all, balanced=False, return_sents=True)

sger_all = SimpleGermanDataset(split="all")
X_sger_all, y_sger_all, sents_sger_all = get_classification_xy(sger_all, balanced=False, return_sents=True)


# filter by length
metric_names_wiki = list(swiki_all[0]["metrics"].keys())
metric_names_ger = list(sger_all[0]["metrics"].keys())
label_mapping_wiki = {1: 'SL', 3: 'EL'}
label_mapping_ger = {1: 'Simple (ES)',2: "Simple (LS)", 3: 'Everyday'}


X_swiki_filtered, y_swiki_filtered, sents_swiki_filtered = filter_by_sentence_length(
    X=X_swiki_all,
    y=y_swiki_all,
    sents=sents_swiki_all,
    metric_names=metric_names_wiki,
    label_mapping=label_mapping_wiki,
    min_length=4,
    max_length=300,
    n_print=0
)



X_sger_filtered, y_sger_filtered, sents_sger_filtered = filter_by_sentence_length(
    X=X_sger_all,
    y=y_sger_all,
    sents=sents_sger_all,
    metric_names=metric_names_ger,
    label_mapping=label_mapping_ger,
    min_length=4,
    max_length=300,
    n_print=0
)

#count sentences per class
count_sentences_per_class(y_swiki_filtered,label_mapping_wiki)

count_sentences_per_class(y_sger_filtered,label_mapping_ger)
print()
#calculate vocabulary per class
voc_size_wiki,_ = calculate_vocabulary_per_class(y_swiki_filtered,sents_swiki_filtered,label_mapping_wiki)

voc_size_ger,_ = calculate_vocabulary_per_class(y_sger_filtered,sents_sger_filtered,label_mapping_ger)

print("SimpleWiki voc sizes:", voc_size_wiki)
print("SimpleGerman voc sizes:", voc_size_ger)
print()


#show common words that are used the most between classes
print_n_most_common_words_between_classes(y_swiki_filtered,sents_swiki_filtered, "SL", "EL", 20, label_mapping=label_mapping_wiki)

print_n_most_common_words_between_classes(y_sger_filtered,sents_sger_filtered,["Simple (ES)","Simple (LS)"],"Everyday",20,label_mapping=label_mapping_ger)


#analyse vocabulary overlaps

print("SimpleWiki matrices")
voc_size, voc_dict = calculate_vocabulary_per_class(y_swiki_filtered,sents_swiki_filtered,label_mapping=label_mapping_wiki)
_,matrix = calculate_vocabulary_overlap_with_percentages(voc_dict, method='jaccard')
print(matrix)
_,matrix = calculate_vocabulary_overlap_with_percentages(voc_dict, method='relative_to_cls1')
print(matrix)
print()


print("SimpleGerman matrices")
voc_size, voc_dict = calculate_vocabulary_per_class(y_sger_filtered,sents_sger_filtered,label_mapping=label_mapping_ger)
_,matrix = calculate_vocabulary_overlap_with_percentages(voc_dict, method='jaccard')
print(matrix)
_,matrix = calculate_vocabulary_overlap_with_percentages(voc_dict, method='relative_to_cls1')
print(matrix)


#plot metric correlations
plot_metric_correlation_heatmap(X_swiki_filtered,metric_names=metric_names_wiki)
#plot_metric_correlation_heatmap(X_sger_filtered,metric_names=metric_names_ger)

#plot metric distributions
x_limits = {'flesch_kincaid': (-50, None)}
plot_metric_distribution_by_complexity(X_swiki_filtered, y_swiki_filtered, metric_names_wiki, label_mapping=label_mapping_wiki,bins=50,x_limits=x_limits, display_plot=False, dataset_name="SimpleWiki")

x_limits = {'flesch_kincaid': (-100, None)}
plot_metric_distribution_by_complexity(X_sger_filtered, y_sger_filtered, metric_names_ger, label_mapping=label_mapping_ger,bins=40,x_limits=x_limits,display_plot=False, dataset_name="SimpleGerman")


