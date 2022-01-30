########################################################################################################
# topicModeling.py
# This program identifies topics and labels each document with topic percents for each of the criteria
# using unigrams, bigrams, and trigrams
########################################################################################################

import warnings
import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd
from tqdm import tqdm
import ast
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import warnings
import regex as re
import nltk
nltk.download('wordnet')
stemmer = WordNetLemmatizer()
tqdm.pandas()

BASE_FP = "../data/training/"
BASE_FP_OUTPUT = '../output/topics/training/'
tokenizer = RegexpTokenizer(r'\w+')


########################################################################################################
# get_topic_distribution_of_each_doc
# Inputs: criteria, best_coherence_run, document_topics, text_tokens, ngram
# Return: merged
# Description: Get and form dataframes out of topic distributions for tokens and documents and output to
# CSVs
########################################################################################################
def get_topic_distribution_of_each_doc(criteria, best_coherence_run, document_topics, text_tokens, ngram):
    # Get Cleaned Twitter Data
    CRITERIA_FP_READ = BASE_FP + criteria + "/" + criteria + "_CLEANED.csv"
    cleaned_df = pd.read_csv(f"{CRITERIA_FP_READ}", dtype='str', lineterminator='\n')
    cleaned_df['tokens'] = cleaned_df.progress_apply(lambda x: ast.literal_eval(x['unigrams']), axis=1)
    cleaned_df['tokens'] = cleaned_df['tokens'].astype('str')

    # Create Topic Probability Columns & make topic distribution dataframe
    cols = []
    for x in range(1, best_coherence_run + 1):
        cols.append(str('topic' + str(x)))
    topic_dist_df = pd.DataFrame(document_topics, columns=cols)

    # Parse probability lists to just probabilities
    for x in range(1, best_coherence_run + 1):
        col_name = str('topic' + str(x))
        topic_dist_df[col_name] = topic_dist_df[col_name].str[1]

    # add token column for second dataframe
    topic_dist_df['tokens'] = text_tokens
    topic_dist_df['tokens'] = topic_dist_df['tokens'].astype('str')

    # Merge dataframes and drop duplicates
    merged = cleaned_df.merge(topic_dist_df, how='left', on=['tokens'])
    del merged['tokens']

    # Write document and token topic distributinos to CSVs
    OUTPUT_FILE = BASE_FP_OUTPUT + criteria + "/" + criteria + "_" + str(ngram);
    merged.to_csv(f'{OUTPUT_FILE}_doc_topic_distribution.csv', header=True, index=False)

    return merged


########################################################################################################
# perform_topic_analysis
# Inputs: cleaned, criteria, ngram
# Return: cleaned_with_topics
# Description: Performs topic analysis with multiple number of topics, finds best number of topic based
# on coherence, retrieves topic distributions for best number of topics
########################################################################################################
def perform_topic_analysis(cleaned, criteria, ngram):
    # access tokens
    if ngram == 1:
        cleaned['tokens'] = cleaned.progress_apply(lambda x: ast.literal_eval(x['unigrams']), axis=1)
        text_tokens = cleaned['tokens'].tolist()
    elif ngram == 2:
        cleaned['tokens'] = cleaned.progress_apply(lambda x: ast.literal_eval(x['bigrams']), axis=1)
        text_tokens = cleaned['tokens'].tolist()
    else:
        cleaned['tokens'] = cleaned.progress_apply(lambda x: ast.literal_eval(x['trigrams']), axis=1)
        text_tokens = cleaned['tokens'].tolist()

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(text_tokens)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in text_tokens]

    # Set up lists
    top_words_per_topic = []
    coherence_scores = []
    document_topics = []

    # Perform Topic analysis for different number of topics
    for n_topics in [2, 3, 4, 5, 6, 7, 8]:
        # generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=n_topics, id2word=dictionary, passes=20)

        # Print model
        print(f"LDA MODEL for {n_topics} topics: \n")
        print(ldamodel.print_topics(num_topics=n_topics, num_words=3))

        # Compute Coherence Score (Score nearest to 0 is better)
        coherence_model_lda = CoherenceModel(model=ldamodel, texts=text_tokens, dictionary=dictionary,
                                             coherence='u_mass')
        coherence_lda = round(coherence_model_lda.get_coherence(), 2)
        print('\nCoherence Score: ', coherence_lda)

        # Save top words to csv
        for t in range(ldamodel.num_topics):
            top_words_per_topic.extend([(n_topics, t,) + x for x in ldamodel.show_topic(t, topn=10)])
        coherence_scores.extend([(n_topics, coherence_lda)])

        # Get document topics for best coherence run and fill nulls with zeros
        best_coherence_run = max(coherence_scores, key=lambda item: item[1])[0]
        if n_topics == best_coherence_run:

            document_topics = []
            for item in corpus:
                topic_dist = ldamodel.get_document_topics(item)

                if len(topic_dist) != best_coherence_run:
                    index = 0

                    for i in topic_dist:
                        if (i[0] != index) and (len(topic_dist) != best_coherence_run):
                            topic_dist.insert(index, (index, 0))
                        index += 1

                    while len(topic_dist) < best_coherence_run:
                        topic_dist.append((len(topic_dist), 0))

                document_topics.append(topic_dist)

    # Write Topic Words & Coherence Scores To CSV
    OUTPUT_FILE = BASE_FP_OUTPUT + criteria + "/" + criteria + "_" + str(ngram);
    pd.DataFrame(top_words_per_topic, columns=['n_topics', 'topic', 'word', 'p']).to_csv(f"{OUTPUT_FILE}_top_words.csv",
                                                                                         index=False)
    pd.DataFrame(coherence_scores, columns=['n_topics', 'coherence_score']).to_csv(
        f"{OUTPUT_FILE}_coherence_scores.csv", index=False)

    # Get Topic Distribution of each Document and Token
    best_coherence_run = max(coherence_scores, key=lambda item: item[1])[0]
    cleaned_with_topics = get_topic_distribution_of_each_doc(criteria, best_coherence_run, document_topics, text_tokens,
                                                             ngram)

    return cleaned_with_topics


########################################################################################################
# main
# Inputs: none
# Return: none
# Description: This program identifies topics and labels each document with topic percents for each
# of the criteria using unigrams, bigrams, and trigrams
########################################################################################################
def main():

    warnings.filterwarnings('ignore')
    # criteria = ['ethics_section']
    criteria = ['anonymity', 'class_collection', 'data_public', 'data_source', 'dem_dist', 'drop', 'ethics_section',
                'feature_reconstruction', 'ground_truth_discussion', 'ground_truth_size', 'informed_consent',
                'irb', 'limitations', 'missing_values', 'noise', 'random_sample', 'replication', 'text']

    for item in criteria:
        CRITERIA_FP_READ = BASE_FP + item + "/" + item + "_CLEANED.csv"
        print("\nWORKING WITH " + item + "!!!!!!\n")
        cleaned_df = pd.read_csv(f"{CRITERIA_FP_READ}", dtype='str', lineterminator='\n', keep_default_na=False)
        for i in range(1, 4):
            print(i)
            perform_topic_analysis(cleaned_df, item, i)


# Driver code
if __name__ == "__main__":
    main()
