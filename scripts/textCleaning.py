########################################################################################################
# textCleaning.py
# This program cleans the text training data for the 18 idealized scoring criteria by removing
# punctuation, numbers, and extra white spaces, lemmatizing, lowercasing, and converting the score into
# binary. It also generates unigrams, bigrams, and trigrams from the text data of each criteria.
########################################################################################################


from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import regex as re
import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')
stemmer = WordNetLemmatizer()
tqdm.pandas()

BASE_FP = '../data/training/'


########################################################################################################
# remove_punctuation
# Inputs: row
# Return: row
# Description: removes all non-alhpabetic, non-space characters from the text input
########################################################################################################
def remove_punctuation(row):
    regex = re.compile('[^a-zA-Z ]')
    row = regex.sub('', row)
    return row


########################################################################################################
# remove_extra_spaces
# Inputs: row
# Return: row
# Description: removes all extra white spaces beyond the word delimiters
########################################################################################################
def remove_extra_spaces(row):
    whitespace_pattern = r'\s+'
    row = re.sub(whitespace_pattern, ' ', row)
    row = row.strip()
    return row


########################################################################################################
# lemmatize_text
# Inputs: row
# Return: row
# Description: lemmatizes the input text
########################################################################################################
def lemmatize_text(row):
    row = row.split()
    row = ' '.join([stemmer.lemmatize(word) for word in row])
    return row


########################################################################################################
# generate_n_grams
# Inputs: final, row, ngram
# Return: final
# Description: converts row into ngram ngrams and returns the list of ngrams (final)
# Citation: https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python/
########################################################################################################
def generate_n_grams(final, row, ngram=1):
    # creat list of flood words
    en_stop = stopwords.words('english')
    # en_stop.extend(['introduction', 'abstract', 'conclusion', 'discussion', 'method', 'result', 'data', 'mental', 'health'])

    grams = [gram for gram in row.split(" ") if gram not in set(en_stop)]
    temp = zip(*[grams[i:] for i in range(0, ngram)])
    ngrams = [' '.join(ngram) for ngram in temp]
    final.append(ngrams)

    return final


########################################################################################################
# clean_text
# Inputs: row
# Return: row
# Description: calls other cleaning methods on the inputted row
########################################################################################################
def clean_text(row):
    row = remove_punctuation(row)
    row = row.lower()
    row = remove_extra_spaces(row)
    row = lemmatize_text(row)
    return row


########################################################################################################
# convert_to_binary
# Inputs: df
# Return: df
# Description: converts the scores into binary, where 0 means absent and 1 means present
########################################################################################################
def convert_to_binary(df):
    df.loc[df['Score'] != 0, 'Score'] = 1
    return df


########################################################################################################
# preprocess_text
# Inputs: df
# Return: df['Cleaned'], df['unigrams'], df['bigrams'], df['trigrams']
# Description: cleans each row of data from the dataframe, generates unigrams, bigrams, and trigrams,
# and returns new columns for the cleaned and tokenized text
########################################################################################################
def preprocess_text(df):
    print(type(df))
    df['Cleaned'] = df.progress_apply(lambda x: clean_text(x['Text']), axis=1, result_type='expand')

    final = []
    df.progress_apply(lambda x: generate_n_grams(final, x['Cleaned'], 1), axis=1, result_type='expand')
    df['unigrams'] = final

    final.clear()
    df.progress_apply(lambda x: generate_n_grams(final, x['Cleaned'], 2), axis=1, result_type='expand')
    df['bigrams'] = final

    final.clear()
    df.progress_apply(lambda x: generate_n_grams(final, x['Cleaned'], 3), axis=1, result_type='expand')
    df['trigrams'] = final

    df = convert_to_binary(df)

    return df['Cleaned'], df['unigrams'], df['bigrams'], df['trigrams']


########################################################################################################
# main
# Inputs: none
# Return: none
# Description: cleans the text training data for the 18 idealized scoring criteria by removing
# punctuation, numbers, and extra white spaces, lemmatizing, lowercasing, and converting the score into
# binary. Also generates unigrams, bigrams, and trigrams for each of the criteria's documents.
########################################################################################################
def main():

    warnings.filterwarnings('ignore')
    # criteria = ['ethics_section']
    criteria = ['anonymity', 'class_collection', 'data_public', 'data_source', 'dem_dist', 'drop', 'ethics_section',
                'feature_reconstruction', 'ground_truth_discussion', 'ground_truth_size', 'informed_consent',
                'irb', 'limitations', 'missing_values', 'noise', 'random_sample', 'replication', 'text']

    for item in criteria:
        CRITERIA_FP_READ = BASE_FP + item + "/" + item + ".csv"
        print("\nWORKING WITH " + item + "!!!!!!\n")

        df = pd.read_csv(f'{CRITERIA_FP_READ}')
        df = df.replace(np.nan, '', regex=True)

        if item != "ethics_section":
            if "Text 2" in df.columns:
                df['Text'] = df['Text'] + df['Text 2']
                del df['Text 2']
            if "Text 3" in df.columns:
                df['Text'] = df['Text'] + df['Text 3']
                del df['Text 3']
            if "Text 4" in df.columns:
                df['Text'] = df['Text'] + df['Text 4']
                del df['Text 4']

        df_for_binary = df[['Link to paper', 'Score', 'Text']]

        df_for_binary['Cleaned'], df_for_binary['unigrams'], df_for_binary['bigrams'], df_for_binary['trigrams'] = preprocess_text(df_for_binary)

        CRITERIA_FP_WRITE = BASE_FP + item + "/" + item + "_CLEANED.csv"
        df_for_binary.to_csv(f'{CRITERIA_FP_WRITE}', index=False)

    # try clustering (knn and set 2 clusters?), compare clusters with scores and see if there's a majority
    # label/if the clusters make sense

    # assuming either go ok, see if i can pick up on patterns with mislabeled papers
    # repeat with the other criteria


if __name__ == '__main__':
    main()
