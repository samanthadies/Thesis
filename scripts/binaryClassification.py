import pandas as pd
import warnings

BASE_FP = '../data/'
BASE_FP_OUTPUT_I = '../output/eda/idealized/'
BASE_FP_OUTPUT_CS = '../output/eda/cs/'
BASE_FP_OUTPUT_PH = '../output/eda/ph/'
BASE_FP_OUTPUT_RESEARCH = '../output/eda/researchers/'
BASE_FP_OUTPUT_VISUALIZATIONS = '../output/visualizations/'


########################################################################################################
# main
# Inputs: none
# Return: none
# Description:
########################################################################################################
def main():

    warnings.filterwarnings('ignore')

    # open file
    df = pd.read_csv(f'{BASE_FP}ethics_section.csv')

    text = df['Text']
    print(text.head())

    # use bag of words (1,2,3-grams) and try clustering (knn and set 2 clusters?), compare
    # clusters with scores and see if there's a majority label/if the clusters make sense

    # use topic modeling to identify topics/words

    # assuming either go ok, see if i can pick up on patterns with mislabeled papers
    # repeat with the other criteria


if __name__ == '__main__':
    main()
