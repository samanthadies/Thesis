########################################################################################################
# unsupervised.py
# This program creates two clusters for each criteria using tf-idf and kMeans, then classifies each
# document in order to calculate conditional accuracies. This process is repeated 10 times for each
# criteria. Average results for all criteria are written to summary_output.txt, while per-run results
# are writen to 'criteria'_results.csv for each of the criteria.
########################################################################################################


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer
import pandas as pd
import warnings
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')
stemmer = WordNetLemmatizer()
tqdm.pandas()
import os


BASE_FP = '../data/training/'
BASE_FP_OUTPUT = '../output/clustering/training/'


########################################################################################################
# analyze_clusters
# Inputs: results, criteria, file
# Return: score0accuracy, score1accuracy
# Description: determines whether unsupervised clustering accurately classifed documents based on
# withheld scores and writes results to file
########################################################################################################
def analyze_clusters(results, criteria, file):

    print("DF info: ")
    print(results.info())

    print("\nDF value counts: ")
    print(results.value_counts())

    #totalCount = 0

    s0in0 = 0
    s0in0_rows = []
    s0in1 = 0
    s0in1_rows = []
    s1in0 = 0
    s1in0_rows = []
    s1in1 = 0
    s1in1_rows = []

    totalScore0 = 0
    totalScore1 = 0

    for row in results.index:

        # get counts of how many of each score are classified into each cluster
        # note: s0in0 means a score of 0 was classified into cluster 0
        if results['scores'][row] == 0 and results['predictions'][row] ==0:
            s0in0 += 1
            s0in0_rows.append(row)
            totalScore0 += 1
        elif results['scores'][row] == 0 and results['predictions'][row] ==1:
            s0in1 += 1
            s0in1_rows.append(row)
            totalScore0 += 1
        elif results['scores'][row] == 1 and results['predictions'][row] == 0:
            s1in0 += 1
            s1in0_rows.append(row)
            totalScore1 += 1
        else:
            s1in1 += 1
            s1in1_rows.append(row)
            totalScore1 += 1

    # calculate percent of given score in cluster
    # note: score0inCluster0 represents percent of cluster 0 that has score 0
    score0inCluster0 = s0in0 / (s0in0 + s1in0)
    score1inCluster0 = s1in0 / (s0in0 + s1in0)

    score0inCluster1 = s0in1 / (s0in1 + s1in1)
    score1inCluster1 = s1in1 / (s0in1 + s1in1)

    misclassified = []
    # if the maximum percent is score0inCluster0, assign cluster labels of absent and present accordingly, calculate
    # accuracies, and create list of mis-classified records (where cluster labels are mutually exclusive)
    if score0inCluster0 == max(max(score0inCluster0, score1inCluster0), max(score0inCluster1, score1inCluster1)):
        c0label = "c0 - absent"
        c1label = "c1 - present"

        misclassified.extend(s0in1_rows)
        if totalScore0 != 0:
            score0accuracy = s0in0 / totalScore0
        else:
            score0accuracy = -1

        misclassified.extend(s1in0_rows)
        if totalScore1 != 0:
            score1accuracy = s1in1 / totalScore1
        else:
            score1accuracy = -1
    # if the maximum percent is score1inCluster0, assign cluster labels of absent and present accordingly, calculate
    # accuracies, and create list of mis-classified records (where cluster labels are mutually exclusive)
    elif score1inCluster0 == max(max(score0inCluster0, score1inCluster0), max(score0inCluster1, score1inCluster1)):
        c0label = "c0 - present"
        c1label = "c1 - absent"

        misclassified.extend(s0in0_rows)
        if totalScore0 != 0:
            score0accuracy = s0in1 / totalScore0
        else:
            score0accuracy = -1

        misclassified.extend(s1in1_rows)
        if totalScore1 != 0:
            score1accuracy = s1in0 / totalScore1
        else:
            score1accuracy = -1
    # if the maximum percent is score0inCluster1, assign cluster labels of absent and present accordingly, calculate
    # accuracies, and create list of mis-classified records (where cluster labels are mutually exclusive)
    elif score0inCluster1 == max(max(score0inCluster0, score1inCluster0), max(score0inCluster1, score1inCluster1)):
        c0label = "c0 - present"
        c1label = "c1 - absent"

        misclassified.extend(s0in0_rows)
        if totalScore0 != 0:
            score0accuracy = s0in1 / totalScore0
        else:
            score0accuracy = -1

        misclassified.extend(s1in1_rows)
        if totalScore1 != 0:
            score1accuracy = s1in0 / totalScore1
        else:
            score1accuracy = -1
    # if the maximum percent is score1inCluster1, assign cluster labels of absent and present accordingly, calculate
    # accuracies, and create list of mis-classified records (where cluster labels are mutually exclusive)
    else:
        c0label = "c0 - absent"
        c1label = "c1 - present"

        misclassified.extend(s0in1_rows)
        if totalScore0 != 0:
            score0accuracy = s0in0 / totalScore0
        else:
            score0accuracy = -1

        misclassified.extend(s1in0_rows)
        if totalScore1 != 0:
            score1accuracy = s1in1 / totalScore1
        else:
            score1accuracy = -1

    if (c0label == "c0 - absent" and c1label == "c1 - absent") or (c0label == "c0 - present" and c1label == "c1 - present"):
        print("\n\n\n\nMISTAKE HAPPENED HERE!!!!\n\n\n\n")

    print()

    # write results to file
    print(str(c0label) + ", " + str(c1label))
    print("Score 0 accuracy: " + str(score0accuracy))
    print("Score 1 accuracy: " + str(score1accuracy))

    file.write("\n\n\n" + str(c0label) + ", " + str(c1label))
    file.write("\nScore 0 accuracy: " + str(score0accuracy))
    file.write("\nScore 1 accuracy: " + str(score1accuracy))
    file.write("\n\nIncorrectly identified: " + str(misclassified))

    # correct = pd.Series(correct)
    # results["correct"] = correct
    # OUTPUT_FILE = BASE_FP_OUTPUT + criteria + "/" + criteria;
    # results.to_csv(f'{OUTPUT_FILE}_results.csv', header=True, index=False)

    return score0accuracy, score1accuracy


########################################################################################################
# classify
# Inputs: item, model, vectorizer
# Return: predicted
# Description: classifies given document using given model and returns the prediction
########################################################################################################
def classify(item, model, vectorizer):
    X = vectorizer.transform([item])
    predicted = model.predict(X)
    predicted = int(predicted)
    return predicted


########################################################################################################
# build_model
# Inputs: scores, document, criteria, file
# Return: score0accuracy, score1accuracy
# Description: vectorizes documents using tf-idf, builds kMeans clusters, predicts where each training
# document would be classified, and calls function to analyze unsupervised clusters and calculate
# conditional accuracies
########################################################################################################
def build_model(scores, document, criteria, file):
    en_stop = stopwords.words('english')
    #en_stop.extend(['introduction', 'abstract', 'conclusion', 'discussion', 'method', 'result', 'data', 'mental', 'health'])

    # vectorize with ngrams = 1, 2, and 3
    vectorizer = TfidfVectorizer(stop_words=en_stop, ngram_range=(1,3))
    X = vectorizer.fit_transform(document)

    # build model
    true_k = 2
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    # print cluster centroids (important ngrams)
    for i in range(true_k):
        file.write("Cluster % d:" % i)
        for ind in order_centroids[i, :10]:
            file.write('\n % s' % terms[ind])
        file.write("\n")

    # classify documents with model
    predictions = []
    for item in document:
        predictions.append(classify(item, model, vectorizer))

    predictions = pd.Series(predictions)
    results = pd.DataFrame()
    results['scores'] = scores
    results['scores'] = results['scores'].astype(int)
    results['predictions'] = predictions

    score0accuracy, score1accuracy = analyze_clusters(results, criteria, file)

    return score0accuracy, score1accuracy


########################################################################################################
# classify_documents
# Inputs: df, criteria, file
# Return: score0accuracy, score1accuracy
# Description: formats document (corpus) for tf-idf and kMeans clustering, and calls function to build
# model.
########################################################################################################
def classify_documents(df, criteria, file):

    document = []
    df.progress_apply(lambda x:  document.append(x['Cleaned']), axis=1, result_type='expand')

    score0accuracy, score1accuracy = build_model(df['Score'], document, criteria, file)

    return score0accuracy, score1accuracy


########################################################################################################
# main
# Inputs: none
# Return: none
# Description: This program creates two clusters for each criteria using tf-idf and kMeans, then classifies
# each document in order to calculate conditional accuracies. This process is repeated 10 times for each
# criteria. Average results for all criteria are written to summary_output.txt, while per-run results
# are writen to 'criteria'_results.csv for each of the criteria.
########################################################################################################
def main():

    warnings.filterwarnings('ignore')
    #criteria = ['ethics_section']
    criteria = ['anonymity', 'class_collection', 'data_public', 'data_source', 'dem_dist', 'drop', 'ethics_section',
                'feature_reconstruction', 'ground_truth_discussion', 'ground_truth_size', 'informed_consent',
                'irb', 'limitations', 'missing_values', 'noise', 'random_sample', 'replication', 'text']

    save_path_2 = BASE_FP_OUTPUT
    file_name_2 = "summary_output.txt"
    completeName2 = os.path.join(save_path_2, file_name_2)
    summary_file = open(completeName2, "w")

    for item in criteria:
       CRITERIA_FP_READ = BASE_FP + item + "/" + item + "_CLEANED.csv"
       print("\nWORKING WITH " + item + "!!!!!!\n")
       cleaned_df = pd.read_csv(f"{CRITERIA_FP_READ}", dtype='str', lineterminator='\n', keep_default_na=False)

       save_path = BASE_FP_OUTPUT + item
       file_name = item + "_output.txt"
       completeName = os.path.join(save_path, file_name)
       file = open(completeName, "w")

       score0 = []
       score1 = []

       # run each criteria 10 times to identify average accuracies
       for i in range(0, 10):
           file.write("Run " + str(i + 1) + "\n\n")
           score0accuracy, score1accuracy = classify_documents(cleaned_df, item, file)
           score0.append(score0accuracy)
           score1.append(score1accuracy)
           file.write("\n\n__________________________________________________\n\n")


       overall_score0 = sum(score0) / len(score0)
       overall_score1 = sum(score1) / len(score1)

       summary_file.write("\n\nCriteria: " + item)
       summary_file.write("\n\nScore 0 accuracy: " + str(overall_score0))
       summary_file.write("\nScore 1 accuracy: " + str(overall_score1))
       summary_file.write("\n\n___________________________________________\n")

       file.close()

    summary_file.close()


# Driver code
if __name__ == "__main__":
    main()