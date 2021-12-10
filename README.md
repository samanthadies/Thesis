# Thesis

# Overview
This repository contains python files used to perform exploratory data analysis on a dataset manually compiled from comupter science publications. With the data in the repository, we aim to

**1.** Quantify the ethical value of transparency as it relates to two parts of the data science life cycle -  data collection and preprocessing - in order to develop guidelines/best practices for computer science research on mental health using social media data.
**2.** Develop a semi-supervised model for transparency modeling that scores publications on how well they adhere to the defined normative behavior for full transparency.

# Main Dataset Descriptions

### [thesis_raw_v3.csv](/data/thesis_raw_v3.csv)
This dataset contains the raw data manually compiled from the computer science publications referenced in each row. There are 50 rows, one per publication, with the following attributes:

* **Link to Paper** - url to pdf version of each publication
* **Title** - publication title
* **Size of Ground Truth** - size of ground truth data for a subset of publications
* **Discussion of Ground Truth** - notes on the collection and discussion of ground truth data in a subset of publications
* **Steps taken to be transparent in data collection and preprocessing steps** - discussion on steps taken to be transparent in a subset of publications
* **Researchers** - a list of the publication's authors in "lastname, lastname, ..." format
* **NumResearchers** - the number of authors
* **Choudhury** - whether or not Munmun De Choudhury was an author of this publication
* **Drezde** - whether or not Mark Drezde was an author of this publication
* **Coppersmith** - whether or not Glen Coppersmith was an author of this publication
* **Year** - publication year
* **Task** - predictive or descriptive computer science task
* **data_source_i** - score for explanation of data source for idealized norm (0-2)
* **class_collection_i** - score for discussion on data class collection for idealized norm (0-2)
* **random_sample_i** - score for explanation of random sampling for idealized norm (0-2)
* **replication_i** - score for study replicability for idealized norm (0-2)
* **dem_dist_i** - score for discussion on demographics for idealized norm (0-2)
* **informed_consent_i** - score for explanation on informed consent for idealized norm (0-2)
* **data_public_i** - score for public vs. private data for idealized norm (0-2)
* **irb_i** - score for explanation of IRB for idealized norm (0-2)
* **human_subject_protection_i** - binned score for discussion on privacy and autonomy for idealized norm (0-4); sum of informed_consent_i, data_public_i, and irb_i
* **ground_truth_size_i** - score for ground truth size for idealized norm (0-2)
* **ground_truth_discussion_i** - score for ground truth discussion for idealized norm (0-2)
* **limitations_i** - score for discussion on limitations for idealized norm (0-2)
* **preprocess_anonymity_i** - score for discussion on maintaining anonymity for idealized norm (0-2)
* **preprocess_drop_i** - score for explanation of dropped values for idealized norm (0-2)
* **preprocess_missing_values_i** - score for explanation of missing values transparency for idealized norm (0-2)
* **preprocess_noise_i** - score for identification of noise for idealized norm (0-2)
* **preprocess_text_i** - score for explanation of text preprocessing for idealized norm (0-2)
* **preprocess_feature_reconstruction_i** - score for explanation of feature reconstruction for idealized norm (0-2)
* **preprocess_i** - binned score for discussion on preprocessing for idealized norm (0-10); sum of preprocess_anonymity_i, preprocess_drop_i, preprocess_missing_values_i, preprocess_noise_i, preprocess_text_i, and preprocess_feature_reconstruction_i
* **ethics_section_i** - score for inclusion of ethics section for idealized norm (0-2)
* **&#37_of_idealized** - score for publication transparency for idealized norm (0-100)
* **data_source_cs** - score for explanation of data source for computer science norm (0-2)
* **class_collection_cs** - score for discussion on data class collection for computer science norm (0-2)
* **random_sample_cs** - score for explanation of random sampling for computer science norm (0-2)
* **replication_cs** - score for study replicability for computer science norm (0-2)
* **dem_dist_cs** - score for discussion on demographics for computer science norm (0-2)
* **irb_cs** - score for explanation of IRB for computer science norm (0-2)
* **ground_truth_size_cs** - score for ground truth size for computer science norm (0-2)
* **ground_truth_discussion_cs** - score for ground truth discussion for computer science norm (0-2)
* **preprocess_drop_cs** - score for explanation of dropped values for computer science norm (0-2)
* **preprocess_feature_reconstruction_cs** - score for explanation of feature reconstruction for computer science norm (0-2)
* **&#37_of_cs** - score for publication transparency for computer science norm (0-100)
* **data_source_ph** - score for explanation of data source for public health norm (0-2)
* **class_collection_ph** - score for discussion on data class collection for public health norm (0-2)
* **random_sample_ph** - score for explanation of random sampling for public health norm (0-2)
* **replication_ph** - score for study replicability for public health norm (0-2)
* **dem_dist_ph** - score for discussion on demographics for public health norm (0-2)
* **informed_consent_ph** - score for explanation on informed consent for public health norm (0-2)
* **irb_ph** - score for explanation of IRB for public health norm (0-2)
* **ground_truth_size_ph** - score for ground truth size for public health norm (0-2)
* **ground_truth_discussion_ph** - score for ground truth discussion for public health norm (0-2)
* **&#37_of_ph** - score for publication transparency for public health norm (0-100)

# Python Files

### [eda.py](/scripts/eda.py)
This program will read in the most-updated, raw dataset (currently [thesis_raw_v3.csv](/data/thesis_raw_v3.csv)), perform basic preprocessing to select the columns of interest, and creates four dataframes for analysis: idealized, which contains columns and scores related to the idealized transparency norm, cs, which contains columns and scores related to the computer science transparency norm, ph, which contains columns and scores related to the public health transparency norm, and researchers, which is identical to idealized with additional features related to specific researchers. The script then generates summary statistics for each of the dataframes (five number summaries, mean, and standard deviation), creates histograms for each of the attributes, creates boxplots of the score response variables, and creates comparative boxplots of the scores for the researchers in the research dataframe. Finally, the script creates a handful of preliminary visualizations including a heatmap and Venn diagrams.

# Output Files and Visualizations

### [Idealized](/output/eda/idealized)
* **&#37_of_idealized_histogram.png** - histogram measuring frequency of the publications with each idealized score
* **boxplot_i_histogram.png** - boxplot of the distribution of the computer science scores for each of the publications
* **class_collection_i_histogram.png** - histogram measuring frequency of the publications with each class_collection idealized score
* **data_public_i_histogram.png** - histogram measuring the frequency of the number of publications with each data_public idealized score
* **data_source_i_histogram.png** - histogram measuring the frequency of the number of publications with each data_source idealized score
* **dem_dist_i_histogram.png** - histogram measuring the frequency of the number of publications with each dem_dist idealized score
* **ethics_section_i_histogram.png** - histogram measuring the frequency of the number of publications with each ethics_section idealized score
* **ground_truth_discussion_i_histogram.png** - histogram measuring the frequency of the number of publications with each ground_truth_discussion idealized score
* **ground_truth_size_i_histogram.png** - histogram measuring the frequency of the number of publications with each ground_truth_size idealized score
* **histograms_i.png** - histogram matrix of all feature histograms
* **informed_consent_i_histogram.png** - histogram measuring the frequency of the number of publications with each informed_consent idealized score
* **irb_i_histogram.png** - histogram measuring the frequency of the number of publications with each irb idealized score
* **limitations_i_histogram.png** - histogram measuring the frequency of the number of publications with each limitations idealized score
* **preprocess_anonymity_i_histogram.png** - histogram measuring the frequency of the number of publications with each preprocess_anonymity idealized score
* **preprocess_drop_i_histogram.png** - histogram measuring the frequency of the number of publications with each preprocess_drop idealized score
* **preprocess_feature_reconstruction_i_histogram.png** - histogram measuring the frequency of the number of publications with each preprocess_feature_reconstruction idealized score
* **preprocess_missing_values_i_histogram.png** - histogram measuring the frequency of the number of publications with each preprocess_missing_values idealized score
* **preprocess_noise_i_histogram.png** - histogram measuring the frequency of the number of publications with each preprocess_noise idealized score
* **preprocess_text_i_histogram.png** - histogram measuring the frequency of the number of publications with each preprocess_text idealized score
* **random_sample_i_histogram.png** - histogram measuring the frequency of the number of publications with each random_sample idealized score
* **replication_i_histogram.png** - histogram measuring the frequency of the number of publications with each replication idealized score
* **idealized_summary_stats.csv** - text file containing count, mean, standard deviation, min, 25th percentile, median, 75th percentile, and max for each of the attributes

### [Computer Science](/output/eda/cs)
* **&#37_of_cs_histogram.png** - histogram measuring frequency of the publications with each computer science score
* **boxplot_cs.png** - boxplot of the distribution of the computer science scores for each of the publications
* **histograms_cs.png** - histogram matrix of all feature histograms
* **class_collection_cs_histogram.png** - histogram measuring frequency of the publications with each class_collection computer science score
* **data_source_cs_histogram.png** - histogram measuring the frequency of the number of publications with each data_source computer science score
* **dem_dist_cs_histogram.png** - histogram measuring the frequency of the number of publications with each dem_dist computer science score
* **ground_truth_discussion_cs_histogram.png** - histogram measuring the frequency of the number of publications with each ground_truth_discussion computer science score
* **ground_truth_size_cs_histogram.png** - histogram measuring the frequency of the number of publications with each ground_truth_size computer science score
* **irb_cs_histogram.png** - histogram measuring the frequency of the number of publications with each irb computer science score
* **preprocess_drop_cs_histogram.png** - histogram measuring the frequency of the number of publications with each preprocess_drop computer science score
* **preprocess_feature_reconstruction_cs_histogram.png** - histogram measuring the frequency of the number of publications with each preprocess_feature_reconstruction computer science score
* **random_sample_cs_histogram.png** - histogram measuring the frequency of the number of publications with each random_sample computer science score
* **replication_cs_histogram.png** - histogram measuring the frequency of the number of publications with each replication computer science score
* **cs_summary_stats.csv** - text file containing count, mean, standard deviation, min, 25th percentile, median, 75th percentile, and max for each of the attributes

### [Public Health](/output/eda/ph)
* **&#37_of_ph_histogram.png** - histogram measuring frequency of the publications with each public health score
* **boxplot_ph.png** - boxplot of the distribution of the public health scores for each of the publications
* **histograms_ph.png** - histogram matrix of all feature histograms
* **class_collection_ph_histogram.png** - histogram measuring frequency of the publications with each class_collection public health score
* **data_source_ph_histogram.png** - histogram measuring the frequency of the number of publications with each data_source public health score
* **dem_dist_ph_histogram.png** - histogram measuring the frequency of the number of publications with each dem_dist public health score
* **ground_truth_discussion_ph_histogram.png** - histogram measuring the frequency of the number of publications with each ground_truth_discussion public health score
* **ground_truth_size_ph_histogram.png** - histogram measuring the frequency of the number of publications with each ground_truth_size public health score
* **informed_consent_ph_histogram.png** - histogram measuring the frequency of the number of publications with each informed_consent public health score
* **irb_ph_histogram.png** - histogram measuring the frequency of the number of publications with each irb public health score
* **random_sample_ph_histogram.png** - histogram measuring the frequency of the number of publications with each random_sample public health score
* **replication_ph_histogram.png** - histogram measuring the frequency of the number of publications with each replication public health score
* **ph_summary_stats.csv** - text file containing count, mean, standard deviation, min, 25th percentile, median, 75th percentile, and max for each of the attributes

### [Researchers](/output/eda/researchers)
* **Choudhury_histogram.png** - histogram of scores for publications authored by De Choudhury according to the idealized score
* **Drezde_histogram.png** - histogram of scores for publications authored by Drezde according to the idealized score
* **Coppersmith_histogram.png** - histogram of scores for publications authored by Coppersmith according to the idealized score
* **Num_Researchers_histogram.png** - histogram measuring frequency of the number of researchers per publication
* **&#37_of_idealized_histogram.png** - histogram measuring frequency of the publications with each idealized score
* **class_collection_i_histogram.png** - histogram measuring frequency of the publications with each class_collection idealized score
* **comp_boxplots_researchers.png** - comparative boxplots of the distributions of idealized scores for De Choudhury's, Drezde's, and Coppersmith's publications
* **comp_hists_researchers.png** - comparative histograms of the distributions of the idealized scores for De Choudhury's, Drezde's, and Coppersmith's publications
* **data_public_i_histogram.png** - histogram measuring the frequency of the number of publications with each data_public idealized score
* **data_source_i_histogram.png** - histogram measuring the frequency of the number of publications with each data_source idealized score
* **dem_dist_i_histogram.png** - histogram measuring the frequency of the number of publications with each dem_dist idealized score
* **ethics_section_i_histogram.png** - histogram measuring the frequency of the number of publications with each ethics_section idealized score
* **ground_truth_discussion_i_histogram.png** - histogram measuring the frequency of the number of publications with each ground_truth_discussion idealized score
* **ground_truth_size_i_histogram.png** - histogram measuring the frequency of the number of publications with each ground_truth_size idealized score
* **histograms_researchers.png** - histogram matrix of all feature histograms
* **informed_consent_i_histogram.png** - histogram measuring the frequency of the number of publications with each informed_consent idealized score
* **irb_i_histogram.png** - histogram measuring the frequency of the number of publications with each irb idealized score
* **limitations_i_histogram.png** - histogram measuring the frequency of the number of publications with each limitations idealized score
* **preprocess_anonymity_i_histogram.png** - histogram measuring the frequency of the number of publications with each preprocess_anonymity idealized score
* **preprocess_drop_i_histogram.png** - histogram measuring the frequency of the number of publications with each preprocess_drop idealized score
* **preprocess_feature_reconstruction_i_histogram.png** - histogram measuring the frequency of the number of publications with each preprocess_feature_reconstruction idealized score
* **preprocess_missing_values_i_histogram.png** - histogram measuring the frequency of the number of publications with each preprocess_missing_values idealized score
* **preprocess_noise_i_histogram.png** - histogram measuring the frequency of the number of publications with each preprocess_noise idealized score
* **preprocess_text_i_histogram.png** - histogram measuring the frequency of the number of publications with each preprocess_text idealized score
* **random_sample_i_histogram.png** - histogram measuring the frequency of the number of publications with each random_sample idealized score
* **replication_i_histogram.png** - histogram measuring the frequency of the number of publications with each replication idealized score
* **researcher_summary_stats.csv** - text file containing count, mean, standard deviation, min, 25th percentile, median, 75th percentile, and max for each of the attributes

### [Visualizations](/output/visualizations)
* **Venn.png** - Venn diagram which depicts the number of criteria in each defined norm and portrays their relationship
* **Venn_worldcloud.png** - Venn diagram which depicts the number of criteria in each defined norm, portrays their relationship, and uses a wordcloud to display the words
* **heatmap.png** - depiction of the idealized scores for each publication, broken down by criteria
* **norms_Venn.jpg** - Venn diagram which depicts the relationship between the criteria (generated with https://miro.com/app/board/o9J_lhwKLfA=/)
* **thesis_scoring.png** - hierarchical display of the idealized scoring schema (generated with https://www.diagrams.net/)
