########################################################################################################
# eda.py
# This program computes basic statistics of a number of attributes from 'thesis_raw.csv'
# and outputs the summary statistics in a files called idealized_summary_stats.csv, cs_summary_stats.csv,
# and ph_summary_stats.csv. The program also creats appropriate histograms and boxplots of the data.
########################################################################################################


import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib_venn import venn3
from matplotlib_venn_wordcloud import venn3_wordcloud
import plotly.express as px
import plotly.graph_objects as go

BASE_FP = '../data/'
BASE_FP_OUTPUT_I = '../output/eda/idealized/'
BASE_FP_OUTPUT_CS = '../output/eda/cs/'
BASE_FP_OUTPUT_PH = '../output/eda/ph/'
BASE_FP_OUTPUT_RESEARCH = '../output/eda/researchers/'
BASE_FP_OUTPUT_VISUALIZATIONS = '../output/visualizations/'


########################################################################################################
# researcher_plots
# Inputs: researchers - df 1
# Return: N/A
# Description: plots figures and descriptive statistics for the three main researchers
########################################################################################################
def researcher_plots(researchers):
    # Make comparative boxplots for De Choudhury, Drezde, and Coppersmith
    researchers_dict = {'Choudhury': [], 'Drezde': [], 'Coppersmith': []}

    df_choudhury = researchers[researchers['Choudhury'] == 1]
    for item in df_choudhury['%_of_idealized']:
        researchers_dict['Choudhury'].append(item)

    df_drezde = researchers[researchers['Drezde'] == 1]
    for item in df_drezde['%_of_idealized']:
        researchers_dict['Drezde'].append(item)

    df_coppersmith = researchers[researchers['Coppersmith'] == 1]
    for item in df_coppersmith['%_of_idealized']:
        researchers_dict['Coppersmith'].append(item)

    fig, ax = plt.subplots()
    ax.boxplot(researchers_dict.values())
    ax.set_xticklabels(researchers_dict.keys())

    plt.savefig(os.path.join(BASE_FP_OUTPUT_RESEARCH, 'comp_boxplots_researchers.png'))
    plt.clf()

    # Make overlapping histograms for De Choudhury, Drezde, and Coppersmith
    plt.hist(df_choudhury['%_of_idealized'], label="Choudhury")
    plt.hist(df_drezde['%_of_idealized'], label="Drezde")
    plt.hist(df_coppersmith['%_of_idealized'], label="Coppersmith")

    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.title("Distribution of Scores")
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(BASE_FP_OUTPUT_RESEARCH, 'comp_hists_researchers.png'))
    plt.clf()


########################################################################################################
# plot_boxplots
# Inputs: idealized - df 1, cs - df 2, ph - df 3
# Return: N/A
# Description: plots boxplots of specified dataframe columns and saves plots
########################################################################################################
def plot_boxplots(idealized, cs, ph):
    # create idealized boxplots
    idealized.boxplot(column=['%_of_idealized'])
    file_name = 'boxplot_i.png'
    plt.savefig(os.path.join(BASE_FP_OUTPUT_I, file_name))
    plt.clf()

    # create cs boxplots
    cs.boxplot(column=['%_of_cs'])
    file_name = 'boxplot_cs.png'
    plt.savefig(os.path.join(BASE_FP_OUTPUT_CS, file_name))
    plt.clf()

    # create ph boxplots
    ph.boxplot(column=['%_of_ph'])
    file_name = 'boxplot_ph.png'
    plt.savefig(os.path.join(BASE_FP_OUTPUT_PH, file_name))
    plt.clf()


########################################################################################################
# plot_histograms
# Inputs: df - dataframe of data, column - column to plot distribution of with histogram
# Return: N/A
# Description: plots histogram of specified dataframe column and saves plot
########################################################################################################
def plot_histograms(df, column, FILE_PATH):
    # create histogram and with label
    df[column].hist()
    title_label = "Distribution of  " + str(column)
    plt.title(title_label)
    plt.xlabel(column)
    plt.ylabel("Frequency")

    # save figure
    file_name = str(column) + '_histogram.png'
    plt.savefig(os.path.join(FILE_PATH, file_name))

    # clear plot
    plt.clf()


########################################################################################################
# plot_histograms_overall
# Inputs: idealized - df 1, cs - df 2, ph - df 3, columns_i - df 1's columns of interest, columns_cs -
# df 2's columns of interest, columns_ph - df 3's columns of interest
# Return: N/A
# Description: creates histograms of each of the columns of interest, plus overall hist plots for each
# df
########################################################################################################
def plot_histograms_overall(idealized, cs, ph, researchers, columns_i, columns_cs, columns_ph, columns_research):
    # create idealized histograms
    idealized.hist()
    file_name = 'histograms_i.png'
    plt.savefig(os.path.join(BASE_FP_OUTPUT_I, file_name))
    plt.clf()

    for column in columns_i:
        plot_histograms(idealized, column, BASE_FP_OUTPUT_I)

    # create cs histograms
    cs.hist()
    file_name = 'histograms_cs.png'
    plt.savefig(os.path.join(BASE_FP_OUTPUT_CS, file_name))
    plt.clf()

    for column in columns_cs:
        plot_histograms(cs, column, BASE_FP_OUTPUT_CS)

    # create ph histograms
    ph.hist()
    file_name = 'histograms_ph.png'
    plt.savefig(os.path.join(BASE_FP_OUTPUT_PH, file_name))
    plt.clf()

    for column in columns_ph:
        plot_histograms(ph, column, BASE_FP_OUTPUT_PH)

    # create researchers histograms
    ph.hist()
    file_name = 'histograms_researchers.png'
    plt.savefig(os.path.join(BASE_FP_OUTPUT_RESEARCH, file_name))
    plt.clf()

    for column in columns_research:
        plot_histograms(researchers, column, BASE_FP_OUTPUT_RESEARCH)


########################################################################################################
# summary_stats_by_df
# Inputs: df - dataframe, columns - columns to get summary stats from
# Return: dataframe with summary statistics
# Description: calculates summary statistics about df's columns using pd.describe()
########################################################################################################
def summary_stats_by_df(columns, df):
    # create overall dataframe with first column
    first_attribute = columns[0]
    stats_df_overall = pd.DataFrame({first_attribute: df[first_attribute].describe()})
    stats_df_overall.reset_index(level=0, inplace=True)
    stats_df_overall.rename(columns={'index': 'summary_stat_type', first_attribute: first_attribute}, inplace=True)

    # generage summary stats of all other columns and merge into one dataframe
    for attribute in columns:
        if attribute != first_attribute:
            stats_df = pd.DataFrame({attribute: df[attribute].describe()})
            stats_df.reset_index(level=0, inplace=True)
            stats_df.rename(columns={'index': 'summary_stat_type', attribute: attribute}, inplace=True)
            stats_df_overall = stats_df_overall.merge(stats_df, on='summary_stat_type')

    return stats_df_overall


########################################################################################################
# generate_summary_stats
# Inputs: idealized - dataframe 1, cs - dataframe 2, ph - dataframe 3
# Return: N/A
# Description: identifies columns of interest, calculates summary statistics, and writes statistics
# to output csv's 'idealized_summary_stats.csv', 'cs_summary_stats.csv', and 'ph_summary_stats.csv'.
# Also generates histograms and boxplots
########################################################################################################
def generate_summary_stats(idealized, cs, ph, researchers):
    # identify columns of interest
    columns_i = ['data_source_i', 'class_collection_i', 'random_sample_i', 'replication_i', 'dem_dist_i',
                 'informed_consent_i', 'data_public_i', 'irb_i', 'ground_truth_size_i', 'ground_truth_discussion_i',
                 'limitations_i', 'preprocess_anonymity_i', 'preprocess_drop_i', 'preprocess_missing_values_i',
                 'preprocess_noise_i', 'preprocess_text_i', 'ethics_section_i', '%_of_idealized']
    columns_cs = ['data_source_cs', 'class_collection_cs', 'random_sample_cs', 'replication_cs', 'dem_dist_cs',
                  'irb_cs', 'ground_truth_size_cs', 'ground_truth_discussion_cs', 'preprocess_drop_cs', '%_of_cs']
    columns_ph = ['data_source_ph', 'class_collection_ph', 'random_sample_ph', 'replication_ph', 'dem_dist_ph',
                  'informed_consent_ph', 'irb_ph', 'ground_truth_size_ph', 'ground_truth_discussion_ph', '%_of_ph']
    columns_research = ['Num_Researchers', 'Choudhury', 'Drezde', 'Coppersmith', 'data_source_i', 'class_collection_i',
                        'random_sample_i', 'replication_i', 'dem_dist_i', 'informed_consent_i', 'data_public_i',
                        'irb_i', 'ground_truth_size_i', 'ground_truth_discussion_i', 'limitations_i',
                        'preprocess_anonymity_i', 'preprocess_drop_i', 'preprocess_missing_values_i',
                        'preprocess_noise_i', 'preprocess_text_i', 'ethics_section_i', '%_of_idealized']

    # generate summary stats
    idealized_stats = summary_stats_by_df(columns_i, idealized)
    cs_stats = summary_stats_by_df(columns_cs, cs)
    ph_stats = summary_stats_by_df(columns_ph, ph)
    researcher_stats = summary_stats_by_df(columns_research, researchers)

    # output summary stats to csv's
    idealized_stats.to_csv(f'{BASE_FP_OUTPUT_I}idealized_summary_stats.csv', index=False)
    cs_stats.to_csv(f'{BASE_FP_OUTPUT_CS}cs_summary_stats.csv', index=False)
    ph_stats.to_csv(f'{BASE_FP_OUTPUT_PH}ph_summary_stats.csv', index=False)
    researcher_stats.to_csv(f'{BASE_FP_OUTPUT_RESEARCH}researcher_summary_stats.csv', index=False)

    # plot appropriate histograms
    plot_histograms_overall(idealized, cs, ph, researchers, columns_i, columns_cs, columns_ph, columns_research)

    # plot appropriate boxplots
    plot_boxplots(idealized, cs, ph)

    # find researcher stats + plots
    researcher_plots(researchers)


########################################################################################################
# create_sub_dfs
# Inputs: df - dataframe
# Return: idealized - dataframe 1, cs - dataframe 2, ph - dataframe 3
# Description: splits dataframe into three sub-dataframes containing scores for idealized, cs, and ph
# norms
########################################################################################################
def create_sub_dfs(df):
    # Drop irrelevent data
    df = df.drop(['Size of Ground Truth', 'Steps taken to be transparent in data collection and preprocessing steps',
                  'Discussion of Ground Truth', 'Year', 'Title'], axis=1)
    df = df.dropna()

    # create idealized df
    idealized = df[['Link to paper', 'Task', 'data_source_i', 'class_collection_i', 'random_sample_i', 'replication_i',
                    'dem_dist_i', 'informed_consent_i', 'data_public_i', 'irb_i', 'ground_truth_size_i',
                    'ground_truth_discussion_i', 'limitations_i', 'preprocess_anonymity_i', 'preprocess_drop_i',
                    'preprocess_missing_values_i', 'preprocess_noise_i', 'preprocess_text_i', 'ethics_section_i',
                    '%_of_idealized']].copy()

    # create cs df
    cs = df[['Link to paper', 'Task', 'data_source_cs', 'class_collection_cs', 'random_sample_cs', 'replication_cs',
             'dem_dist_cs', 'irb_cs', 'ground_truth_size_cs', 'ground_truth_discussion_cs', 'preprocess_drop_cs',
             '%_of_cs']].copy()

    # create ph df
    ph = df[['Link to paper', 'Task', 'data_source_ph', 'class_collection_ph', 'random_sample_ph', 'replication_ph',
             'dem_dist_ph', 'informed_consent_ph', 'irb_ph', 'ground_truth_size_ph', 'ground_truth_discussion_ph',
             '%_of_ph']].copy()

    researchers = df[['Num_Researchers', 'Choudhury', 'Drezde', 'Coppersmith', 'data_source_i', 'class_collection_i',
                      'random_sample_i', 'replication_i', 'dem_dist_i', 'informed_consent_i', 'data_public_i',
                      'irb_i', 'ground_truth_size_i', 'ground_truth_discussion_i', 'limitations_i',
                      'preprocess_anonymity_i', 'preprocess_drop_i', 'preprocess_missing_values_i',
                      'preprocess_noise_i', 'preprocess_text_i', 'ethics_section_i', '%_of_idealized']].copy()

    return idealized, cs, ph, researchers


########################################################################################################
# create_venn
# Inputs: N/A
# Return: N/A
# Description: Create Venn diagrams to explain relationship between cs, ph, and idealized norms
# References:
# Venn diagram base - https://github.com/konstantint/matplotlib-venn
# Extended Venn diagram with wordcloud - https://github.com/paulbrodersen/matplotlib_venn_wordcloud
# Wordcloud documentation - https://github.com/amueller/word_cloud
########################################################################################################
def create_venn():
    # Identify sets
    circle_cs = ['Data Source', 'Class Collection', 'Random Sample', 'Replication', 'Dem Dist', 'IRB', 'GT Size',
                 'GT Discussion', 'Drop']
    circle_ph = ['Data Source', 'Class Collection', 'Random Sample', 'Replication', 'Dem Dist', 'Consent', 'IRB',
                 'GT Size', 'GT Discussion']
    circle_i = ['Data Source', 'Class Collection', 'Random Sample', 'Replication', 'Dem Dist', 'Consent', 'IRB',
                'Public Data', 'GT Size', 'GT Discussion', 'Limitations', 'Anonymity', 'Drop', 'Missing Values',
                'Noise', 'Text', 'Ethics Section']
    labels = ['Computer Science', 'Public Health', 'Idealized']

    # create Venn diagram 1
    venn3([set(circle_cs), set(circle_ph), set(circle_i)], set_labels=labels)
    plt.savefig(os.path.join(BASE_FP_OUTPUT_VISUALIZATIONS, 'Venn.png'))
    plt.clf()

    # create Venn diagram 2 (with words)
    venn3_wordcloud([set(circle_cs), set(circle_ph), set(circle_i)], set_colors=['w', 'w', 'w'],
                    set_edgecolors=['k', 'k', 'k'], wordcloud_kwargs={'min_font_size': 14})
    plt.savefig(os.path.join(BASE_FP_OUTPUT_VISUALIZATIONS, 'Venn_wordcloud.png'))
    plt.clf()


########################################################################################################
# create_heatmap
# Inputs: N/A
# Return: N/A
# Description: Create heatmap to display the paper score breakdown
########################################################################################################
def create_heatmap(idealized):
    df = idealized.drop(['Link to paper', 'Task', '%_of_idealized'], axis=1)
    df_list = df.values.tolist()
    fig = go.Figure(data=go.Heatmap(z=df_list, x=df.columns, colorscale='purples'))
    fig.show()
    # do 'pip install -U kaleido' and try again to save image when have better wifi
    fig.write_image(os.path.join(BASE_FP_OUTPUT_VISUALIZATIONS, 'heatmap.png'))


########################################################################################################
# make_visualizations
# Inputs: idealized - df 1, cs - df 2, ph - df 3, researchers - df 4
# Return: N/A
# Description: Generates visualizations based off of scores and definitions
########################################################################################################
def make_visualizations(idealized, cs, ph, researchers):
    # Create Venn diagrams to explain relationship between cs, ph, and idealized norms
    create_venn()

    # Create heatmap to display the paper score breakdown
    # Maybe try with the binned values instead of the individual pieces? if doing binned, scale binned to
    # between 0 and 2
    create_heatmap(idealized)


########################################################################################################
# main
# Inputs: none
# Return: none
# Description: reads in thesis data and generates summary statistics
########################################################################################################
def main():

    # open file
    # df = pd.read_csv(f'{BASE_FP}thesis_raw.csv')
    df = pd.read_csv(f'{BASE_FP}thesis_raw_v2.csv')

    idealized, cs, ph, researchers = create_sub_dfs(df)

    #generate_summary_stats(idealized, cs, ph, researchers)

    make_visualizations(idealized, cs, ph, researchers)


if __name__ == '__main__':
    main()