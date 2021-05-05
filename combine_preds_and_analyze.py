#################################################################
# Title: Combine Predictions and Analyze
# Author: Sam Cohen

# Notes:
# These functions combine all the results of the research pipeline
# and performs joint analysis on all the results.
##################################################################

# Packages
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

def load_ten_k_sents(file_path):
    """
    This function loads the 10-K sentences into Python.
    :param file_path: the path of the file
    :return: the 10-K sentences
    """
    ten_k = np.load(file_path, allow_pickle=True)
    return ten_k

def combine_predictions():
    """
    This function combines multiple text files that have been created as the output of the research pipeline.
    Those files are the 10-K sentences, the information on the companies behind the 10-K sentences, the
    predictions for specificity, and the predictions for climate.
    Some code is obtained from https://stackoverflow.com/questions/39717828/python-combine-two-text-files-into-one-line-by-line.
    :return: nothing, but outputs a file with all the above information combined into one text file
    """
    # arrays to track metrics
    total_climate = []
    specificities = []

    with open("Domain-Agnostic-Sentence-Specificity-Prediction/dataset/data/ten_k_sentences.txt") as xh:
      with open('Domain-Agnostic-Sentence-Specificity-Prediction/dataset/data/ten_k_info.txt') as yh:
        with open("Domain-Agnostic-Sentence-Specificity-Prediction/predictions.txt") as zh:
            with open("Domain-Agnostic-Sentence-Specificity-Prediction/climate_predictions.txt") as th:
                with open("combined.txt", "w") as wh:
                    #Read first file
                    xlines = xh.readlines()
                    #Read second file
                    ylines = yh.readlines()
                    #Read third file
                    zlines = zh.readlines()
                    #Read fourth file
                    tlines = th.readlines()
                    #Print lengths
                    print('sentence length:', len(xlines))
                    print('info length:', len(ylines))
                    print('specificity prediction length:', len(zlines))
                    print('climate prediction length:', len(tlines))

                    #Combine content of both lists
                    #combine = list(zip(ylines,xlines))
                    #Write to third file
                    for i in range(len(xlines)):
                        if i == 0:
                            print()
                        else:
                            # need to add climate predictions
                            #total_climate.append()
                           # regex = re.compile(
                            #    r'[0-9]\.[0-9]{3,6}')

                            #matches = regex.finditer(document['10-K'])
                            specificities.append(float(zlines[i-1].strip()))
                            total_climate.append(int(tlines[i]))
                            line = ylines[i].strip() + '\t' + zlines[i-1].strip() + '\t' + tlines[i].strip() + '\t' + xlines[i]
                            wh.write(line)

    print(75 * '=')
    print("Specificity Statistics:")
    print('Mean Specificity:', np.mean(np.array(specificities)))
    print('Standard Deviation Specificity:', np.std(np.array(specificities)))
    print('Max Specificity:', np.max(np.array(specificities)))
    print('Min Specificity:', np.min(np.array(specificities)))

    print(75*'=')

    print("Climate Prediction Statistics:")
    print('Climate Related Sentences Sum:', np.sum(np.array(total_climate)))
    print('Non Climate Related Sentences Sum:', len(total_climate) - np.sum(np.array(total_climate)))
    print('Climate Related Sentences Percent', np.sum(np.array(total_climate))/len(total_climate))

    print(75*'=')

def eda_plots():
    """
    This function reads in the file with combined information and outputs some EDA plots on the data.
    :return: nothing, but outputs EDA plots
    """
    print(os.getcwd())
    df = pd.read_csv("combined.txt", delimiter="\t", names=["CIK", "Year", "Stock Ticker", "Company Name", "Industry", "Specificity", "Climate Prediction", "Sentence"])
    print(df.info())
    print(df.columns)

    # drop data sources that have less than 20 sentences
    # drop sentences that are less than 10 words
    new_df = df.groupby(["Stock Ticker", "Year"])
    new_df = new_df.agg({"Sentence": "nunique"})
    new_df = new_df.rename(columns={"Sentence":"num_sents"})
    df = df.merge(new_df, how='inner', on=["Stock Ticker", "Year"])
    df["sentence_length"] = df['Sentence'].str.split().apply(len)
    df = df[df["num_sents"]>=20]
    df = df[df["sentence_length"]>10]

    training_sents()
    unique_company_count(df)
    plot_companies_by_sector(df,  "Count of Companies by Sector", 'output/eda_companies_by_sector')
    plot_sents_by_sector(df, "Count of Sentences by Sector", 'output/eda_sentences_by_sector.jpg')
    plot_companies_per_year(df,  "Count of Companies by Year", 'output/eda_companies_by_year.jpg')
    plot_sents_per_year(df, "Count of Sentences by Year", 'output/eda_sentences_by_year.jpg')
    num_sents_per_filing(df, "Histogram of Sentences by Filing", 'output/eda_hist_sentences_by_filing')
    hist_sent_specificity(df, "Sentence Specificity Distribution", 'output/eda_dist_sentence_specificity')
    climate_bar(df, "Climate Predictions", 'output/eda_climate_preds.jpg')
    stats_on_sents_per_10k(df)
    t_test(df)

    print('end')

def unique_company_count(df):
    """
    This function prints the number of unique companies in the final dataset.
    :param df: dataframe with the combined information
    :return: nothing, but prints the number of unique companies
    """
    # number of unique companies
    print("Number of Unique Companies:", df["Stock Ticker"].nunique())
    print("Number of Unique Filings:", df.groupby(["Stock Ticker", "Year"]).agg("nunique"))

def plot_companies_by_sector(df, title, out_path):
    """
    This function plots the companies by sector.
    The code comes from https://www.kite.com/python/answers/how-to-count-unique-values-in-a-pandas-dataframe-group-in-python.
    :param df: dataframe with combined results
    :param title: title of the new chart
    :param out_path: the output location of the charts
    :return: nothing, but outputs charts
    """
    # Number of companies by sector
    sectored_df = df.groupby("Industry")
    sectored_df = sectored_df.agg({"Stock Ticker": "nunique"})
    sectored_df = sectored_df.reset_index()

    # Plot
    sns.set(style="whitegrid")
    plt.title(title)
    ax = sns.barplot(data=sectored_df, x="Industry", y="Stock Ticker")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    #plt.xticks(ticks=np.arange(len(sectored_df["Industry"])),labels=sectored_df["Industry"], fontsize=16)
    plt.yticks(ticks=range(0,max(sectored_df["Stock Ticker"])+1, 5))
    plt.xlabel("Industry Sector")
    plt.ylabel("Number of Companies")
    sns.despine(left=True)
    plt.show()
    plt.savefig(out_path)

def plot_sents_by_sector(df, title, out_path):
    """
    This function plots the sentences by sector.
    The code comes from https://www.kite.com/python/answers/how-to-count-unique-values-in-a-pandas-dataframe-group-in-python.
    :param df: dataframe with combined results
    :param title: title of the new chart
    :param out_path: the output location of the charts
    :return: nothing, but outputs charts
    """
    # Number of companies by sector
    sectored_df = df.groupby("Industry")
    sectored_df = sectored_df.agg({"Sentence": "nunique"})
    sectored_df = sectored_df.reset_index()

    # Plot
    sns.set(style="whitegrid")
    plt.title(title)
    ax = sns.barplot(data=sectored_df, x="Industry", y="Sentence")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    #plt.xticks(ticks=np.arange(len(sectored_df["Industry"])),labels=sectored_df["Industry"], fontsize=16)
    plt.yticks(ticks=range(0,max(sectored_df["Sentence"])+1, 5000))
    plt.xlabel("Industry Sector")
    plt.ylabel("Number of Sentences")
    sns.despine(left=True)
    plt.show()
    plt.savefig(out_path)

def plot_companies_per_year(df, title, out_path):
    """
    This function plots the number of companies per year.
    :param df: dataframe of combined information
    :param title: title of the new chart
    :param out_path: the location of the new chart
    :return: nothing, but saves a new chart
    """
    # Number of companies per year
    yeared_df = df.groupby("Year")
    yeared_df = yeared_df.agg({"Stock Ticker": "nunique"})
    yeared_df = yeared_df.reset_index()

    # Plot
    sns.set(style="whitegrid")
    plt.title(title)
    ax = sns.barplot(data=yeared_df, x="Year", y="Stock Ticker")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    plt.yticks(ticks=range(0,max(yeared_df["Stock Ticker"])+1, 50))
    #ax.set_yticklabels(ax.get_yticklabels())
    plt.xlabel("Years")
    plt.ylabel("Number of Companies")
    sns.despine(left=True)
    plt.show(ax=ax)
    plt.savefig(out_path)

def plot_sents_per_year(df, title, out_path):
    """
    This function plots the number of sentences per year.
    :param df: dataframe of combined information
    :param title: title of the new chart
    :param out_path: the location of the new chart
    :return: nothing, but saves a new chart
    """
    # Number of companies per year
    yeared_df = df.groupby("Year")
    yeared_df = yeared_df.agg({"Sentence": "nunique"})
    yeared_df = yeared_df.reset_index()

    # Plot
    sns.set(style="whitegrid")
    plt.title(title)
    ax = sns.barplot(data=yeared_df, x="Year", y="Sentence")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    plt.yticks(ticks=range(0,max(yeared_df["Sentence"])+1, 5000))
    #ax.set_yticklabels(ax.get_yticklabels())
    plt.xlabel("Years")
    plt.ylabel("Number of Sentences")
    sns.despine(left=True)
    plt.show(ax=ax)
    plt.savefig(out_path)

def num_sents_per_filing(df, title, out_path):
    """
        This function plots the number of sentences per filing.
        :param df: dataframe of combined information
        :param title: title of the new chart
        :param out_path: the location of the new chart
        :return: nothing, but saves a new chart
        """
    # Number of companies per year
    yeared_df = df.groupby(["Stock Ticker", "Year"])
    yeared_df = yeared_df.agg({"Sentence": "nunique"})
    yeared_df = yeared_df.reset_index()

    # Plot
    sns.set(style="whitegrid")
    plt.title(title)
    ax = sns.histplot(data=yeared_df, x="Sentence", bins=20)
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    plt.yticks(ticks=range(0, 250, 50))
    plt.xticks(ticks=range(0, 850, 100))
    # ax.set_yticklabels(ax.get_yticklabels())
    plt.xlabel("Number of Sentences")
    plt.ylabel("Count")
    sns.despine(left=True)
    plt.show(ax=ax)
    plt.savefig(out_path)

def hist_sent_specificity(df, title, out_path):
    """
        This function plots the distribution of sentences specificity.
        :param df: dataframe of combined information
        :param title: title of the new chart
        :param out_path: the location of the new chart
        :return: nothing, but saves a new chart
        """

    # Plot

    ax = sns.displot(df, x="Specificity")
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    #plt.yticks(ticks=range(0, 250, 50))
    #plt.xticks(ticks=range(0, 850, 100))
    # ax.set_yticklabels(ax.get_yticklabels())
    sns.set(style="whitegrid")
    plt.title(title)
    #plt.xlabel("Specificity")
    #plt.ylabel("Density")
    sns.despine(left=True)
    plt.show(ax=ax)
    plt.savefig(out_path)

def climate_bar(df, title, out_path):
    """
    This function plots the number of climate and non-climate related sentences.
    :param df: dataframe of combined information
    :param title: title of the new chart
    :param out_path: the location of the new chart
    :return: nothing, but saves a new chart
    """
    # Number of companies per year
    #yeared_df = df.groupby("Climate Prediction")
    #yeared_df = yeared_df.agg({"Sentence": "nunique"})
   # yeared_df = yeared_df.reset_index()

    # Plot
    sns.set(style="whitegrid")
    plt.title(title)
    ax = sns.countplot(data=df, x="Climate Prediction")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="right", fontsize=8)
    plt.yticks(ticks=range(0,300000, 25000))
    #ax.set_yticklabels(ax.get_yticklabels())
    plt.xlabel("Climate Prediciton")
    plt.ylabel("Count")
    sns.despine(left=True)
    plt.show(ax=ax)
    plt.savefig(out_path)

def t_test(df):
    """
        This function plots the distribution of climate prediction.
        :param df: dataframe of combined information
        :param title: title of the new chart
        :param out_path: the location of the new chart
        :return: nothing, but saves a new chart
        """
    climate_df = df[df["Climate Prediction"] == 1]
    non_climate_df = df[df["Climate Prediction"] == 0]
    print("Mean of Climate Preds:", np.mean(climate_df["Specificity"].values))
    print("Var of Climate Preds:", np.var(climate_df["Specificity"].values))
    print("Mean of Non-Climate Preds:", np.mean(non_climate_df["Specificity"].values))
    print("Var of Non-Climate Preds:", np.var(non_climate_df["Specificity"].values))
    t, p = scipy.stats.ttest_ind(climate_df["Specificity"].values, non_climate_df["Specificity"].values)
    print("T-Score:", t)
    print("P-value:", p)

    # Plot
    ax = sns.displot(df, x="Specificity", col="Climate Prediction", multiple="dodge")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    # plt.yticks(ticks=range(0, 250, 50))
    # plt.xticks(ticks=range(0, 850, 100))
    # ax.set_yticklabels(ax.get_yticklabels())
    sns.set(style="whitegrid")
    #plt.title("Comparison of Climate vs. Non-Climate Specificities")
    # plt.xlabel("Specificity")
    # plt.ylabel("Density")
    sns.despine(left=True)
    plt.show(ax=ax)

def anova_year(df):
    """
        This function plots the distribution of climate prediction.
        :param df: dataframe of combined information
        :param title: title of the new chart
        :param out_path: the location of the new chart
        :return: nothing, but saves a new chart
        """
    climate_df = df[df["Climate Prediction"] == 1]
    non_climate_df = df[df["Climate Prediction"] == 0]
    print("Mean of Climate Preds:", np.mean(climate_df["Specificity"].values))
    print("Var of Climate Preds:", np.var(climate_df["Specificity"].values))
    print("Mean of Non-Climate Preds:", np.mean(non_climate_df["Specificity"].values))
    print("Var of Non-Climate Preds:", np.var(non_climate_df["Specificity"].values))
    t, p = scipy.stats.ttest_ind(climate_df["Specificity"].values, non_climate_df["Specificity"].values)
    print("T-Score:", t)
    print("P-value:", p)

    # Plot
    ax = sns.displot(df, x="Specificity", col="Climate Prediction", multiple="dodge")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    # plt.yticks(ticks=range(0, 250, 50))
    # plt.xticks(ticks=range(0, 850, 100))
    # ax.set_yticklabels(ax.get_yticklabels())
    sns.set(style="whitegrid")
    #plt.title("Comparison of Climate vs. Non-Climate Specificities")
    # plt.xlabel("Specificity")
    # plt.ylabel("Density")
    sns.despine(left=True)
    plt.show(ax=ax)

def stats_on_sents_per_10k(df):
    """
    This function prints stats on the 10-K sentences.
    :param df: dataframe of the combined information
    :return: nothing, but printed statistics
    """
    new_df = df.groupby(["Stock Ticker", "Year"])
    new_df = new_df.agg({"Sentence": "nunique"})
    new_df = new_df.reset_index()

    print(75 * '=')
    print("Specificity Statistics:")
    print('Mean Number of Sentences per 10-K:', np.mean(new_df["Sentence"]))
    print('Standard Deviation Number of Sentences per 10-K:', np.std(new_df["Sentence"]))
    print('Max Number of Sentences per 10-K:', np.max(new_df["Sentence"]))
    print('Min Number of Sentences per 10-K:', np.min(new_df["Sentence"]))

def sentences_climate_related(df):
    """
    This function finds the percent of sentences that are climate related by sector and prints out statistics.
    :param df: dataframe of combined information
    :return: nothing, but charts and statistics
    """
    # Here we are finding the percent of sentences that are climate related by sector
    new_df = df.groupby(["Stock Ticker", "Year"])
    new_df = new_df.agg({"Sentence": "nunique"})
    new_df = new_df.reset_index()

    print(75 * '=')
    print("Specificity Statistics:")
    print('Mean Number of Sentences per 10-K:', np.mean(new_df["Sentence"]))
    print('Standard Deviation Number of Sentences per 10-K:', np.std(new_df["Sentence"]))
    print('Max Number of Sentences per 10-K:', np.max(new_df["Sentence"]))
    print('Min Number of Sentences per 10-K:', np.min(new_df["Sentence"]))

def training_sents():
    import climate_identification_models
    climate_identification_models.load_training_data()