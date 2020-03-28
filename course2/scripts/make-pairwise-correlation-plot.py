from string import ascii_letters
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

OUTFILE = './images/pairwise-correlations.png'
DATAFILE = './data/world-happiness.csv'

sns.set(style="white")


def ingest_data():
    print('... ingesting data')

    # load the data and print the shape
    df = pd.read_csv(DATAFILE, index_col=0)
    # print("df: {} x {}".format(df.shape[0],df.shape[1]))

    # clean up the column names and remove some
    df.columns = [re.sub("\s+", "_", col) for col in df.columns.tolist()]

    # remove missing data
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    print(f'... removed {before - after} missing values')

    return df


def plot_pairwise_correlations(df):
    print('... generating plot')
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    plt.gcf().subplots_adjust(bottom=0.3, left=0.25)
    sns_plot = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                           square=True, linewidths=.5, cbar_kws={"shrink": .5})

    return sns_plot


def save_sns_plot(p):
    print('... writing image')
    p.figure.savefig(OUTFILE)


if __name__ == "__main__":
    df = ingest_data()
    sns_plot = plot_pairwise_correlations(df)
    save_sns_plot(sns_plot)
