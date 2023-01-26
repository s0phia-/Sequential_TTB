import matplotlib.pyplot as plt
import plotnine as p9
import os

import pandas
import pandas as pd


def get_data(filepath):
    """
    read in data from the csv files that resulted from the experiments
    :param filepath: path to result csvs
    :return: a pandas dataframe with all the results together in long format
    """
    files = os.listdir(filepath)

    all_returns = pd.DataFrame()

    for csv_name in files:
        csv_data = pd.read_csv(filepath + '/' + csv_name)
        agent_class, agent_number = csv_name.split('_')

        n = len(csv_data)
        pd_data = pd.DataFrame({'agent': [agent_class] * n,
                                'agent_number': [agent_number] * n,
                                'episode': csv_data['episode_number'],
                                'return': csv_data['return']})

        all_returns = pd.concat([all_returns, pd_data], axis=0)

    return all_returns


def average_over_runs(data: pandas.DataFrame):
    """
    average out the agent index, whether multiple agents' performance is being averaged
    :param data: results data
    :return: averaged results data
    """
    av_data = data.groupby(['agent', 'episode'])['return'].mean().reset_index()
    return av_data


def plot_gg(pd_df, legend_pos):
    fig, plot = (p9.ggplot(data=pd_df,
                           mapping=p9.aes(x='episode', y='return', color='agent')) +
                 p9.geom_line() +
                 p9.geom_point() +
                 p9.theme_bw() +
                 p9.labs(x="Episode", y="Total Return", color="Model") +
                 p9.theme(text=p9.element_text(size=9), legend_position=legend_pos)
                 ).draw(show=True, return_ggplot=True)
    return plot, fig
