import matplotlib.pyplot as plt
import plotnine as p9
import pandas as pd
import csv
import time


def plot_gg(pd_df, legend_pos):
    fig, plot = (p9.ggplot(data=pd_df,
                           mapping=p9.aes(x='step', y='return', color='agent')) +
                 p9.geom_line() +
                 p9.geom_point() +
                 p9.theme_bw() +
                 p9.labs(x="step", y="Total Return", color="Model") +
                 p9.theme(text=p9.element_text(size=20), legend_position=legend_pos)
                 ).draw(show=True, return_ggplot=True)
    return plot, fig


def gather_data(path_to_file):
    df = pd.read_csv(path_to_file)
    return df


def results_writer(results_file):
    f = open(results_file, 'w')
    writer = csv.writer(f)
    writer.writerow(['step', 'return', 'agent', 'run_id'])
    run_id = 'run' + str(time.time())
    return f, writer, run_id
