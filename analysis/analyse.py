from analysis_methods import plot_gg, average_over_runs, get_data

file_path = " "

data = get_data(file_path)
data = average_over_runs(data)

plot_gg(data, "bottom")
