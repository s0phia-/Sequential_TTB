from analysis_methods import plot_gg, average_over_runs, get_data

file_path = "../results/hex_run_1"

data = get_data(file_path)
data = average_over_runs(data)

plot_gg(data, "right")

plot, fig = plot_gg(data, "bottom")
fig.savefig('image1.png', dpi=300, bbox_inches='tight')

plot, fig = plot_gg(data, "none")
fig.savefig('image2.png', dpi=300, bbox_inches='tight')
