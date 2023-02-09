from analysis_methods import plot_gg, tidy_data_episodes, get_data, tidy_data_steps, plot_scatter

file_path = "../results/test"

data = get_data(file_path)
ep_data = tidy_data_episodes(data)

plot_gg(ep_data, "right")

step_data = tidy_data_steps(data)

plot_scatter(step_data, "right")

# plot, fig = plot_gg(data, "bottom")
# fig.savefig('image1.png', dpi=300, bbox_inches='tight')
#
# plot, fig = plot_gg(data, "none")
# fig.savefig('image2.png', dpi=300, bbox_inches='tight')
