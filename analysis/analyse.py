from analysis_methods import plot_gg, tidy_data_episodes, get_data, tidy_data_steps, plot_scatter

# file_path = "../results/smalltet"
# data = get_data(file_path)

import pandas as pd
csv_data = pd.read_csv('../results/ttb/ExploitV_0.csv')
data = pd.DataFrame({'agent': 'v' * len(csv_data),
                     'agent_number': '0' * len(csv_data),
                     'step': csv_data['step'],
                     'episode': csv_data['episode_number'],
                     'return': csv_data['return']})
print("done stage 1")

ep_data = tidy_data_episodes(data)

print("done stage 2")

plot_gg(ep_data, "right")

# step_data = tidy_data_steps(data)
#
# plot_scatter(step_data, "right")

# plot, fig = plot_gg(data, "bottom")
# fig.savefig('image1.png', dpi=300, bbox_inches='tight')
#
# plot, fig = plot_gg(data, "none")
# fig.savefig('image2.png', dpi=300, bbox_inches='tight')
