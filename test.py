import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 6]

fig, ax = plt.subplots()
ax.scatter(x, y, color='blue', marker='o', label='Data Points')

# Add labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Scatter Plot Example')

# Set aspect ratio of plot
plt.gca().set_aspect(0.8)

max_factor = 1.1
inf_factor = 1.15
y_max_orig = 631.25
y_max_tick = y_max_orig * max_factor
y_inf_tick = y_max_orig * inf_factor
y_buffer_space = y_max_orig * 1.2

# Alter original x and y limits with new y_max
plt.xlim(0, y_max_orig)
plt.ylim(0, y_max_tick)

# Create diagonal dashed line
plt.plot([0, y_max_tick], [0, y_max_tick], linestyle='dashed', linewidth=0.75, c='black')

# Create gray shaded area below line y=x
x = np.linspace(0, y_max_tick, 100)
y = x
plt.fill_between(x, y, color='lightgray', where=(y <= x), alpha=0.5, zorder=0)

# create solid horizontal line for infinity
plt.axvline(0, color='black')
plt.gca().axhline(y_inf_tick, linestyle='solid', linewidth=0.75, c='black')
# Create new y_ticks
y_ticks = [i for i in plt.yticks()[0]]
y_ticks[-1] = y_inf_tick
y_ticks.append(y_buffer_space)

# Create new y_labels with new y_ticks
y_labels = [str('{:.2f}'.format(round(i, 2))) for i in y_ticks]
y_labels[-2] = r'$\infty$'
y_labels[-1] = ''

# Set y_ticks and labels
ax.set_yticks(y_ticks, labels=y_labels)
plt.yticks()[1][-2].set_fontsize(18)


# Add legend
plt.legend()

# Show plot
plt.show()
