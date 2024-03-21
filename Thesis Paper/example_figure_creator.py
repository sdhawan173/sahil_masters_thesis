import os
import math
import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.spatial import Delaunay, Voronoi
import gudhi
import pickle


def create_base_plot(persistence_points, title, max_dim=2):
    # Create Plot
    fig, ax = plt.subplots()
    cmap = matplotlib.cm.Set1.colors
    plt.gca().set_aspect(0.8)
    plt.rc(group='axes', labelsize=12)
    plt.xlabel('Birth')
    plt.ylabel('Death')

    for point in persistence_points:
        ax.scatter(point[1][0], point[1][1], color=cmap[point[0]], alpha=0.75, s=45, zorder=10, clip_on=False)
    ax.set_title('Persistence Diagram of\n{}'.format(title))
    ax.legend(
        handles=[
            mpatches.Patch(
                color=cmap[dim],
                label=r'$H_{}$'.format(str(dim))) for dim in list(range(0, max_dim + 1))
        ],
        loc=4
    )

    # Create frequency labels for overlapping points
    frequency_values, frequency_coords = persistence_frequencies(persistence_points)

    # Create annotations for overlapping points
    for index, annotation in enumerate(frequency_values):
        x_coord, y_coord = frequency_coords[index]
        ax.annotate(annotation, (x_coord, y_coord), xytext=(5, -12), textcoords='offset points', fontsize=12, zorder=11)
    return fig, ax


def persistence_frequencies(persistence_points):
    """
    :param persistence_points: list of persistence diagram points
    return: list of frequencies for points that occur more than once, coordinates of points that occur more than once.
    """
    # Create list of indices for unique points in the persistence points list
    unique_points = set((point[1][0], point[1][1]) for point in persistence_points)
    unique_indices = [index for index, point in enumerate(persistence_points) if
                      (point[1][0], point[1][1]) in unique_points]

    # Create list of frequencies for overlapping unique points
    frequency_values = []
    frequency_coords = []
    for index in range(len(persistence_points)):
        point_count = 0

        # Count number of overlapping points
        if unique_indices.__contains__(index):
            point_count = persistence_points.count(persistence_points[index])

        # Add total of overlapping points and respective coordinates
        if point_count > 1:
            append_value = str(point_count)
            frequency_values.append(append_value)
            frequency_coords.append([persistence_points[index][1][0],
                                     persistence_points[index][1][1]])
    return frequency_values, frequency_coords


def infinity_handler(persistence_points, max_factor=1.1, inf_factor=1.15, max_dim=2):
    persistence_points = [(point[0], (point[1][0], point[1][1])) for point in persistence_points if point[0] <= max_dim]
    y_values = [point[1][1] for point in persistence_points]

    # Set value for infinity line/infinity annotation by increasing tick height of max y value
    y_max_orig = None
    y_max_tick = None
    y_inf_tick = None
    inf_bool = False
    if not len(np.unique(y_values)) == 1:
        if y_values.__contains__(math.inf):
            # If persistence points contain infinity, repalce math.inf with math.nan
            # Get maximum non-infinity y_value
            y_max_orig = max([val for val in y_values if not math.isinf(val)])

            # Increase maximum y_value by increase factors to create location for infinity, set alternate y_max
            y_max_tick = y_max_orig * max_factor
            y_inf_tick = y_max_orig * inf_factor

            # replace math.inf value with new y_max value
            for index in range(len(y_values)):
                if math.isinf(y_values[index]):
                    y_values[index] = y_inf_tick
    elif len(np.unique(y_values)) == 1:
        y_values = [2.25] * len(y_values)
        y_max_orig = 2
        y_max_tick = 2
        y_inf_tick = 2.25
        if len(y_values) == 1:
            inf_bool = True

    # Update persistence points with new y values
    new_persistence = []
    for point, y_val in zip(persistence_points, y_values):
        new_persistence.append([point[0], [point[1][0], y_val]])
    return new_persistence, y_max_orig, y_max_tick, y_inf_tick, inf_bool


def mpl_tick_handler(ax, y_max_orig, y_max_tick, y_inf_tick, inf_bool):
    # Create gap between infinity line and top of plot
    y_buffer_space = y_max_orig * 1.2

    # Alter original x and y limits with new y_max
    plt.xlim(0, y_max_orig)
    plt.ylim(0, y_max_tick)

    if not inf_bool:
        # Create new y_ticks
        y_ticks = [i for i in plt.yticks()[0]]
        y_ticks[-1] = y_inf_tick
        y_ticks.append(y_buffer_space)

        # Create new y_labels with new y_ticks
        y_labels = [str('{:.2f}'.format(round(i, 2))) for i in y_ticks]
        y_labels[-2] = r'$\infty$'
        y_labels[-1] = ''
    elif inf_bool:
        # Create new x_ticks
        x_ticks = [i for i in plt.yticks()[0]]

        # Create new x_labels with new x_ticks
        x_labels = ['' for _ in x_ticks]
        x_labels[0] = str(0)
        ax.set_xticks(x_ticks, labels=x_labels)

        # Create new y_ticks
        y_ticks = [i for i in plt.yticks()[0]]
        y_ticks[-1] = y_inf_tick
        y_ticks.append(y_buffer_space)

        # Create new y_labels with new y_ticks
        y_labels = ['' for _ in y_ticks]
        y_labels[0] = str(0)
        y_labels[-2] = r'$\infty$'

    # Set y_ticks and labels
    ax.set_yticks(y_ticks, labels=y_labels)
    plt.yticks()[1][-2].set_fontsize(18)

    # Create diagonal dashed line
    plt.plot([0, y_max_tick], [0, y_max_tick], linestyle='dashed', linewidth=0.75, c='black')

    # Create gray shaded area below line y=x
    x = np.linspace(0, y_max_tick, 100)
    y = x
    plt.fill_between(x, y, color='lightgray', where=(y <= x), alpha=0.5, zorder=0)

    # create solid horizontal line for infinity
    plt.axvline(0, color='black')
    plt.gca().axhline(y_inf_tick, linestyle='solid', linewidth=0.75, c='black')


def save_persistence_points(persistence_points, save_file_name):
    with open(save_file_name, 'w') as file:
        for line in persistence_points:
            file.write(f"{line}\n")


def plot_persdia_main(persistence_points, title, save_dir, show_plot=False):
    """
    plots the persistence diagram of the gudhi_simplex_tree generated by smm.create_simplex.
    :param persistence_points: persistence points of simplex tree from selected complex
    :param title: Title of plot
    :param save_dir: String of full file path (including extension) to save plot file to
    :param show_plot: Boolean to show plot, default=False
    """
    print('\nPlotting Persistence Diagram ...')

    # Adjust dimensions to be shown
    persistence_points, y_max_orig, y_max_tick, y_inf_tick, inf_bool = infinity_handler(persistence_points)

    # Create initial plot
    fig, ax = create_base_plot(persistence_points, title)

    # Assign values for certain ticks
    mpl_tick_handler(ax, y_max_orig, y_max_tick, y_inf_tick, inf_bool)

    file_path = save_dir.split('.')[0]
    extension = save_dir.split('.')[1]
    plt.savefig(file_path + '.' + extension, bbox_inches='tight', dpi=300, format=extension)
    print('Persistence Diagram saved to {}'.format(save_dir))

    if show_plot:
        plt.show()


def save_plot_state(persistence_points, y_max_orig, y_max_tick, y_inf_tick, inf_bool, save_dir, plot_title):
    save_state = (persistence_points, y_max_orig, y_max_tick, y_inf_tick, inf_bool, plot_title)
    save_file_name = '{}.pkl'.format(save_dir)
    with open(save_file_name, 'wb') as file:
        pickle.dump(save_state, file)


def load_plot_state(save_dir):
    with open(save_dir + '.pkl', 'rb') as file:
        restored_variables = pickle.load(file)
    os.remove(save_dir + '.pkl')
    return restored_variables


def multi_plot_persdia(persistence_points, plot_title, save_path, ext='png', multi_run=False,
                       override_orig=None, y_max_tick=None, y_inf_tick=None, all_y_inf_tick=None, inf_bool=None):
    print('\nPlotting Persistence Diagram ...')
    y_max_orig = None

    if override_orig is None:
        # Adjust dimensions to be shown
        persistence_points, y_max_orig, y_max_tick, y_inf_tick, inf_bool = infinity_handler(persistence_points)

        if multi_run:
            save_plot_state(persistence_points, y_max_orig, y_max_tick, y_inf_tick, inf_bool, save_path, plot_title)
    elif not multi_run:
        if all_y_inf_tick is not None:
            change_indices = []
            for point in persistence_points:
                for inf_tick_value in all_y_inf_tick:
                    if point[1][1] == inf_tick_value:
                        change_indices.append(persistence_points.index(point))
            for index in change_indices:
                persistence_points[index][1][1] = y_inf_tick

    # Create initial plot
    fig, ax = create_base_plot(persistence_points, plot_title)

    if override_orig is not None:
        y_max_orig = override_orig

    # Assign values for certain ticks
    mpl_tick_handler(ax, y_max_orig, y_max_tick, y_inf_tick, inf_bool)

    plt.savefig(save_path + '.' + ext, bbox_inches='tight', dpi=300, format=ext)
    print('Persistence Diagram saved to {}'.format(save_path + '.' + ext))


def create_point_cloud(scale=0.12, size=5, interval=0.1, make_plot=False):
    np.random.seed(666)
    x = [np.cos(t * 2 * np.pi) for t in np.arange(0, size, interval)]
    y = [np.sin(t * 2 * np.pi) for t in np.arange(0, size, interval)]
    x_noise = np.random.normal(scale=scale, size=len(x))
    y_noise = np.random.normal(scale=scale, size=len(y))
    x = x + x_noise
    y = y + y_noise
    if make_plot:
        plt.scatter(x, y, c='black')
        plt.title('Sample Data Point Cloud')
        plt.gca().set_aspect(1, adjustable='box')
        plt.savefig(os.getcwd() + '/point_cloud_plot.png', bbox_inches='tight', dpi=300, format='png')
        # plt.show()
    return np.array(x), np.array(y)


def vr_point_cloud(x, y, r, title, save_name):
    # noinspection PyUnresolvedReferences
    vr_complex = gudhi.RipsComplex(points=zip(x, y), max_edge_length=r)
    vr_simplex_tree = vr_complex.create_simplex_tree(max_dimension=3)
    vr_persistence = vr_simplex_tree.persistence()
    plot_persdia_main(vr_persistence, title, os.getcwd() + '/' + save_name + '.png')
    # plt.show()


def visualize_vr_complexes(x, y, radii=(0, 0.25, 0.5, 1.0)):
    # Vietoris-Rips Complex at different radii
    for r in radii:
        # Create Rips complex
        # noinspection PyUnresolvedReferences
        rips = gudhi.RipsComplex(points=zip(x, y), max_edge_length=r)
        simplex_tree = rips.create_simplex_tree(max_dimension=2)

        # Get simplices
        simplices = np.array([simplex[0] for simplex in simplex_tree.get_skeleton(2) if len(simplex[0]) == 3])

        # Plot the complex
        plt.figure()
        plt.scatter(x, y, c='black')
        for simplex in simplices:
            plt.plot([x[simplex[0]], x[simplex[1]], x[simplex[2]], x[simplex[0]]],
                     [y[simplex[0]], y[simplex[1]], y[simplex[2]], y[simplex[0]]], 'k-')
            plt.fill([x[simplex[0]], x[simplex[1]], x[simplex[2]], x[simplex[0]]],
                     [y[simplex[0]], y[simplex[1]], y[simplex[2]], y[simplex[0]]], 'r', alpha=0.3)

        plt.title(f'Vietoris-Rips Complex at $r$={r}', fontsize=15)
        plt.gca().set_aspect(1, adjustable='box')
        plt.savefig(os.getcwd() + '/point_cloud_plot_r{}.png'.format(str(r).replace('.', '_')),
                    bbox_inches='tight', dpi=300, format='png')
        # plt.show()

        if r == 0.5:
            vr_point_cloud(x, y, r,
                           'Sample Data Point Cloud VR Complex, $r$={}'.format(str(r)),
                           'point_cloud_persdia_vr_0_5')


def black_hole_example():
    # Image dimensions
    width, height = 50, 50

    # Create grid of coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Circle parameters
    circle_center = (width // 2, height // 2)
    circle_radius = 10

    # Calculate distance from each pixel to circle center
    distance = np.sqrt((x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2)

    # Create image array with white background
    image = np.ones((height, width), dtype=bool)

    # Set pixels inside the circle to black
    image[distance <= circle_radius] = False

    plt.imshow(image, cmap='gray')
    # plt.axis('off')
    plt.title('Image Sample Data')
    plt.gca().set_aspect(1, adjustable='box')
    plt.savefig(os.getcwd() + '/image_data_plot.png', bbox_inches='tight', dpi=300, format='png')
    # plt.show()

    # Convert image to list of 2D points
    points = np.column_stack(np.where(image))

    # Create Rips complex
    # noinspection PyUnresolvedReferences
    alpha = gudhi.AlphaComplex(points=points)
    simplex_tree = alpha.create_simplex_tree()
    persistence = simplex_tree.persistence()
    plot_persdia_main(persistence, 'Image Sample Data Alpha Complex', os.getcwd() + '/image_data_persdia.png')


def visualize_delaunay_voronoi(x, y):
    points = np.array([(xp, yp) for xp, yp in zip(x, y)])
    # Create Delaunay triangulation
    delaunay_complex = Delaunay(points)
    persistence = []
    for _ in range(len(delaunay_complex.simplices) - 1):
        persistence.append([0, (0, math.inf)])
    plot_persdia_main(persistence, 'Delaunay Complex of Sample Data Point Cloud',
                      os.getcwd() + '/point_cloud_persdia_del.png')

    # Plot Delaunay triangulation and Voronoi diagram overlayed
    plt.figure(figsize=(8, 6))
    scatter_plot = plt.scatter(x, y, c='black', zorder=4)
    plt.title('Delaunay Triangulation and Voronoi Diagram of\n Sample Data Point Cloud', fontsize=15)
    plt.gca().set_aspect(1, adjustable='box')

    # Get the limits of the original scatter plot
    x_min, x_max = scatter_plot.axes.get_xlim()
    y_min, y_max = scatter_plot.axes.get_ylim()

    # Set the limits of the axes to match the original scatter plot
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.triplot(points[:, 0], points[:, 1], delaunay_complex.simplices, color='black', zorder=2)
    plot_voronoi(points, plt.gca())
    plt.plot(points[:, 0], points[:, 1], 'o', markersize=5, color='none', zorder=1)

    # Plot triangles
    for simplex in delaunay_complex.simplices:
        triangle = points[simplex]
        plt.fill(triangle[:, 0], triangle[:, 1], 'r', alpha=0.625, zorder=1)

    plt.tight_layout()
    plt.savefig(os.getcwd() + '/point_cloud_plot_del.png',
                bbox_inches='tight', dpi=300, format='png')


def plot_voronoi(points, ax):
    vor = Voronoi(points)
    num_cells = len(vor.regions)
    cmap = mcolors.LinearSegmentedColormap.from_list('white_to_black', [(1, 1, 1), (0, 0, 0)])  # Create custom colormap

    # Calculate the bounding box of the Voronoi diagram
    xmin, xmax = np.min(vor.vertices[:, 0]), np.max(vor.vertices[:, 0])
    ymin, ymax = np.min(vor.vertices[:, 1]), np.max(vor.vertices[:, 1])

    # Define additional points outside the bounding box
    extra_points = np.array([
        [xmin - 10, ymin - 10],
        [xmin - 10, ymax + 10],
        [xmax + 10, ymin - 10],
        [xmax + 10, ymax + 10]
    ])

    # Combine original and extra points
    all_points = np.append(vor.points, extra_points, axis=0)

    # Recalculate Voronoi diagram
    vor = Voronoi(all_points)

    for i, region in enumerate(vor.regions):
        polygon = [vor.vertices[i] for i in region]
        ax.fill(*zip(*polygon), edgecolor='black', facecolor='none', alpha=1, zorder=4)
        ax.fill(*zip(*polygon), edgecolor='none', facecolor=cmap(i / num_cells), alpha=0.75, zorder=0)


def visualize_alpha_complexes(x, y, radii):
    # Vietoris-Rips Complex at different radii
    for iter_num, r_squared in enumerate(radii):
        # Create Alpha complex
        # noinspection PyUnresolvedReferences
        alpha_complex = gudhi.AlphaComplex(points=list(zip(x, y)))
        simplex_tree = alpha_complex.create_simplex_tree()
        persdia_title = 'Alpha Complex, Sample Data Point Cloud'
        save_persdia_name = os.getcwd() + '/point_cloud_persdia_alpha_voronoi'.format(str(r_squared).replace(".", "_"))
        if r_squared is not None:
            simplex_tree.prune_above_filtration(r_squared)
        elif r_squared is None:
            persdia_title = 'Alpha Complex, Sample Data Point Cloud'.format(r_squared)
        persistence = simplex_tree.persistence()
        if r_squared is None:
            print(persistence)
            plot_persdia_main(persistence, persdia_title, save_persdia_name + '.png')

        # Plot the complex
        plt.figure()
        scatter_plot = plt.scatter(x, y, c='black', zorder=4)
        # Get the limits of the base scatter plot
        x_min, x_max = scatter_plot.axes.get_xlim()
        y_min, y_max = scatter_plot.axes.get_ylim()
        full_min = min(x_min, y_min)
        full_max = max(x_max, y_max)
        # Set the limits of the axes to match the base scatter plot
        plt.xlim(full_min, full_max)
        plt.ylim(full_min, full_max)

        edge_simplices = np.array([simplex[0] for simplex in simplex_tree.get_skeleton(1) if len(simplex[0]) == 2])
        for point in edge_simplices:
            plt.plot([x[point[0]], x[point[1]]], [y[point[0]], y[point[1]]], 'k-')

        # plot filled-in triangles
        tri_simplices = np.array([simplex[0] for simplex in simplex_tree.get_skeleton(2) if len(simplex[0]) == 3])
        for point in tri_simplices:
            # draw triangle lines
            plt.plot([x[point[0]], x[point[1]], x[point[2]], x[point[0]]],
                     [y[point[0]], y[point[1]], y[point[2]], y[point[0]]], 'k-')
            # fill in triangle
            plt.fill([x[point[0]], x[point[1]], x[point[2]], x[point[0]]],
                     [y[point[0]], y[point[1]], y[point[2]], y[point[0]]], 'r_squared', alpha=0.625)

        # Calculate Voronoi diagram
        points = np.array([(xp, yp) for xp, yp in zip(x, y)])
        plot_voronoi(points, plt.gca())

        plt.gca().set_aspect(1, adjustable='box')
        plot_title = ''
        plot_save_name = ''
        if r_squared is not None:
            # Draw circles around points
            for x_point, y_point in zip(x, y):
                plt.gca().add_patch(
                    plt.Circle((x_point, y_point),
                               math.sqrt(r_squared),
                               fill=False,
                               color='black',
                               alpha=1,
                               zorder=1)
                )
            plot_title = ('Alpha Complex, Voronoi Diagram:\n'
                          'Point Cloud, at $r_squared^2$={}, $r_squared={}$').format(str(round(r_squared, 4)), str(round(math.sqrt(r_squared), 4)))
            plot_save_name = os.getcwd() + '/point_cloud_plot_alpha_{}.png'.format(iter_num)
        elif r_squared is None:
            plot_title = ('Alpha Complex, Voronoi Diagram:\n'
                          'Point Cloud (Completed)')
            plot_save_name = os.getcwd() + '/point_cloud_plot_alpha_voronoi.png'
        plt.title(plot_title, fontsize=15)
        plt.savefig(plot_save_name, bbox_inches='tight', dpi=300, format='png')


def circumcenter_radius(a, b):
    # Calculate midpoints
    midAB = (a + b) / 2

    # Calculate slopes
    slope_AB = (b[1] - a[1]) / (b[0] - a[0])

    # Calculate perpendicular bisector slope
    slope_perpendicular = -1 / slope_AB

    # Calculate intercepts
    intercept_perpendicular = midAB[1] - slope_perpendicular * midAB[0]

    # Calculate circumcenter
    circumcenter_x = (slope_AB * a[0] - slope_perpendicular * midAB[0] + midAB[1] - a[1]) / (
                slope_AB - slope_perpendicular)
    circumcenter_y = slope_perpendicular * circumcenter_x + intercept_perpendicular
    circumcenter = np.array([circumcenter_x, circumcenter_y])

    # Calculate circumradius
    circumradius = np.linalg.norm(circumcenter - a)

    return circumcenter, circumradius


def gabriel(n=5):
    min_distance=0.75
    np.random.seed(666)
    points = np.random.rand(n, 2)
    for i in range(1, n):
        for j in range(i):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < min_distance:
                direction = points[i] - points[j]
                direction /= np.linalg.norm(direction)
                points[i] += (min_distance - dist) * direction

    # Calculate Delaunay triangulation
    tri = Delaunay(points)

    # Get all edges of the Delaunay triangulation
    edges = set()
    edge_colors = plt.cm.tab10.colors
    for simplex in tri.simplices:
        edges.add((min(simplex[0], simplex[1]), max(simplex[0], simplex[1])))
        edges.add((min(simplex[1], simplex[2]), max(simplex[1], simplex[2])))
        edges.add((min(simplex[0], simplex[2]), max(simplex[0], simplex[2])))

    # Plot Delaunay triangulation
    fig, ax = plt.subplots()
    ax.plot(
        points[:, 0], points[:, 1],
        'o',
        color='black',
        markersize=5,
        zorder=len(edges) + 2
    )

    # Plot each edge with a unique color
    for i, edge in enumerate(edges):
        A = points[edge[0]]
        B = points[edge[1]]
        circumcenter, circumradius = circumcenter_radius(A, B)
        ax.add_patch(
            plt.Circle(
                circumcenter,
                circumradius,
                color=edge_colors[i % len(edge_colors)],
                fill=False,
                linewidth=2,
                zorder=i
            )
        )
        ax.plot(
            [A[0], B[0]], [A[1], B[1]],
            color=edge_colors[i % len(edge_colors)],
            linewidth=2,
            zorder=i
        )

    # Annotate points
    labels = [string.ascii_uppercase[i] for i in range(len(points))]  # Using chr() to generate lowercase letters
    for i, label in enumerate(labels):
        ax.annotate(label,
                    (points[i][0], points[i][1]),
                    fontsize=15,
                    xytext=(5, -6.5),
                    textcoords='offset points',
                    color='black',
                    fontweight='bold',
                    zorder=len(edges)+3)

    plt.title('Gabriel Graph of Delaunay Triangulation', fontsize=15)
    plt.axis('equal')
    plot_save_name = os.getcwd() + '/gabriel_circles.png'
    plt.savefig(plot_save_name, bbox_inches='tight', dpi=300, format='png')


# x_points, y_points = create_point_cloud(scale=0.12, size=5, interval=0.1)
# black_hole_example()
# visualize_delaunay_voronoi(x_points, y_points)
x_points, y_points = create_point_cloud(scale=0.2, size=1, interval=0.1)
visualize_alpha_complexes(
    x_points,
    y_points,
    radii=(0, 0.14, 0.23076726048056004, 0.29, 0.30678737722699123, 0.3069246020010885, 0.8045271684187107, None)
)
# gabriel()
