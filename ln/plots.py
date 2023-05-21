import os
import random
import matplotlib as mpl
if not "DISPLAY" in os.environ.keys():
    mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from math import floor, log10

label_size = 10
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['legend.numpoints'] = 1


def get_counts(samples, normalize=False):
    """
    Counts the number of occurrences of each value in samples.
    :param samples: list with the samples
    :param normalize: boolean, indicates if counts have to be normalized
    :return: list of two lists: first list returns x values (unique values in samples), second list returns occurrence
    counts
    """

    xs, ys = np.unique(samples, return_counts=True)

    if normalize:
        total = sum(ys)
        ys = [float(y)/float(total) for y in ys]

    return [xs, ys]


def get_cdf(samples, normalize=False):
    """
    Compute the cumulative count over samples.
    :param samples: list with the samples
    :param normalize: boolean, indicates if counts have to be normalized
    :return: list of two lists: first list returns x values (unique values in samples), second list returns cumulative
    occurrence counts (number of samples with value <= xi).
    """

    [xs, ys] = get_counts(samples, normalize)
    ys = np.cumsum(ys)

    return [xs, ys]


def plot_distribution(xs, ys, title, xlabel, ylabel, log_axis=None, save_fig=False, legend=None, legend_loc=1,
                      font_size=20, y_sup_lim=None, x_sup_lim=None, subplot=None, label_rotation=None,
                      calculated_values=None):
    """
    Plots a set of values (xs, ys) with matplotlib.
    :param xs: either a list with x values or a list of lists, representing different sample sets to be plotted in the
    same figure.
    :param ys: either a list with y values or a list of lists, representing different sample sets to be plotted in the
    same figure.
    :param title: String, plot title
    :param xlabel: String, label on the x axis
    :param ylabel: String, label on the y axis
    :param log_axis: String (accepted values are False, "x", "y" or "xy"), determines which axis are plotted using
    logarithmic scale
    :param save_fig: String, figure's filename or False (to show the interactive plot)
    :param legend: list of strings with legend entries or None (if no legend is needed)
    :param legend_loc: integer, indicates the location of the legend (if present)
    :param font_size: integer, title, xlabel and ylabel font size
    :param y_sup_lim: float, y axis superior limit (if None or not present, use default matplotlib value)
    :param calculated_values: contains means and medians of the calculated data
    :return: None
    :type: None
    """
    mpl.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    if subplot is None:
        plt.figure(figsize=(12, 10))
        ax = plt.subplot(111)
    else:
        if subplot[2] == 1:
            plt.figure(figsize=(28, 36))

        ax = plt.subplot(subplot[0], subplot[1], subplot[2])
    plt.grid(b=True, color='0.85', linestyle='-', linewidth=0.25)
    # Plot data
    if not (isinstance(xs[0], list) or isinstance(xs[0], np.ndarray)):
        plt.plot(xs, ys)  # marker='o'
    else:
        pivot = 0
        x = 0
        for i in range(len(xs)):
            plt.plot(xs[i], ys[i], linestyle='solid')  # marker='o'
            if isinstance(calculated_values, list) and all(v is not None for v in calculated_values):
                calc_values = calculated_values[0]['1st'] if x in [0, 1] else calculated_values[1]['2nd']
                set_mean_median(xs, ys, calc_values, legend, i, pivot, "".rjust(7))

            if isinstance(calculated_values, dict) and calculated_values is not None:
                set_mean_median(xs, ys, calculated_values, legend, i, pivot)
            pivot += 3
            x += 1

    # Plot title and xy labels
    plt.title(title, {'color': 'k', 'fontsize': font_size})
    plt.ylabel(ylabel, {'color': 'k', 'fontsize': font_size})
    plt.xlabel(xlabel, {'color': 'k', 'fontsize': font_size})

    # Rotate x labels
    if label_rotation:
        plt.xticks(rotation=label_rotation)

    # Change axis to log scale
    if log_axis == "y":
        plt.yscale('log')
    elif log_axis == "x":
        plt.xscale('log')
    elif log_axis == "xy":
        plt.loglog()

    # Include legend
    if legend:
        lgd = ax.legend(legend, loc=legend_loc)

    # Force y limit
    if y_sup_lim:
        ymin, ymax = plt.ylim()
        plt.ylim(ymin, y_sup_lim)

    if x_sup_lim:
        xmin, xmax = plt.xlim()
        plt.xlim(xmin, x_sup_lim)

    # Output result
    if not save_fig:
        pass
    elif save_fig == "show":
        plt.show()
    else:
        plt.savefig(save_fig + '.pdf', format='pdf', dpi=None, bbox_inches=None)
        plt.close()


def plot_chart_scatter(xs, ys, title, xlabel, ylabel, font_size, save_fig=False, legend=None, legend_loc=None):
    colors = ['green', 'orange', 'sienna', 'darkkhaki', 'olive', 'yellow']
    marker = ['o', 'v']
    # for i in range(len(xs)):
    # plt.bar(xs[0], ys[0], width, color=colors[0])
    # plt.bar(xs[1], ys[1], width, color=colors[1])
    # pl.xticks(xs[0])
    # plt.xticks([r + width for r in range(len(xs[0]))])
    figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='r')
    # sns.barplot(x=xs[0], y=ys[0])
    # plt.xscale('log')

    for i in range(len(xs)):
        plt.scatter(xs[i], ys[i], marker=marker[i], color=colors[i])
    plt.xscale('log')
    plt.title(title, {'color': 'k', 'fontsize': font_size})
    plt.xlabel(xlabel, {'color': 'k', 'fontsize': font_size})
    plt.ylabel(ylabel, {'color': 'k', 'fontsize': font_size})
    plt.legend(legend, loc=legend_loc)

    plt.savefig(save_fig + '.pdf', format='pdf', dpi=None, bbox_inches=None)
    plt.close()


def plot_chart_bar(data, labels, title, xlabel, ylabel, font_size, save_fig=False, legend=None, legend_loc=None,
                   is_format=False):
    colors = ['green', 'orange', 'sienna', 'darkkhaki', 'olive', 'yellow']
    x = np.arange(len(labels))
    width = 0.45  # the width of the bars: can also be len(x) sequence

    font_dict = {'fontsize': font_size}

    plt.rcParams.update({'font.size': 3})
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/2, data[0], width)
    bar2 = ax.bar(x + width/2, data[1], width)

    ax.set_xlabel(xlabel, fontdict=font_dict)
    ax.set_ylabel(ylabel, fontdict=font_dict)
    ax.set_title(title, fontsize=font_size)
    ax.legend(legend, loc=legend_loc, fontsize=font_size-2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.yaxis.offsetText.set_fontsize(font_size)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    rotation = 'vertical' if is_format else 'horizontal'
    fmt = "%.3e" if is_format else "%.1e"
    labels_bar1 = []
    labels_bar2 = []
    for i in range(len(data[0])):
        labels_bar1.append(sci_notation(data[0][i], 3 if is_format else 1))
        labels_bar2.append(sci_notation(data[1][i], 3 if is_format else 1))

    ax.bar_label(bar1, labels=labels_bar1, padding=1, rotation=rotation, fmt=fmt)
    ax.bar_label(bar2, labels=labels_bar2, padding=1, rotation=rotation, fmt=fmt)

    fig.tight_layout()

    plt.savefig(save_fig + '.pdf', format='pdf', dpi=None, bbox_inches=None)
    plt.close()


def sci_notation(num, decimal_digits=1, precision=None, exponent=None, x_label=False):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if num == 0:
        return "0"
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    if x_label and coeff == 1.0:
        return r"$10^{}$".format(exponent)
    else:
        return r"${0:.{2}f}\times10^{{{1:d}}}$".format(coeff, exponent, precision)


def set_mean_median(xs: list, ys: list, calculated_values: dict, legend: str, i: int, pivot: int, spaces=""):
    # Percentile value
    if not any(isinstance(el, list) for el in xs):  # and log_axis == "x":
        color = 'blue' if i == 0 else 'tomato' if i == 1 else 'forestgreen' if i == 2 else 'brown'
        marker_median = 'o' if i == 0 else 'v' if i == 1 else '+' if i == 2 else '>'
        marker_mean = 'P' if i == 0 else '*' if i == 1 else 'x' if i == 2 else '<'
        val = calculated_values
        if len(xs) == 2 or len(xs) == 4:
            if "unrestricted" in calculated_values:
                val = calculated_values["unrestricted"] if i in [0, 2] else calculated_values["restricted"]
            elif "enabled" in calculated_values:
                val = calculated_values["enabled"] if i in [0, 2] else calculated_values["disabled"]
        # median = np.median(xs[i])
        # percentiles = np.where(xs[i] == median)
        # plt.plot(median, ys[i][percentiles[0]], marker=marker_median, color=color,
        #          linestyle='none')
        # mean = round(np.mean(xs[i]))
        # percentiles = np.where(xs[i] == min(xs[i], key=lambda x:abs(x-mean)))
        # mean = xs[i][percentiles[0]]
        # plt.plot(mean, ys[i][percentiles[0]], marker=marker_mean, color=color,
        #          linestyle='none')
        real_median = set_percentile_values(val["median"], xs, ys, marker_median, color, i)

        real_mean = set_percentile_values(val["mean"], xs, ys, marker_mean, color, i)

        legend.insert(pivot + 1, spaces + "Median: {} ".format(round(real_median, 5)))
        legend.insert(pivot + 2, spaces + "Mean: {} ".format(round(real_mean, 5)))


def set_percentile_values(real, xs, ys, marker, color, i):
    real = [real] if isinstance(real, float) else real
    j = 0 if len(xs) == 2 or len(xs) == 4 else i
    percentiles = np.where(xs[i] == min(xs[i], key=lambda x: abs(x - real[j])))
    median = xs[i][percentiles[0]]
    plt.plot(median, ys[i][percentiles[0]], marker=marker, color=color,
             linestyle='none')
    return real[j]


def plot_scatter(x, y, z, title, xlabel, ylabel, zlabel, save_fig):
    css4_colors = {
        'black': '#000000',
        'lightcoral': '#F08080',
        'brown': '#A52A2A',
        'red': '#FF0000',
        'tomato': '#FF6347',
        'sienna': '#A0522D',
        'sandybrown': '#F4A460',
        'tan': '#D2B48C',
        'goldenrod': '#DAA520',
        'gold': '#FFD700',
        'darkkhaki': '#BDB76B',
        'olive': '#808000',
        'yellow': '#FFFF00',
        'olivedrab': '#6B8E23',
        'yellowgreen': '#9ACD32',
        'chartreuse': '#7FFF00',
        'limegreen': '#32CD32',
        'green': '#008000',
        'mediumspringgreen': '#00FA9A',
        'turquoise': '#40E0D0',
        'lightseagreen': '#20B2AA',
        'darkslategray': '#2F4F4F',
        'aqua': '#00FFFF',
        'deepskyblue': '#00BFFF',
        'steelblue': '#4682B4',
        'dodgerblue': '#1E90FF',
        'slategray': '#708090',
        'navy': '#000080',
        'blue': '#0000FF',
        'mediumslateblue': '#7B68EE',
        'blueviolet': '#8A2BE2',
        'darkviolet': '#9400D3',
        'mediumorchid': '#BA55D3',
        'violet': '#EE82EE',
        'fuchsia': '#FF00FF',
        'mediumvioletred': '#C71585',
        'deeppink': '#FF1493',
        'palevioletred': '#DB7093',
        'crimson': '#DC143C',
        'pink': '#FFC0CB'}

    mpl.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    axis_font = {'fontname': 'sans-serif', 'size': '12'}

    # Creating figure
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection='3d')

    # Add x, y gridlines
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.2,
            alpha=1)

    # Creating color map
    c_map = plt.get_cmap('hsv')
    colors = random.sample(list(css4_colors), len(x)) if len(x) <= len(css4_colors) else \
        list(css4_colors) * (len(x) // len(css4_colors)) + list(css4_colors)[:(len(x) % len(css4_colors))]

    sizes = [2000 if s >= 1000 else s * 100 for s in x]

    sctt = ax.scatter3D(x, y, z, zdir=y,
                        alpha=0.8,
                        c=colors,
                        cmap=c_map,
                        marker='o',
                        s=sizes)

    plt.title(title, **axis_font)
    ax.set_xlabel(xlabel, **axis_font)
    ax.set_ylabel(ylabel, **axis_font)
    ax.set_zlabel(zlabel, **axis_font)

    # ax.view_init(90, -120)
    max_nodes_index = max(range(len(x)), key=x.__getitem__)
    max_capacity_index = max(range(len(y)), key=y.__getitem__)
    max_fee_index = max(range(len(z)), key=z.__getitem__)
    scatter1_proxy = mpl.lines.Line2D([0], [0], linestyle="none", c=colors[max_nodes_index], marker='o')
    scatter2_proxy = mpl.lines.Line2D([0], [0], linestyle="none", c=colors[max_capacity_index], marker='o')
    scatter3_proxy = mpl.lines.Line2D([0], [0], linestyle="none", c=colors[max_fee_index], marker='o')
    ax.legend([scatter1_proxy, scatter2_proxy, scatter3_proxy], ['Max Nodes: {}'.format(x[max_nodes_index]),
                                                                 'Avg. Capacity: {}'.format(y[max_capacity_index]),
                                                                 'Avg. Fee: {}'.format(z[max_fee_index])], numpoints=1)
    ax.figure.colorbar(sctt, ax=ax, shrink=0.5, aspect=10)

    plt.savefig(save_fig + '.pdf', format='pdf', dpi=None, bbox_inches=None)
    plt.close()


def plot_heatmap(m, x_values, y_values, title, xlabel, ylabel, save_fig=False, label_rotation=None):
    mpl.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    vmin, vmax = 0, 1
    im = ax.imshow(m.transpose(), origin='lower', vmin=vmin, vmax=vmax, cmap='RdYlGn_r', alpha=0.7)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)  # One colorbar per subplot
    #cbar.ax.set_ylabel("$\widetilde{TBT}$", rotation=-90, va="bottom")

    # Loop over data dimensions and create text annotations.
    mt = m.transpose()
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            text = ax.text(j, i, "{:.2f}".format(mt[i, j]), ha="center", va="center", color="black", alpha=1)

    # Named ticks
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_xticklabels(x_values)
    ax.set_yticklabels(y_values)

    # Axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Rotate x labels
    if label_rotation:
        plt.xticks(rotation=label_rotation)

    fig.tight_layout()
    plt.title(title)

    # Output result
    if not save_fig:
        pass
    elif save_fig == "show":
        plt.show()
    else:
        plt.savefig(save_fig + '.pdf', format='pdf', dpi=600)
        plt.close()
