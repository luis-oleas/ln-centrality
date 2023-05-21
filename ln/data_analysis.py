import os
import math
import json
import random
import numpy as np
import datetime
import itertools
from operator import itemgetter
from os import listdir, path, makedirs
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from ln.utils import load_pickle
from ln.plots import get_cdf, plot_distribution, plot_heatmap, plot_scatter, plot_chart_scatter, plot_chart_bar, \
    sci_notation

location = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))) + (
                            '\\data' if os.sys.platform == 'win32' else '/data')
results_folder = location + "/results/"
figs_folder = location + "/figs/"

METRIC_NAMES = ["degree", "strength", "incoming_strength", "outgoing_strength", "opsahl", "incoming_opsahl",
                "outgoing_opsahl", "betweenness", "weighted_betweenness", "weighted_betweenness_cap",
                "flow_based_betweenness", "current_flow_based_betweenness", "closeness", "weighted_closeness"]
# METRIC_NAMES = ["degree", "strength", "incoming_strength", "outgoing_strength", "opsahl", "incoming_opsahl",
# "outgoing_opsahl"]
# METRIC_NAMES = ["betweenness", "weighted_betweenness", "weighted_betweenness_cap", "flow_based_betweenness",
# "current_flow_based_betweenness"]
# METRIC_NAMES = ["closeness", "weighted_closeness"]
METRIC_NAMES_DEGREE = ["degree", "strength", "incoming_strength", "outgoing_strength", "opsahl", "incoming_opsahl",
                       "outgoing_opsahl"]
METRIC_NAMES_BET_CLO = ["betweenness", "weighted_betweenness", "weighted_betweenness_cap", "flow_based_betweenness",
                        "current_flow_based_betweenness", "closeness", "weighted_closeness"]
DATA_NAMES = ["total_channels", "total_channels_disabled", "max_channels", "max_channels_disabled"]
COMPONENT_NAMES = ["main_connected_component", "connected_component", "capacity", "fee"]

NORMALIZED_VERSIONS = ["betweenness", "weighted_betweenness", "weighted_betweenness_cap",
                       "current_flow_based_betweenness", "closeness", "weighted_closeness"]

METRIC_UNR_SUF = "_unrestricted"
METRIC_REST_SUF = "_restricted"
METRIC_NORM_SUF = "_norm"


def create_if_not_exists(folder):
    if not path.exists(folder):
        makedirs(folder)


def filename_to_date(str):
    s = str.replace("__", "_").split("_")
    return "{}-{}-{} {}:{}".format(s[1], s[2], s[3], s[4], s[5])


def filename_clean(str):
    return str[8:25]


def rmse(y_true, y_pred):
    '''
    Compute Root Mean Square Percentage Error between two arrays.
    '''
    loss = np.sqrt(
        np.mean(
            np.nan_to_num(
                np.square(((np.array(y_true) - np.array(y_pred)) / np.array(y_true)))
            ),
            axis=0)
    )

    return loss


def generate_results_files():
    # Get all graph names without extension nor G1/G2 suffix
    pickle_files = set([path.splitext(f)[0][:-3] for f in listdir(results_folder) if path.splitext(f)[1] == ".pickle"
                        and ('1st' not in path.splitext(f)[0] and '2nd' not in path.splitext(f)[0])])

    # Load date
    for filename in pickle_files:
        g1 = load_pickle(results_folder, filename + "_G1")
        # g2 = load_pickle(results_folder, filename + "_G2")
        r1 = {e[0]: e[1] for e in g1.nodes(True)}
        # A G1 i G2 hi ha els mateixos atributs. Per tant, només desem r1 al fitxer de resultats.
        # r2 = {e[0]: e[1] for e in g2.nodes(True)}
        with open(results_folder + filename + '_metrics.json', 'w') as fp:
            json.dump(r1, fp, indent=4)


def generate_results_files_simulations():
    # Get all graph names without extension nor G1/G2 suffix
    pickle_files = set([path.splitext(f)[0][:-7] for f in listdir(results_folder) if path.splitext(f)[1] == ".pickle"
                        and ('1st' in path.splitext(f)[0] or '2nd' in path.splitext(f)[0])])

    # Load date
    for filename in pickle_files:
        g1 = load_pickle(results_folder, filename + "_G1_1st")
        g2 = load_pickle(results_folder, filename + "_G1_2nd")
        r1 = {e[0]: e[1] for e in g1.nodes(True)}
        # A G1 i G2 hi ha els mateixos atributs. Per tant, només desem r1 al fitxer de resultats.
        r2 = {e[0]: e[1] for e in g2.nodes(True)}
        with open(results_folder + filename + '_metrics_1st.json', 'w') as fp:
            json.dump(r1, fp, indent=4)
        with open(results_folder + filename + '_metrics_2nd.json', 'w') as fp:
            json.dump(r2, fp, indent=4)


def generate_results_files_chunks():
    # Get all graph names without extension nor G1/G2 suffix
    pickle_files = set([path.splitext(f)[0][:-3] for f in listdir(results_folder) if path.splitext(f)[1] == ".pickle"])

    # Load date
    counter = 500
    numb = 0
    step = 100
    begin = 100

    for filename in pickle_files:
        g1 = load_pickle(results_folder, filename + "_G1")
        list_nodes = list(g1.nodes(data=True))
        end = len(list_nodes)
        # for i in range(510, 610, 10):
        for i in range(begin, end, step):
            r1 = {e[0]: e[1] for e in itertools.islice(list_nodes, counter, i)}
            counter = i
            numb = int(counter/step)
            with open(results_folder + filename + '_' + str(numb) + '_metrics.json', 'w') as fp:
                json.dump(r1, fp, indent=4)
        # if counter < 600:
        if counter < end:
            numb += 1
            r1 = {e[0]: e[1] for e in itertools.islice(list_nodes, counter, end)}
            with open(results_folder + filename + '_' + str(numb) + '_metrics.json', 'w') as fp:
                json.dump(r1, fp, indent=4)


def analyse_restriction(log_axis=None, is_extra_legend=False):
    # Get all graph names without extension nor G1/G2 suffix
    metrics_files = [f for f in listdir(results_folder) if f.endswith("_metrics.json")
                     or f.endswith("_metrics_1st.json")]

    # Process one file at a time (we are not looking at temporal evolution here)
    for metric_file in metrics_files:
        print("Processing file: {}".format(metric_file))
        fig_name = figs_folder + filename_clean(metric_file) + "_path_restrictions" + (
            "" if log_axis is None else "_log")

        with open(results_folder + metric_file, 'r') as f:
            r = json.load(f)

        for i, m in enumerate(METRIC_NAMES):
            print("\t" + m)
            values_unr = [round(0.0, 4) if math.isnan(v[m + METRIC_UNR_SUF]) else v[m + METRIC_UNR_SUF] for v in
                          r.values()]
            values_rest = [round(0.0, 4) if math.isnan(v[m + METRIC_REST_SUF]) else v[m + METRIC_REST_SUF] for v in
                           r.values()]
            [xs1, ys1] = get_cdf(values_unr, normalize=True)
            [xs2, ys2] = get_cdf(values_rest, normalize=True)
            mean_value_unr = np.mean(values_unr)
            median_value_unr = np.median(np.sort(values_unr))

            mean_value_res = np.mean(values_rest)
            median_value_res = np.median(np.sort(values_rest))
            calculated_values = None if not is_extra_legend else \
                {"unrestricted": {"median": median_value_unr, "mean": mean_value_unr},
                 "restricted": {"median": median_value_res, "mean": mean_value_res}}
            """
            if m == "weighted_betweenness":
                print(all(v == 0 for v in values_unr))
                print([v for v in values_unr if v != 0])

                print(all(v == 0 for v in values_rest))
                print([v for v in values_rest if v != 0])

                for k, v in r.items():
                    if not np.isfinite(v[m + METRIC_UNR_SUF]):
                        print(k, v)
                    if not np.isfinite(v[m + METRIC_REST_SUF]):
                        print(k, v)
            """
            mse = mean_squared_error(values_unr, values_rest, squared=False)
            # rmspe = rmse(values_unr, values_rest)
            title = "{} (RMSE ".format(m) + ("{:.4f})".format(mse) if len(str("{:.4f}").format(mse)) <= 15
                                             else "{:.2E})".format(mse))
            plot_distribution([xs1, xs2], [ys1, ys2],
                              title=title,
                              xlabel="Value",
                              ylabel="$Pr(X \leq x)$",
                              log_axis=log_axis,
                              save_fig=False if i != len(METRIC_NAMES) - 1 else fig_name,
                              legend=["Unrestricted", "Restricted"],
                              legend_loc=4,
                              font_size=14,
                              y_sup_lim=None,
                              x_sup_lim=None,  # 1e+275 if m == "weighted_betweenness" and log_axis == "x" else False,
                              subplot=(4, 4, i + 1),
                              calculated_values=calculated_values
                              )


def get_data_files(file, m, simulation, is_extra_legend=False, num_nodes=5):
    print('\t\t' + simulation)
    val_unr = [[val['alias'], (round(0.0, 4) if math.isnan(val[m + METRIC_UNR_SUF]) else val[m + METRIC_UNR_SUF])]
                for key, val in file.items()]
    values_unr = [v[1] for v in val_unr]
    if m in METRIC_NAMES_DEGREE:
        for index, val in enumerate(sorted(val_unr, key=itemgetter(1), reverse=True)[:num_nodes]):
            print('\t\t\t' + str(index + 1) + '\t' + str(val))

    val_rest = [[val['alias'], (round(0.0, 4) if math.isnan(val[m + METRIC_REST_SUF]) else val[m + METRIC_REST_SUF])]
                for key, val in file.items()]
    values_rest = [v[1] for v in val_rest]
    if m in METRIC_NAMES_BET_CLO:
        # print(*[x for x in sorted(val_rest, key=itemgetter(1), reverse=True)[:4]], sep="\n\t\t\t")
        # print('\n\t\t'.join((str(x) for x in sorted(val_rest, key=itemgetter(1), reverse=True)[:4])))

        for index, val in enumerate(sorted(val_rest, key=itemgetter(1), reverse=True)[:num_nodes]):
            print('\t\t\t' + str(index + 1) + '\t' + str(val))

    [xs1, ys1] = get_cdf(values_unr, normalize=True)
    [xs2, ys2] = get_cdf(values_rest, normalize=True)
    mean_value_unr = np.mean(values_unr)
    median_value_unr = np.median(np.sort(values_unr))

    mean_value_res = np.mean(values_rest)
    median_value_res = np.median(np.sort(values_rest))
    calculated_values = None if not is_extra_legend else \
        {simulation: {"unrestricted": {"median": median_value_unr, "mean": mean_value_unr},
                      "restricted": {"median": median_value_res, "mean": mean_value_res}}}
    mse = mean_squared_error(values_unr, values_rest, squared=False)

    return xs1, ys1, xs2, ys2, calculated_values, mse


def analyse_restriction_simulations(log_axis="x", is_extra_legend=False, num_nodes=5):
    # Get all graph names without extension nor G1/G2 suffix
    metrics_files = sorted([f for f in listdir(results_folder) if f.endswith("_metrics_1st.json")])
    metrics_files.extend(sorted([f for f in listdir(results_folder) if f.endswith("_metrics_2nd.json")]))

    # Process one file at a time (we are not looking at temporal evolution here)
    sims = {}
    for metric_file in metrics_files:
        print("Processing file: {}".format(metric_file))
        name = filename_clean(metric_file)
        sims[name] = [match for match in metrics_files if name in match]

    for sim in sims:
        fig_name = figs_folder + sim + "_path_restrictions" + (
            "" if log_axis is None else "_log_sim")

        files = []
        for file in sims[sim]:
            with open(results_folder + file, 'r') as f:
                files.append(json.load(f))
        if len(files) == 2:
            for i, m in enumerate(METRIC_NAMES):
                print("\t" + m)

                xs11, ys11, xs21, ys21, calculated_values1, mse1 = get_data_files(files[0], m, '1st',
                                                                                  is_extra_legend, num_nodes)
                xs12, ys12, xs22, ys22, calculated_values2, mse2 = get_data_files(files[1], m, "2nd",
                                                                                  is_extra_legend, num_nodes)

                def rmse_label(mse, sim_label):
                    return sim_label + "{:.4f}".format(mse) if len(str("{:.4f}").format(mse)) <= 15 \
                        else "{:.2E}".format(mse)

                title = "{} \nRMSE (".format(m) + rmse_label(mse1, '1$^{st}$: ') \
                        + '\n' + "".rjust(13) + rmse_label(mse2, '2$^{nd}$: ') + ")"

                calculated_values = [calculated_values1, calculated_values2]

                legend = ["1$^{st}$ - 2$^{nd}$: Unrestricted", "1$^{st}$: Restricted", "2$^{nd}$: Restricted"] \
                    if not is_extra_legend else ["1$^{st}$: Unrestricted", "".rjust(7) + "Restricted",
                                                 "2$^{nd}$: Unrestricted", "".rjust(7) + "Restricted"]
                x = [xs11, xs21, xs22] if not is_extra_legend else [xs11, xs21, xs12, xs22]
                y = [ys11, ys21, ys22] if not is_extra_legend else [ys11, ys21, ys12, ys22]
                num_plots = 3 if len(METRIC_NAMES) == 5 else 4
                plot_distribution(x, y,
                                  title=title,
                                  xlabel="Value",
                                  ylabel="$Pr(X \leq x)$",
                                  log_axis=log_axis,
                                  save_fig=False if i != len(METRIC_NAMES) - 1 else fig_name,
                                  legend=legend,
                                  legend_loc=4,
                                  font_size=14,
                                  y_sup_lim=None,
                                  x_sup_lim=None,  # 1e+275 if m == "weighted_betweenness" and log_axis == "x" else False,
                                  subplot=(num_plots, num_plots, i + 1),
                                  calculated_values=calculated_values
                                  )


def analyse_total_max_channels(log_axis=None, is_extra_legend=False):
    # Get all graph names without extension nor G1/G2 suffix
    metrics_files = [f for f in listdir(results_folder) if f.endswith("_metrics.json")
                     or f.endswith("_metrics_1st.json")]

    # Process one file at a time (we are not looking at temporal evolution here)
    for metric_file in metrics_files:
        print("Processing file: {}".format(metric_file))
        fig_name = figs_folder + filename_clean(metric_file) + "_path_restrictions" + \
                   ("" if log_axis is None else "_log_data")

        with open(results_folder + metric_file, 'r') as f:
            r = json.load(f)
        xs = []
        ys = []
        means = []
        medians = []

        for i, m in enumerate(DATA_NAMES):
            print("\t" + m)
            values = [v[m] for v in r.values()]
            [x, y] = get_cdf(values, normalize=True)
            xs.append(x)
            ys.append(y)
            means.append(np.mean(values))
            medians.append(np.median(np.sort(values)))

        calculated_values = None if not is_extra_legend else {"median": medians, "mean": means}

        plot_distribution(xs, ys,
                          title="Total # of (Disabled) Channels & \nMax # of (Disabled) Channels",
                          xlabel="Value",
                          ylabel="$Pr(X \leq x)$",
                          log_axis=log_axis,
                          save_fig=fig_name,
                          legend=["# Channels by Pair Node", "# Disabled Channels by Pair Node",
                                  "Max # Chan btw Node and any Neighbor",
                                  "Max # Disabled Chan btw Node and any Neighbor"],
                          legend_loc=4,
                          font_size=14,
                          y_sup_lim=None,
                          x_sup_lim=None,  # 1e+275 if m == "weighted_betweenness" and log_axis == "x" else False,
                          subplot=None,
                          calculated_values=calculated_values
                          )


def analyse_channels(log_axis=None, is_extra_legend=False):
    # Get all graph names without extension nor G1/G2 suffix
    metrics_files = [f for f in listdir(results_folder) if f.endswith("_metrics.json")
                     or f.endswith("_metrics_1st.json")]

    # Process one file at a time (we are not looking at temporal evolution here)
    for metric_file in metrics_files:
        print("Processing file: {}".format(metric_file))
        fig_name = figs_folder + filename_clean(metric_file) + "_channels_pair_nodes"

        with open(results_folder + metric_file, 'r') as f:
            r = json.load(f)
        xs_total = []
        ys_total = []
        xs_disabled = []
        ys_disabled = []
        means = []
        medians = []

        print("\tchannels_pair_nodes")
        values = [v['channels'] for v in r.values()]
        values_disabled = []
        values_total = []
        for channels in values:
            if channels is not None:
                for c in channels:
                    # _ = values_enabled.append(0) if len(channels[c]['enabled']) == 0 else
                    # values_enabled.append(len(channels[c]['enabled']))
                    _ = None if len(channels[c]['disabled']) == 0 else values_disabled.append(
                        len(channels[c]['disabled']))
                    values_total.append(len(channels[c]['enabled']) + len(channels[c]['disabled']))

        [xd, yd] = get_cdf(values_disabled, normalize=True)
        [xt, yt] = get_cdf(values_total, normalize=True)

        mean_value_dis = np.mean(values_disabled)
        median_value_dis = np.median(np.sort(values_disabled))

        mean_value_tot = np.mean(values_total)
        median_value_tot = np.median(np.sort(values_total))
        calculated_values = None if not is_extra_legend else \
            {"disabled": {"median": median_value_dis, "mean": mean_value_dis},
             "enabled": {"median": median_value_tot, "mean": mean_value_tot}}

        plot_distribution([xt, xd], [yt, yd],
                          title="Total # of Channels by Pair of Nodes",
                          xlabel="Value",
                          ylabel="$Pr(X \leq x)$",
                          log_axis=log_axis,
                          save_fig=fig_name,
                          legend=["\nTotal\nMin:{}\nMax:{}".format(min(values_total), max(values_total)),
                                  "\nDisabled\nMin:{}\nMax:{}".format(min(values_disabled), max(values_disabled))],
                          legend_loc=4,
                          font_size=14,
                          y_sup_lim=None,
                          x_sup_lim=None,
                          subplot=None,
                          calculated_values=calculated_values
                          )


def analyse_capacity_channels(is_scatter=False, is_format=False):
    # Get all graph names without extension nor G1/G2 suffix
    metrics_files = [f for f in listdir(results_folder) if f.endswith("_metrics.json")
                     or f.endswith("_metrics_1st.json")]

    # Process one file at a time (we are not looking at temporal evolution here)
    for metric_file in metrics_files:
        print("Processing file: {}".format(metric_file))
        fig_name = figs_folder + filename_clean(metric_file) + "_channels_capacity"

        with open(results_folder + metric_file, 'r') as f:
            r = json.load(f)
        values = [v['channels'] for v in r.values()]
        cap_chan = {}
        cap_chan_dis = {}
        chan_cap = {}
        chan_cap_dis = {}
        cap = []
        cap_dis = []
        for channels in values:
            if channels is not None:
                for chan in channels:
                    if len(channels[chan]['enabled']) > 0:
                        for c in channels[chan]['enabled']:
                            capacity = int(c['capacity'])
                            cap.append(capacity)
                            chan_cap[capacity] = 1 if capacity not in chan_cap else chan_cap[capacity] + 1
                            cap_chan[capacity] = capacity if capacity not in cap_chan else cap_chan[capacity] + capacity
                    if len(channels[chan]['disabled']) > 0:
                        for c in channels[chan]['disabled']:
                            capacity = int(c['capacity'])
                            cap.append(capacity)
                            cap_dis.append(capacity)
                            chan_cap[capacity] = 1 if capacity not in chan_cap else chan_cap[capacity] + 1
                            cap_chan[capacity] = capacity if capacity not in cap_chan else cap_chan[capacity] + capacity
                            chan_cap_dis[capacity] = 1 if capacity not in chan_cap_dis else chan_cap_dis[capacity] + 1
                            cap_chan_dis[capacity] = capacity if capacity not in cap_chan_dis else \
                                cap_chan_dis[capacity] + capacity

        xe_chan, ye_chan = assign_arrays(chan_cap)
        xd_chan, yd_chan = assign_arrays(chan_cap_dis)
        xe_cap, ye_cap = assign_arrays(cap_chan)
        xd_cap, yd_cap = assign_arrays(cap_chan_dis)

        spaces = "".rjust(5)
        max_total_chan = '{}Max # Channels:{} - \n{}capacity:{}'.\
            format(spaces, max(ye_chan), spaces, list(chan_cap.keys())[list(chan_cap.values()).index(max(ye_chan))])
        max_total_disabled_chan = '{}Max # Disabled Channels:{} - \n{}capacity:{}'.\
            format(spaces, max(yd_chan), spaces,
                   list(chan_cap_dis.keys())[list(chan_cap_dis.values()).index(max(yd_chan))])

        max_cap = list(cap_chan.keys())[list(cap_chan.values()).index(max(ye_cap))]
        max_cap_dis = list(cap_chan_dis.keys())[list(cap_chan_dis.values()).index(max(yd_cap))]

        max_total_cap = '{}Max Cumulative Capacity Channels:{} - \n{}capacity:{} - \n{}# channels:{}'.\
            format(spaces, max(ye_cap), spaces, max_cap, spaces, chan_cap[max_cap] - chan_cap_dis[max_cap])

        max_total_disabled_cap = '{}Max Cumulative Capacity Disabled Channels:{} - \n{}capacity:{} - \n{}# channels:{}'.\
            format(spaces, max(yd_cap), spaces, max_cap_dis, spaces, chan_cap_dis[max_cap_dis])

        if is_scatter:
            plot_chart_scatter([xe_chan, xd_chan], [ye_chan, yd_chan],
                               title="Number (Disabled) Channels by Capacity",
                               xlabel="Capacity",
                               ylabel="# of Channels",
                               font_size=10,
                               save_fig=fig_name,
                               legend=["Total Channels\n"+max_total_chan,
                                       "Total Disabled Channels\n"+max_total_disabled_chan],
                               legend_loc="best")

            plot_chart_scatter([xe_cap, xd_cap], [ye_cap, yd_cap],
                               title="Cumulative Capacity on (Disabled) Channels by Capacity",
                               xlabel="Capacity",
                               ylabel="Cumulative Capacity per Channels",
                               font_size=10,
                               save_fig=fig_name + '_cumulative',
                               legend=["Cum. Cap. Channels\n" + max_total_cap,
                                       "Cum. Cap. Disabled Channels\n" + max_total_disabled_cap],
                               legend_loc="best")
        else:
            n_cap = []
            n_cap_dis = []
            n_chan = []
            n_chan_dis = []
            labels = []

            def count(val_tot, val_dis, low, high):
                sub_tot = list(x for x in val_tot if low <= x < high)
                sub_dis = list(x for x in val_dis if low <= x < high)
                return (len(sub_tot), sum(sub_tot)), (len(sub_dis), sum(sub_dis))
            notation = "3.0e"
            for i in range(3, 9):
                ini = (10 ** i, int(10 ** (i + 1) / 2))
                tot, dis = count(cap, cap_dis, ini[0], ini[1])
                n_chan.append(tot[0])
                n_cap.append(tot[1])
                n_chan_dis.append(dis[0])
                n_cap_dis.append(dis[1])
                # labels.append(re.sub('[()]', '', str(ini)))
                # labels.append(format(ini[0], notation) + " - " + format(ini[1], notation))
                labels.append(sci_notation(ini[0], 0, x_label=True) + " $-$ " + sci_notation(ini[1], 0, x_label=True))
                med = (int(10 ** (i + 1) / 2), 10 ** (i + 1))
                tot, dis = count(cap, cap_dis, med[0], med[1])
                n_chan.append(tot[0])
                n_cap.append(tot[1])
                n_chan_dis.append(dis[0])
                n_cap_dis.append(dis[1])
                # labels.append(re.sub('[()]', '', str(med)))
                # labels.append(format(med[0], notation) + " - " + format(med[1], notation))
                labels.append(sci_notation(med[0], 0, x_label=True) + " $-$ " + sci_notation(med[1], 0, x_label=True))
            plot_chart_bar([n_chan, n_chan_dis], labels,
                           title="Number (Disabled) Channels by Capacity",
                           xlabel="Capacity",
                           ylabel="# of Channels",
                           font_size=6,
                           save_fig=fig_name + '_his',
                           legend=["Total Channels\n" + max_total_chan,
                                   "Total Disabled Channels\n" + max_total_disabled_chan],
                           legend_loc="best", is_format=is_format)

            plot_chart_bar([n_cap, n_cap_dis], labels,
                           title="Cumulative Capacity on (Disabled) Channels by Capacity",
                           xlabel="Capacity",
                           ylabel="Cumulative Capacity by # of Channels",
                           font_size=6,
                           save_fig=fig_name + '_cumulative_his',
                           legend=["Cum. Cap. Channels\n" + max_total_cap,
                                   "Cum. Cap. Disabled Channels\n" + max_total_disabled_cap],
                           legend_loc="best", is_format=is_format)


def assign_arrays(data, sample=1):
    xd = []
    yd = []
    for key, val in sorted(random.sample(data.items(), int(len(data)*sample)), key=lambda kv: kv[0],
                           reverse=False):
        xd.append(key)
        yd.append(val)
    return xd, yd


def analyse_components():
    # Get all graph names without extension nor G1/G2 suffix
    metrics_files = [f for f in listdir(results_folder) if f.endswith("_metrics.json")
                     or f.endswith("_metrics_1st.json")]

    # Process one file at a time (we are not looking at temporal evolution here)
    for metric_file in metrics_files:
        print("Processing file: {}".format(metric_file))
        fig_name = figs_folder + filename_clean(metric_file) + "_connected_components"

        with open(results_folder + metric_file, 'r') as f:
            r = json.load(f)
        vals = {}
        # ["main_connected_component", "connected_component", "capacity", "fee"]
        for key, values in r.items():
            if values['connected_component'] not in vals:
                vals[values['connected_component']] = {'fee': values['fee'], 'capacity': values['capacity'],
                                                       'counter': 1, 'node': key, 'alias': values['alias']}
            else:
                vals[values['connected_component']]['counter'] += 1

        x = []  # counter
        y = []  # capacity
        z = []  # fee
        for i in sorted(vals.keys()):
            x.append(vals[i]['counter'])
            y.append(vals[i]['capacity'])
            z.append(vals[i]['fee'])

        # max_nodes = max(x, key=len)

        plot_scatter(x, y, z,
                     title="Connected Components on LN",
                     xlabel="Number of nodes",
                     ylabel="Capacity",
                     zlabel="Fee",
                     save_fig=fig_name
                     )


def get_top_k(arr, k, absolute=True):
    """
        Computes the percentage of the total centrality that is explained by the top k most central nodes.
            If `absolute`, k is the number of nodes; otherwise, k is the percentage of nodes

            get_top_k([10, 1, 0, 2, 1, 1, 0, 0, 0, 0], 1, absolute=True)
                66.6
                > The top-1 node (10) has 66.6% of the total centrality (10/15)

            get_top_k([10, 1, 0, 2, 1, 1, 0, 0, 0, 0], 50, absolute=False)
                100.00
                > The 50% of top k nodes concentrate the 100% of the total centrality
    """

    total = sum(arr)
    k_abs = k if absolute else int(k*len(arr)/100)
    k_val = sum(sorted(arr, reverse=True)[:k_abs])

    if total == 0:
        # TODO: we should not be here!!!
        print("Warning: sum of centralities is 0")
        return 0

    return k_val/total*100


def analyse_topk_centrality(normalized=True):
    if normalized:
        topk_metric_names = [m + METRIC_UNR_SUF + (METRIC_NORM_SUF if m in NORMALIZED_VERSIONS else "") for m in
                             METRIC_NAMES]
    else:
        topk_metric_names = [m + METRIC_UNR_SUF for m in METRIC_NAMES]

    # Get all graph names without extension nor G1/G2 suffix
    metrics_files = sorted([f for f in listdir(results_folder) if f.endswith("_metrics.json")
                            or f.endswith("_metrics_1st.json")  or f.endswith("_metrics_2nd.json")])
    absolute_k = [1, 10, 20]
    perc_k = [1, 10, 20]

    # Process one file at a time (we are not looking at temporal evolution here)
    metrics_topk_values = {metric_name: defaultdict(lambda: []) for metric_name in topk_metric_names}
    for metric_file in metrics_files:
        print("Processing file: {}".format(metric_file))
        with open(results_folder + metric_file, 'r') as f:
            r = json.load(f)

        for i, metric_name in enumerate(topk_metric_names):
            print("\t"+metric_name)
            values_unr = [v[metric_name] for v in r.values()]
            #values_rest = [v[m + METRIC_REST_SUF] for v in r.values()]

            for k_val in absolute_k:
                y = get_top_k(values_unr, k_val, True)
                metrics_topk_values[metric_name][k_val].append(y)

            for k_val in perc_k:
                y = get_top_k(values_unr, k_val, False)
                metrics_topk_values[metric_name]["{}%".format(k_val)].append(y)

    for i, metric_name in enumerate(topk_metric_names):
        legend = absolute_k + ["{}%".format(k_val) for k_val in perc_k]
        ys = [metrics_topk_values[metric_name][k_val] for k_val in legend]
        x = [filename_to_date(metrics_file) for metrics_file in metrics_files]
        xs = [x for _ in legend]
        plot_distribution(xs, ys,
                          title="{} top-k".format(metric_name),
                          xlabel="Snapshot",
                          ylabel="Percentage of total centrality explained by top-k",
                          log_axis=None,
                          save_fig=False if i != len(topk_metric_names) - 1 else figs_folder + "topk",
                          legend=legend,
                          legend_loc=4,
                          font_size=14,
                          y_sup_lim=None,
                          subplot=(5, 3, i+1),
                          label_rotation=30
                          )


def analyse_centrality_differences():
    # Get all graph names without extension nor G1/G2 suffix
    metrics_files = [f for f in listdir(results_folder) if f.endswith("_metrics.json")
                     or f.endswith("_metrics_1st.json") or f.endswith("_metrics_2nd.json")]

    # Process one file at a time (we are not looking at temporal evolution here)

    for metric_file in metrics_files:
        print("Processing file: {}".format(metric_file))
        with open(results_folder + metric_file, 'r') as f:
            r = json.load(f)

        sorted_nodes_by_metric = {}
        for i, metric_name in enumerate(METRIC_NAMES):
            print("\t"+metric_name)
            metric_values_per_node = {node: metric_value[metric_name + METRIC_UNR_SUF] for node, metric_value in
                                      r.items()}
            sorted_nodes_by_metric[metric_name] = sorted(metric_values_per_node, key=metric_values_per_node.get,
                                                         reverse=True)

        corr_mat = np.zeros((len(METRIC_NAMES), len(METRIC_NAMES)))
        pval_mat = np.zeros((len(METRIC_NAMES), len(METRIC_NAMES)))
        leg = sorted_nodes_by_metric.keys()

        for i, k1 in enumerate(leg):
            for j, k2 in enumerate(leg):
                corr_mat[i, j], pval_mat[i, j] = spearmanr(sorted_nodes_by_metric[k1], sorted_nodes_by_metric[k2])

        plot_heatmap(corr_mat, leg, leg,
                     title="Spearman rank correlation",
                     xlabel="Measure",
                     ylabel="Measure",
                     #save_fig="show",
                     save_fig=figs_folder + filename_clean(metric_file) + "_centralities_correlation",
                     label_rotation=90)


def number_of_snapshots():
    first = datetime.datetime(2018, 10, 12)
    current = datetime.datetime.now()
    days = (current - first).days
    print("One per day: {} snapshots".format(days))
    print("\t{} hours to compute".format(days))
    print("\t{} days to compute".format(days/24))

    print("One per week: {} snapshots".format(days/7))
    print("\t{} hours to compute".format(days/7))
    print("\t{} days to compute".format(days/24/7))


def graph_metrics(label: bool):
    create_if_not_exists(figs_folder)

    generate_results_files()
    generate_results_files_simulations()
    print('\tCUMULATIVE DISTRIBUTION FUNCTION NORMAL')
    analyse_restriction(is_extra_legend=label)
    print('\tCUMULATIVE DISTRIBUTION FUNCTION LOG')
    analyse_restriction(log_axis="x", is_extra_legend=label)
    analyse_restriction_simulations(log_axis="x", is_extra_legend=label, num_nodes=10)
    print('\tTOTAL & MAX (DISABLED) CHANNELS')
    analyse_total_max_channels(log_axis="x", is_extra_legend=label)
    analyse_channels(log_axis="x", is_extra_legend=label)
    analyse_capacity_channels(is_format=label)
    print('\tCONNECTED COMPONENTS')
    analyse_components()
    print('\tTOPK CENTRALITY')
    analyse_topk_centrality()
    print('\tSPEARMAN RANK CORRELATION')
    analyse_centrality_differences()

    number_of_snapshots()


if __name__ == "__main__":
    graph_metrics(False)
