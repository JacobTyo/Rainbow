import os
import plotly
from plotly.graph_objs import Scatter, Layout, Figure
from plotly.graph_objs.scatter import Line
import torch
import json
import numpy as np

import argparse

# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

    ys = torch.tensor(ys_population, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(
        1).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour),
                         name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent),
                          name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

    if '_Q' in title:
        y_axis_title = 'Q'
    else:
        y_axis_title = 'Reward'

    plotly.offline.plot({
        'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
        'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': y_axis_title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)


def get_average(data):
    ys = torch.tensor(data, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(
        1).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std
    return ys_mean.numpy()


def get_trace_data(data):
    ys = torch.tensor(data, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(
        1).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std
    return ys_min.numpy(), ys_lower.numpy(), ys_mean.numpy(), ys_upper.numpy(), ys_max.numpy()


def get_trace(file_path, colours, shw_lgd=False, lgd_name="", normalizer=1,):
    data = {}
    with open(file_path + '.json') as f:
        data = json.load(f)

    # get data to average with
    with open(file_path.split('/')[0] + '/' + file_path.split('/')[1] + '/' + file_path.split('/')[2] + '_2/' +
              file_path.split('/')[3].split('_data')[0] + '_2_data.json') as f:
        data2 = json.load(f)

    # now normalize and get lines to plot:
    # data['Rewards'] = data['Rewards'] / normalizer
    # data2['Rewards'] = data2['Rewards'] / normalizer

    data['Rewards'] = np.divide(np.add(np.asarray(data['Rewards']), np.asarray(data2['Rewards'])), 2).tolist()

    t1_min, t1_lower, t1_mean, t1_upper, t1_max = get_trace_data(data['Rewards'])
    # get the actual "trace" for plotly

    trace1_min = Scatter(x=time_scale, y=t1_min, line=Line(color=colours[1], dash='dash'), name='Min')
    trace1_lower = Scatter(x=time_scale, y=t1_lower, line=Line(color=transparent), name='UpperStdDev', showlegend=False)
    trace1_mean = Scatter(x=time_scale, y=t1_mean, fill='tonexty', fillcolor=colours[1], line=Line(color=colours[0]),
                          name=lgd_name, showlegend=shw_lgd, orientation="h")
    trace1_upper = Scatter(x=time_scale, y=t1_upper, fill='tonexty', fillcolor=colours[1], line=Line(color=transparent),
                           name='LowerStdDev', showlegend=False)
    trace1_max = Scatter(x=time_scale, y=t1_max, line=Line(color=colours[1], dash='dash'), name='Max')


    return [trace1_lower, trace1_mean, trace1_upper]  # [trace1_min, trace1_lower, trace1_mean, trace1_upper, trace1_max]


def get_trace_from_baseline_data(data, lgd_name, shw_lgd, colours):
    t1_min, t1_lower, t1_mean, t1_upper, t1_max = get_trace_data(data)
    # get the actual "trace" for plotly

    trace1_min = Scatter(x=time_scale, y=t1_min, line=Line(color=colours[1], dash='dash'), name='Min')
    trace1_lower = Scatter(x=time_scale, y=t1_lower, line=Line(color=transparent), name='UpperStdDev', showlegend=False)
    trace1_mean = Scatter(x=time_scale, y=t1_mean, fill='tonexty', fillcolor=colours[1], line=Line(color=colours[0]),
                          name=lgd_name, showlegend=shw_lgd, orientation="h")
    trace1_upper = Scatter(x=time_scale, y=t1_upper, fill='tonexty', fillcolor=colours[1], line=Line(color=transparent),
                           name='LowerStdDev', showlegend=False)
    trace1_max = Scatter(x=time_scale, y=t1_max, line=Line(color=colours[1], dash='dash'), name='Max')

    return [trace1_lower, trace1_mean,
            trace1_upper]  # [trace1_min, trace1_lower, trace1_mean, trace1_upper, trace1_max]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="R")
    args = parser.parse_args()

    exp = args.experiment

    # get the final average to compare against
    berzerk_10M = {}
    with open('./results/scratch_berzerk_10M/scratch_berzerk_10M_data.json') as json_file:
        berzerk_10M = json.load(json_file)

    krull_10M = {}
    with open('./results/scratch_krull_10M/scratch_krull_10M_data.json') as json_file:
        krull_10M = json.load(json_file)

    riverraid_10M = {}
    with open('./results/scratch_riverraid_10M/scratch_riverraid_10M_data.json') as json_file:
        riverraid_10M = json.load(json_file)

    time_scale = berzerk_10M['Ts']
    normalizer_index = time_scale.index(3000000) + 1  # make sure to get through this index

    # berzerk_normalizer = get_average(berzerk_10M['Rewards'])[normalizer_index]
    # krull_normalizer = get_average(krull_10M['Rewards'])[normalizer_index]
    # riverraid_normalizer = get_average(riverraid_10M['Rewards'])[normalizer_index]

    # print('berzerk_normalizer: ', berzerk_normalizer)
    # print('krull_normalizer: ', krull_normalizer)
    # print('riverraid_normalizer: ', riverraid_normalizer)

    # now start building plots:
    # ===========================================================================================
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'
    colors1 = [mean_colour, std_colour] # blue
    colors2 = ['rgba(127, 191, 63, 1)', 'rgba(226, 249, 204, 0.7)']  # green
    colors0 = ['rgba(222, 47, 47, 1)', 'rgba(242, 150, 150, 0.2)']  # red
    colors3 = ['rgba(170, 90, 251, 1)', 'rgba(183, 112, 255, 0.2)']  # purple
    colorsBlk = ['rgba(0, 0, 0, 1)', 'rgba(153, 153, 153, 0.2)']  # black

    bline_trace_berz = get_trace_from_baseline_data(berzerk_10M['Rewards'][0:normalizer_index],
                                                    lgd_name='Baseline',
                                                    shw_lgd=True,
                                                    colours=colorsBlk)

    bline_trace_krull = get_trace_from_baseline_data(krull_10M['Rewards'][0:normalizer_index],
                                                    lgd_name='Krull Baseline',
                                                    shw_lgd=False,
                                                    colours=colorsBlk)
    bline_trace_riv = get_trace_from_baseline_data(riverraid_10M['Rewards'][0:normalizer_index],
                                                    lgd_name='River Raid Baseline',
                                                    shw_lgd=False,
                                                    colours=colorsBlk)

    bline_trace_berz_nt = get_trace_from_baseline_data(berzerk_10M['Rewards'][0:normalizer_index],
                                                    lgd_name='Berzerk Baseline',
                                                    shw_lgd=False,
                                                    colours=colorsBlk)

    bline_trace_krull_nt = get_trace_from_baseline_data(krull_10M['Rewards'][0:normalizer_index],
                                                     lgd_name='Krull Baseline',
                                                     shw_lgd=False,
                                                     colours=colorsBlk)
    bline_trace_riv_nt = get_trace_from_baseline_data(riverraid_10M['Rewards'][0:normalizer_index],
                                                   lgd_name='River Raid Baseline',
                                                   shw_lgd=False,
                                                   colours=colorsBlk)

    # Plot 1: 1NF
    # 3x3 plot
    # top left = pretrined berzerk:
    trace1_pb = get_trace('./results/pretrained_berzerk_1'+exp+'_3M/pretrained_berzerk_1'+exp+'_3M_data', colors1)
    trace2_kfb = get_trace('./results/krull_from_berzerk_1'+exp+'_3M/krull_from_berzerk_1'+exp+'_3M_data', colors1, shw_lgd=True, lgd_name='Pre-trained on<br>Berzerk')
    trace3_rfb = get_trace('./results/riverraid_from_berzerk_1'+exp+'_3M/riverraid_from_berzerk_1'+exp+'_3M_data', colors1)
    trace4_bfk = get_trace('./results/berzerk_from_krull_1'+exp+'_3M/berzerk_from_krull_1'+exp+'_3M_data', colors2, shw_lgd=True, lgd_name='Pre-trained on<br>Krull')
    trace5_pk = get_trace('./results/pretrained_krull_1'+exp+'_3M/pretrained_krull_1'+exp+'_3M_data', colors2)
    trace6_rfk = get_trace('./results/riverraid_from_krull_1'+exp+'_3M/riverraid_from_krull_1'+exp+'_3M_data', colors2)
    trace7_bfr = get_trace('./results/berzerk_from_riverraid_1'+exp+'_3M/berzerk_from_riverraid_1'+exp+'_3M_data', colors0, shw_lgd=True, lgd_name='Pre-trained on<br>River Raid')
    trace8_kfr = get_trace('./results/krull_from_riverriad_1'+exp+'_3M/krull_from_riverriad_1'+exp+'_3M_data', colors0)
    trace9_pr = get_trace('./results/pretrained_riverraid_1'+exp+'_3M/pretrained_riverraid_1'+exp+'_3M_data', colors0)

    trace_3pb = get_trace('./results/pretrained_berzerk_3' + exp + '_3M/pretrained_berzerk_3' + exp + '_3M_data', colors1)
    trace_3kfb = get_trace('./results/krull_from_berzerk_3' + exp + '_3M/krull_from_berzerk_3' + exp + '_3M_data', colors1)
    trace_3rfb = get_trace('./results/riverraid_from_berzerk_3' + exp + '_3M/riverraid_from_berzerk_3' + exp + '_3M_data', colors1)
    trace_3bfk = get_trace('./results/berzerk_from_krull_3' + exp + '_3M/berzerk_from_krull_3' + exp + '_3M_data', colors2)
    trace_3pk = get_trace('./results/pretrained_krull_3' + exp + '_3M/pretrained_krull_3' + exp + '_3M_data', colors2)
    trace_3rfk = get_trace('./results/riverraid_from_krull_3' + exp + '_3M/riverraid_from_krull_3' + exp + '_3M_data', colors2)
    trace_3bfr = get_trace('./results/berzerk_from_riverraid_3' + exp + '_3M/berzerk_from_riverraid_3' + exp + '_3M_data', colors0)
    trace_3kfr = get_trace('./results/krull_from_riverriad_3' + exp + '_3M/krull_from_riverriad_3' + exp + '_3M_data', colors0)
    trace_3pr = get_trace('./results/pretrained_riverraid_3' + exp + '_3M/pretrained_riverraid_3' + exp + '_3M_data', colors0)


    fig = plotly.tools.make_subplots(rows=2,
                                     cols=3,
                                     subplot_titles=['Berzerk', 'Krull', 'River Raid', 'Berzerk', 'Krull', 'River Raid'],
                                     vertical_spacing=0.17
                                     )

    # plot baselines first to make key pretty
    for i in bline_trace_berz:
        fig.append_trace(i, 1, 1)

    for i in bline_trace_krull:
        fig.append_trace(i, 1, 2)

    for i in bline_trace_riv:
        fig.append_trace(i, 1, 3)

    # top left plot (1, 1) ========================================================================================
    for i in trace1_pb:
        fig.append_trace(i, 1, 1)

    for i in trace4_bfk:
        fig.append_trace(i, 1, 1)

    for i in trace7_bfr:
        fig.append_trace(i, 1, 1)

    # top center plot (1, 2) ========================================================================================
    for i in trace2_kfb:
        fig.append_trace(i, 1, 2)

    for i in trace5_pk:
        fig.append_trace(i, 1, 2)

    for i in trace8_kfr:
        fig.append_trace(i, 1, 2)

    # top right plot (1, 3) ========================================================================================
    for i in trace3_rfb:
        fig.append_trace(i, 1, 3)

    for i in trace6_rfk:
        fig.append_trace(i, 1, 3)

    for i in trace9_pr:
        fig.append_trace(i, 1, 3)

    # bottom left plot (2, 1) ========================================================================================
    for i in bline_trace_berz_nt:
        fig.append_trace(i, 2, 1)

    for i in trace_3pb:
        fig.append_trace(i, 2, 1)

    for i in trace_3bfk:
        fig.append_trace(i, 2, 1)

    for i in trace_3bfr:
        fig.append_trace(i, 2, 1)

    # bottom center plot (2, 2) ========================================================================================
    for i in bline_trace_krull_nt:
        fig.append_trace(i, 2, 2)

    for i in trace_3kfb:
        fig.append_trace(i, 2, 2)

    for i in trace_3pk:
        fig.append_trace(i, 2, 2)

    for i in trace_3kfr:
        fig.append_trace(i, 2, 2)

    # bottom right plot (2, 3) ========================================================================================
    for i in bline_trace_riv_nt:
        fig.append_trace(i, 2, 3)

    for i in trace_3rfb:
        fig.append_trace(i, 2, 3)

    for i in trace_3rfk:
        fig.append_trace(i, 2, 3)

    for i in trace_3pr:
        fig.append_trace(i, 2, 3)

    fig['layout'].update(height=750, width=1000)#, title=exp)
    fig['layout'].update(legend=dict(orientation="h"))
    if exp == 'NF':
        fig['layout'].update(title='Partial Freezing')
    elif exp == 'R':
        fig['layout'].update(title='Fine Tuning')
    else:
        print('!!!!!!!!!!! No plot title, wtf?????')
    fig['layout']['yaxis1'].update(title='4 Layers Transplanted')
    fig['layout']['yaxis4'].update(title='2 Layers Transplanted')
    fig['layout']['xaxis1'].update(title='<b>a)</b>')
    fig['layout']['xaxis2'].update(title='<b>b)</b>')
    fig['layout']['xaxis3'].update(title='<b>c)</b>')
    fig['layout']['xaxis4'].update(title='<b>d)</b>')
    fig['layout']['xaxis5'].update(title='<b>e)</b>')
    fig['layout']['xaxis6'].update(title='<b>f)</b>')

    plotly.offline.plot(fig, filename='results_stacked_'+exp+'.html')

