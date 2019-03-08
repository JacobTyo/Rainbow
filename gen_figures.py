import os
import plotly
from plotly.graph_objs import Scatter, Layout, Figure
from plotly.graph_objs.scatter import Line
import torch
import json

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


def get_trace(file_path, normalizer, colours):

    data = {}
    with open(file_path) as json_file:
        data = json.load(json_file)

    # now normalize and get lines to plot:
    data['Rewards'] = data['Rewards'] / normalizer
    t1_min, t1_lower, t1_mean, t1_upper, t1_max = get_trace_data(data['Rewards'])
    # get the actual "trace" for plotly

    trace1_min = Scatter(x=time_scale, y=t1_min, line=Line(color=colours[1], dash='dash'), name='Min')
    trace1_lower = Scatter(x=time_scale, y=t1_lower, line=Line(color=transparent), name='UpperStdDev', showlegend=False)
    trace1_mean = Scatter(x=time_scale, y=t1_mean, fill='tonexty', fillcolor=colours[1], line=Line(color=colours[0]),
                          name='Mean', showlegend=False)
    trace1_upper = Scatter(x=time_scale, y=t1_upper, fill='tonexty', fillcolor=colours[1], line=Line(color=transparent),
                           name='LowerStdDev', showlegend=False)
    trace1_max = Scatter(x=time_scale, y=t1_max, line=Line(color=colours[1], dash='dash'), name='Max')

    return [trace1_lower, trace1_mean, trace1_upper]  # [trace1_min, trace1_lower, trace1_mean, trace1_upper, trace1_max]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="3R_3M")
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
    normalizer_index = time_scale.index(3000000)
    berzerk_normalizer = get_average(berzerk_10M['Rewards'])[normalizer_index]
    krull_normalizer = get_average(krull_10M['Rewards'])[normalizer_index]
    riverraid_normalizer = get_average(riverraid_10M['Rewards'])[normalizer_index]

    print('berzerk_normalizer: ', berzerk_normalizer)
    print('krull_normalizer: ', krull_normalizer)
    print('riverraid_normalizer: ', riverraid_normalizer)

    # now start building plots:
    # ===========================================================================================
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'
    colors1 = [mean_colour, std_colour] # blue
    colors2 = ['rgba(127, 191, 63, 1)', 'rgba(226, 249, 204, 0.7)'] # green
    colors0 = ['rgba(222, 47, 47, 1)', 'rgba(242, 150, 150, 0.2)']  #red
    colors3 = ['rgba(170, 90, 251, 1)', 'rgba(183, 112, 255, 0.2)'] # purple
    # Plot 1: 1NF
    # 3x3 plot
    # top left = pretrined berzerk:
    trace1_pb = get_trace('./results/pretrained_berzerk_'+exp+'/pretrained_berzerk_'+exp+'_data.json', berzerk_normalizer, colors0)
    trace2_kfb = get_trace('./results/krull_from_berzerk_'+exp+'/krull_from_berzerk_'+exp+'_data.json', krull_normalizer, colors1)
    trace3_rfb = get_trace('./results/riverraid_from_berzerk_'+exp+'/riverraid_from_berzerk_'+exp+'_data.json', riverraid_normalizer, colors1)
    trace4_bfk = get_trace('./results/berzerk_from_krull_'+exp+'/berzerk_from_krull_'+exp+'_data.json', berzerk_normalizer, colors2)
    trace5_pk = get_trace('./results/pretrained_krull_'+exp+'/pretrained_krull_'+exp+'_data.json', krull_normalizer, colors0)
    trace6_rfk = get_trace('./results/riverraid_from_krull_'+exp+'/riverraid_from_krull_'+exp+'_data.json', riverraid_normalizer, colors2)
    trace7_bfr = get_trace('./results/berzerk_from_riverraid_'+exp+'/berzerk_from_riverraid_'+exp+'_data.json', berzerk_normalizer, colors3)
    trace8_kfr = get_trace('./results/krull_from_riverriad_'+exp+'/krull_from_riverriad_'+exp+'_data.json', krull_normalizer, colors3)
    trace9_pr = get_trace('./results/pretrained_riverraid_'+exp+'/pretrained_riverraid_'+exp+'_data.json', riverraid_normalizer, colors0)

    # plot_me = [trace1_pb, trace2_kfb, trace3_rfb, trace4_bfk, trace5_pk, trace6_rfk, trace7_bfr, trace8_kfr, trace9_pr]
    #
    # layout = Layout(xaxis=dict(), yaxis=dict(),
    #                 xaxis2=dict(), yaxis2=dict(),
    #                 xaxis3=dict(), yaxis3=dict(),
    #                 xaxis4=dict(), yaxis4=dict(),
    #                 xaxis5=dict(), yaxis5=dict(),
    #                 xaxis6=dict(), yaxis6=dict(),
    #                 xaxis7=dict(), yaxis7=dict(),
    #                 xaxis8=dict(), yaxis8=dict(),
    #                 xaxis9=dict(), yaxis9=dict(),
    #                 xaxis10=dict(), yaxis10=dict(),
    #                 xaxis11=dict(), yaxis11=dict(),
    #                 xaxis12=dict(), yaxis12=dict(),
    #                 xaxis13=dict(), yaxis13=dict(),
    #                 xaxis14=dict(), yaxis14=dict(),
    #                 xaxis15=dict(), yaxis15=dict(),
    #                 xaxis16=dict(), yaxis16=dict(),
    #                 xaxis17=dict(), yaxis17=dict(),
    #                 xaxis18=dict(), yaxis18=dict(),
    #                 )

    fig = plotly.tools.make_subplots(rows=3, cols=3)

    for i in trace1_pb:
        fig.append_trace(i, 1, 1)
    #fig.append_trace(trace1_pb, 1, 1)

    for i in trace2_kfb:
        fig.append_trace(i, 1, 2)
    #fig.append_trace(trace2_kfb, 1, 2)

    #fig.append_trace(trace3_rfb, 1, 3)
    for i in trace3_rfb:
        fig.append_trace(i, 1, 3)

    #fig.append_trace(trace4_bfk, 2, 1)
    for i in trace4_bfk:
        fig.append_trace(i, 2, 1)

    #fig.append_trace(trace5_pk, 2, 2)
    for i in trace5_pk:
        fig.append_trace(i, 2, 2)

    #fig.append_trace(trace6_rfk, 2, 3)
    for i in trace6_rfk:
        fig.append_trace(i, 2, 3)

    #fig.append_trace(trace7_bfr, 3, 1)
    for i in trace7_bfr:
        fig.append_trace(i, 3, 1)

    #fig.append_trace(trace8_kfr, 3, 2)
    for i in trace8_kfr:
        fig.append_trace(i, 3, 2)

    #fig.append_trace(trace9_pr, 3, 3)
    for i in trace9_pr:
        fig.append_trace(i, 3, 3)

    fig['layout'].update(height=450, width=900)#, title=exp)
    # fig = Figure(data=plot_me, layout=layout)

    plotly.offline.plot(fig, filename='results_'+exp+'.html')

