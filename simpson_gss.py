#!/usr/bin/env python
# coding: utf-8

# # Simpson paradoxes over time

# Copyright 2021 Allen B. Downey
# 
# License: [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
# 
# [Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt2/blob/master/simpson_wages.ipynb)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (9, 5)

def decorate(**options):
    """Decorate the current axes.
    
    Call decorate with keyword arguments like
    decorate(title='Title',
             xlabel='x',
             ylabel='y')
             
    The keyword arguments can be any of the axis properties
    https://matplotlib.org/api/axes_api.html
    """
    ax = plt.gca()
    ax.set(**options)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)

    plt.tight_layout()



def stretch_x(factor = 0.03):
    low, high = plt.xlim()
    space = (high-low) * factor
    plt.xlim(low - space, high + space)



def anchor_legend(x, y):
    """Place the upper left corner of the legend box.
    
    x: x coordinate
    y: y coordinate
    """
    plt.legend(bbox_to_anchor=(x, y), loc='upper left', ncol=1)   
    plt.tight_layout()



from statsmodels.nonparametric.smoothers_lowess import lowess

def make_lowess(series):
    """Use LOWESS to compute a smooth line.
    
    series: pd.Series
    
    returns: pd.Series
    """
    y = series.values
    x = series.index.values

    smooth = lowess(y, x, frac=0.8)
    index, data = np.transpose(smooth)

    return pd.Series(data, index=index) 



def plot_series_lowess(series, color, indexed=False, plot_series=True):
    """Plots a series of data points and a smooth line.
    
    series: pd.Series
    color: string or tuple
    """
    if plot_series:
        x = series.index
        y = series.values
        plt.plot(x, y, 'o', color=color, alpha=0.3, label='_')
        # series.plot(linewidth=0, marker='o', color=color, alpha=0.3, label='_')
        
    smooth = make_lowess(series)
    if indexed:
        smooth /= smooth.iloc[0] / 100

    style = '--' if series.name=='all' else '-'
    smooth.plot(style=style, label=series.name, color=color)



from itertools import cycle

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def plot_columns_lowess(table, columns, color_map=None, **options):
    """Plot the columns in a DataFrame.
    
    table: DataFrame with a cross tabulation
    columns: list of column names, in the desired order
    colors: mapping from column names to colors
    """
    color_it = cycle(colors)
    
    for col in columns:
        series = table[col]
        color = color_map[col] if color_map else next(color_it)
        plot_series_lowess(series, color, **options)



def get_xresult(results):
    """
    """
    param = results.params['x']
    pvalue = results.pvalues['x']
    conf_int = results.conf_int().loc['x'].values
    stderr = results.bse['x']
    return [param, pvalue, stderr, conf_int]


def valid_group(group, yvarname):
    """
    """

    # make sure we have at least 100 values
    num_valid = group[yvarname].notnull().sum()
    if num_valid < 100:
        return False
    
    # make sure all the answers aren't the same
    counts = group[yvarname].value_counts()
    most_common = counts.max()
    
    nonplurality = num_valid - most_common
    if nonplurality < 20:
        return False
        
    return True



import statsmodels.formula.api as smf

def run_subgroups(gss, xvarname, yvarname, gvarname, yvalue=None):
    if xvarname == yvarname:
        return False, False, False, 0
    
    is_continuous = (yvarname == 'log_realinc')
    
    # prepare the y variable
    if is_continuous:
        # continuous
        gss['y'] = gss[yvarname]
        ylabel = yvarname
    else:
        # if discrete, code so `yvalue` is 1;
        # all other answers are 0
        yvar = gss[yvarname]
        counts = yvar.value_counts()
        
        # if yvalue is not provided, use the most common value
        if yvalue is None:
            yvalue = counts.idxmax()

        d = counts.copy()
        d[:] = 0
        d[yvalue] = 1
        gss['y'] = yvar.replace(d)
        ylabel = yvarname + '=' + str(yvalue)
        
    gss['x'] = gss[xvarname]
    #xvalues = gss['x'].unique()

    formula = 'y ~ x'
    
    if is_continuous:
        results = smf.ols(formula, data=gss).fit(disp=False)
    else:
        results = smf.logit(formula, data=gss).fit(disp=False)
    #pred_df = pd.DataFrame(results.predict(xvalues), 
    #                       columns=['all'], index=xvalues)
    #print(pred_df)

    param = results.params['x']
    pvalue = results.pvalues['x']
    conf_int = results.conf_int().loc['x'].values
    stderr = results.bse['x']

    columns = ['param', 'pvalue', 'stderr', 'conf_inf']
    result_df = pd.DataFrame(columns=columns, dtype=object)
    result_df.loc['all'] = get_xresult(results)

    grouped = gss.groupby(gvarname)
    for name, group in grouped:
        if not valid_group(group, yvarname):
            continue
    
        if is_continuous:
            results = smf.ols(formula, data=group).fit(disp=False)
        else:
            results = smf.logit(formula, data=group).fit(disp=False)
        result_df.loc[name] = get_xresult(results)

    result_df.ylabel = ylabel
    return result_df



xvarname_binned = {'log_realinc': 'log_realinc10',
                      'year': 'year5',
                      'age': 'age5',
                      'cohort': 'cohort10',
                      }

def summarize(gss, xvarname, yvarname, gvarname, yvalue=None):

    result_df = run_subgroups(gss, xvarname, yvarname, gvarname, yvalue)

    xbinned = xvarname_binned[xvarname]

    series_all = gss.groupby(xbinned)['y'].mean() * 100
    series_all.name = 'all'

    table = gss.pivot_table(index=xbinned, columns=gvarname, values='y', aggfunc='mean') * 100
    table.name = yvarname
    table.ylabel = result_df.ylabel
    table.index.name = xbinned
    table.columns.name = gvarname
    
    return series_all, table



def visualize(series_all, table):
    """
    """
    plot_series_lowess(series_all, 'gray', indexed=False, plot_series=False)
    plot_columns_lowess(table, table.columns)

    yvarname = table.name
    ylabel = table.ylabel
    xvarname = table.index.name
    gvarname = table.columns.name
    
    title = '%s vs %s grouped by %s' % (yvarname, xvarname, gvarname)
    decorate(xlabel=xvarname, 
             ylabel=ylabel,
             title=title)
    anchor_legend(1.02, 1.02)


