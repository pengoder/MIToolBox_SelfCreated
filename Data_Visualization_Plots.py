# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:13:08 2018

@author: pqian
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style as style
import seaborn as sns
style.use('ggplot')

def gen_scatter_plot(x_data, y_data, x_label='', y_label='', title='', yscale_log=False):
    # Create the plot object
    _, ax = plt.subplots()
    
    # Plot the data, set the size(s), color and transparency (alpha)
    ax.scatter(x_data, y_data, s = 10, alpha = 0.75)
    
    if yscale_log == True:
        ax.set_yscale('log')
        
    # Label the axes and providing a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
def gen_line_plot(x_data, y_data, x_label='', y_label='', title=''):
    # Create the plot object
    _, ax = plt.subplots()
    
    # Plot the best fit line, set the linewidth(lw), color and transparency (alpha) of the line
    ax.plot(x_data, y_data, lw = 2, alpha = 1)
            
    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
def gen_histogram(data, n_bins, cumulative=False, x_label='', y_label='', title=''):
    _, ax = plt.subplots()
    ax.hist(data, n_bins = n_bins, cumulative = cumulative, color = '#539caf')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
def gen_overlaid_histogram(data1, data2, n_bins=0, data1_name='', data2_name='', x_label='', y_label ='', title=''):
    # Set the bounds for the bins so that the two distributions are fairly compared
    max_nbins = 10
    data_range = [min(min(data1), min(data2)), max(max(data1), max(data2))]
    binwidth = (data_range[1] - data_range[2]) / max_nbins
    
    if n_bins == 0:
        bins = np.arrange(data_range[0], data_range[1] + binwidth, binwidth)
    else:
        bins = n_bins
    
    # Create the plot
    _, ax = plt.subplots()
    ax.hist(data1, bins = bins, alpha = 1, label = data1_name)
    ax.hist(data2, bins = bins, alpha = 0.75, label = data2_name)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc = 'best')

def gen_bar_plot(x_data, y_data, error_data, x_label='', y_label='', title=''):
    _, ax = plt.subplots()
    # Draw bars, position them in the center o fth tick mark on the x-axis
    ax.bar(x_data, y_data, align = 'center')
    # Draw error bars to show standard deviation, set ls to 'none' to remove line between points
    ax.errorbar(x_data, y_data, yerr = error_data, ls = 'none', lw = 2, capthick = 2)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
def gen_stackedbar_plot(x_data, y_data_list, y_data_names='', x_label='', y_label='', title=''):
    _, ax = plt.subplots()
    # Draw bars, one category at a time
    for i in range(0, len(y_data_list)):
        if i == 0:
            ax.bar(x_data, y_data_list[i], align = 'center', label = y_data_names[i])
        else:
            # For each category after the first, the bottom of the bar will be the top of the last category
            ax.bar(x_data, y_data_list[i], bottom = y_data_list[i - 1], align = 'center', label = y_data_names[i])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc = 'upper right')
    
def gen_groupedbar_plot(df, x_name, y_name, by_group, x_label, y_label, title):
    #use seaborn
    _, ax = plt.subplots()
    ax = sns.catplot(x=x_name, y=y_name, hue=by_group, data=df, kind='bar')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc = 'upper right')
