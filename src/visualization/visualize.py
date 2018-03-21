import os
import logging
from dotenv import find_dotenv, load_dotenv
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, os.path.join(Path(os.getcwd()).parents[1], 'src', 'data'))
# from build_data import encode_sparsematrix
import build_data as bd


def plot_numvariables(data, xaxis, measure, ctype='float32', 
                      cgroup='MBA_DELINQUENCY_STATUS_next', loc=6):
    columns = data.select_dtypes(include=[ctype]).copy().columns.values
    dict_col = dict(zip(columns, [measure]*len(columns)))    
    
    g_data = data.groupby([xaxis, cgroup], as_index=False).agg(dict_col)
    g_data.set_index(xaxis, inplace=True)
    
    plt.rcParams['agg.path.chunksize'] = 10000 # =100000 Kernel died, restarting    
    fig, ax_array = plt.subplots(math.ceil(len(columns)/2),2, figsize=(15, 21))        
    c=0    
    for i, ax_row in enumerate(ax_array):
        try:
            for j, axes in enumerate(ax_row):                           
                g_data.groupby(cgroup)[columns[c]].plot(ax=axes, legend = True) 
                axes.legend_.remove()
                axes.set_xlabel('')
                axes.set_ylabel(columns[c], rotation=70)           
                c +=1
                if (c >= len(columns)): break
        except Exception as e:
            print('Error in axis size: ', i , ' ', j)
            
    if ax_array.size > len(columns):
        ax_array[-1,-1].set_visible(False) #to hide the whole subplot
        
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.9, hspace=0.25,
                        wspace=0.25)                    
    handles, labels = axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc=loc, fancybox=True, shadow=True) #, )
              # ncol=3, fancybox=True, shadow=True) # loc=loc)
    return fig    
    

def plot_spread_numvariables(data, ctype='float32', cgroup='MBA_DELINQUENCY_STATUS_next'):    
    columns = data.select_dtypes(include=[ctype]).copy().columns.values
    
    plt.rcParams['agg.path.chunksize'] = 10000 # =100000 Kernel died, restarting    
    fig, ax_array = plt.subplots(math.ceil(len(columns)/2),2, figsize=(15, 21))    
    # fig.suptitle(xaxis +' (' +  measure + ')', fontsize=18)
    c=0    
    for i, ax_row in enumerate(ax_array):
        try:
            for j, axes in enumerate(ax_row):                           
                data.boxplot(column=columns[c], by=cgroup, ax=axes)                
                axes.set_xlabel('')
                axes.set_ylabel(columns[c], rotation=70)  
                axes.set_title('')
                c +=1
                if (c >= len(columns)): break
        except Exception as e:
            print('Error in axis size: ', i , ' ', j)
            
    if ax_array.size > len(columns):
        ax_array[-1,-1].set_visible(False) #to hide the whole subplot
        
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.9, hspace=0.25,
                        wspace=0.25)                    
    fig.savefig(figures_dir + 'Spread_Numerical_Variables ('  + cgroup + ').png')
    plt.close()
    plt.clf()
    
    
def plot_boolvariables(data, columns, xaxis='TIME', name_fig ='nan_'):            
    plt.rcParams['agg.path.chunksize'] = 10000 # =100000 Kernel died, restarting    
    fig, ax_array = plt.subplots(math.ceil(len(columns)/2),2, figsize=(15, 20))    
    fig.suptitle('Plot for Continuous Variables ('+ xaxis + ') group by '+ name_fig + ' VARIABLES', fontsize=18)
    c=0    
    for i, ax_row in enumerate(ax_array):
        try:
            for j, axes in enumerate(ax_row):                           
                g_data = data.groupby([xaxis, columns[c][1]], as_index=False)[columns[c][0]].mean()
                g_data.set_index(xaxis, inplace=True)
                g_data.groupby(columns[c][1])[columns[c][0]].plot(ax=axes, legend = True)                                 
                axes.set_xlabel('')
                axes.set_ylabel(columns[c][0], rotation=70)           
                c +=1
                if (c >= len(columns)): break                            
        except Exception as e:
            print('Error in axis size: ', i , ' ', j, 'e: ', str(e))
            
    if ax_array.size > len(columns):
        ax_array[-1,-1].set_visible(False) #to hide the whole subplot
        
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.9, hspace=0.25,
                        wspace=0.25)                    
    fig.savefig(figures_dir + 'Boxplot for Continous Variables ('+ xaxis + ') group by '+ name_fig + ' VARIABLES')
    plt.close()
    plt.clf()

def scatter_variables(data, xaxis):
    columns = data.select_dtypes(include=['float32']).copy().columns.values
    
    cond = [data.MBA_DELINQUENCY_STATUS_next=='C', data.MBA_DELINQUENCY_STATUS_next=='0', data.MBA_DELINQUENCY_STATUS_next=='3',
            data.MBA_DELINQUENCY_STATUS_next=='6', data.MBA_DELINQUENCY_STATUS_next=='9', data.MBA_DELINQUENCY_STATUS_next=='F',
            data.MBA_DELINQUENCY_STATUS_next=='R']
    
    condlabel = ['MBA_DELINQUENCY_STATUS_next==C', 'MBA_DELINQUENCY_STATUS_next==0', 'MBA_DELINQUENCY_STATUS_next==3',
                 'MBA_DELINQUENCY_STATUS_next==6', 'MBA_DELINQUENCY_STATUS_next==9', 'MBA_DELINQUENCY_STATUS_next==F',
                 'MBA_DELINQUENCY_STATUS_next==R']
    
    dict_col = dict(zip(columns, ['mean']*len(columns)))    
    
    plt.rcParams['agg.path.chunksize'] = 10000 # =100000 Kernel died, restarting
    for z in np.arange(len(cond)):        
        filtered_data = (data[cond[z]].groupby([xaxis, 'LOAN_ID'], as_index=False).agg(dict_col))
        fig, ax_array = plt.subplots(math.ceil(len(columns)/2),2, figsize=(15, 22))    
        fig.suptitle(xaxis +' (' + condlabel[z] + ')', fontsize=18)
        c=-1
        for i, ax_row in enumerate(ax_array):
            for j, axes in enumerate(ax_row):                        
                c +=1
                if (c > len(columns)): break
                axes.scatter(filtered_data[xaxis], filtered_data[columns[c]])
                # axes.set_xlabel(xaxis + condlabel[z])
                axes.set_ylabel(columns[c], rotation=70)                    
                
                    
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.9, hspace=0.25,
                            wspace=0.25)        
        #plt.show()        
        fig.savefig(figures_dir + xaxis +' (' + condlabel[z] + ')')
        plt.close()
        plt.clf()
        
        
def plot_crosstable(data, col_index, sparsematrix, hue="MBA_DELINQUENCY_STATUS_next"):
    carat_table = pd.crosstab(index=data[col_index], 
                          columns=data[hue])
    my_plot = carat_table.plot(kind="bar", figsize=(15,8),stacked=True)
    rot = 70 if (len(data[col_index].unique()) >= 10) else 0     
    my_plot.set_xticklabels(carat_table.index, rotation=rot)    
    if sparsematrix:
        my_plot.set_xlabel('')
        my_plot.get_figure().savefig(figures_dir + carat_table.index[0] + " (" + hue + ").png")
    else:
        my_plot.get_figure().savefig(figures_dir + col_index + " (" + hue + ").png")
    

               
def plot_columnsarray(data, xaxis_array, sparsematrix=True, 
                      plot_type='crosstable', hue="MBA_DELINQUENCY_STATUS_next", 
                      measure='mean', loc='upper right'):
    
    for x in xaxis_array:
        if sparsematrix:
            data['SUMMARY'] = bd.encode_sparsematrix(data, x)              
            col = 'SUMMARY'
        else: col = x
        
        if plot_type == 'crosstable':
            plot_crosstable(data, col, sparsematrix, hue)
        elif (plot_type == 'numvar'):
            fig = plot_numvariables(data,col, measure) 
            fig.suptitle(x +' (' +  measure + ')', fontsize=18)
            fig.savefig(figures_dir + x +' (' + measure + ')')    
        elif (plot_type == 'numvarbyflag'):
            fig = plot_numvariables(data,'TIME', measure, cgroup=col, loc=loc)    
            fig.suptitle(x +' by TIME (' + measure + ')', fontsize=18)
            fig.savefig(figures_dir + x +' by TIME (' + measure + ')')
    
    plt.close()
    plt.clf()
        # else if plot_type == 'boxplot'
            

def plot_table(data, cgroup, carray, table_name):
    from pandas.tools.plotting import table
    
    nan_data = data.groupby(cgroup)[carray].sum().T
    
    ax = plt.subplot(frame_on=False) # no visible frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    table(ax, nan_data)  # where df is your data frame
    plt.savefig(figures_dir + table_name)

def count_outliers(data):
    c_data = data.select_dtypes(include=['float32']).copy()
    c_quantile = c_data.quantile([0.25, 0.75]).T
    c_quantile['IQR'] = c_quantile[0.75] - c_quantile[0.25]
    c_quantile['Q1-IQR_*_1.5'] = c_quantile[0.25] - c_quantile['IQR'] * 1.5
    c_quantile['Q3+IQR_*_1.5'] =c_quantile[0.75] + c_quantile['IQR'] * 1.5     
    c_quantile['Q1-IQR_*_3'] = c_quantile[0.25] - c_quantile['IQR'] * 3
    c_quantile['Q3+IQR_*_3'] = c_quantile[0.75] + c_quantile['IQR'] * 3            
    
    ''' # define and fill the columns:
    c_quantile['HIGHOUTLIERS1.5'] = [c_data.T.loc[i][c_data.T.loc[i] > c_quantile['Q3+IQR_*_1.5'].loc[i]].count() for i in c_data.T.index]
    c_quantile['LOWOUTLIERS1.5'] = [c_data.T.loc[i][c_data.T.loc[i] < c_quantile['Q1-IQR_*_1.5'].loc[i]].count() for i in c_data.T.index]
    c_quantile['HIGHOUTLIERS3'] = [c_data.T.loc[i][c_data.T.loc[i] > c_quantile['Q3+IQR_*_3'].loc[i]].count() for i in c_data.T.index]
    c_quantile['LOWOUTLIERS3'] = [c_data.T.loc[i][c_data.T.loc[i] < c_quantile['Q1-IQR_*_3'].loc[i]].count() for i in c_data.T.index]
    '''
    
    #another way:
    c_data = c_data.T
    c_quantile['HIGHOUTLIERS1.5'] = 0.0
    c_quantile['LOWOUTLIERS1.5'] = 0.0
    c_quantile['HIGHOUTLIERS3'] = 0.0
    c_quantile['LOWOUTLIERS3'] = 0.0
    for i in c_data.index:
        c_quantile['HIGHOUTLIERS1.5'].loc[i] = c_data.loc[i][c_data.loc[i] > c_quantile['Q3+IQR_*_1.5'].loc[i]].count() 
        c_quantile['LOWOUTLIERS1.5'].loc[i] = c_data.loc[i][c_data.loc[i] < c_quantile['Q1-IQR_*_1.5'].loc[i]].count()
        c_quantile['HIGHOUTLIERS3'].loc[i] = c_data.loc[i][c_data.loc[i] > c_quantile['Q3+IQR_*_3'].loc[i]].count()
        c_quantile['LOWOUTLIERS3'].loc[i] = c_data.loc[i][c_data.loc[i] < c_quantile['Q1-IQR_*_3'].loc[i]].count()
    
    return c_quantile


def plot_discrete_variables(data):
    time_array =['YEAR', 'MONTH', 'TIME', 'LOANAGE', 'NUMBER_OF_UNITS']
    plot_columnsarray(data, time_array, sparsematrix=False, plot_type='numvar', measure='mean')
    plot_columnsarray(data, time_array, sparsematrix=False, plot_type='numvar', measure='median')
    plot_spread_numvariables(data)


def plot_categorical_variables(data):
    cat_array = ['ORIGINAL_TERM', 'OCCUPANCY_TYPE',  'PRODUCT_TYPE', 'LOAN_PURPOSE',
             'DOCUMENTATION_TYPE', 'CHANNEL', 'LOAN_TYPE',  'STATE_', 
             'BUYDOWN_FLAG',  'NEGATIVE_AMORTIZATION', 'PREPAY_PENALTY', 
             'IO_FLAG', 'CONVERTIBLE_FLAG', 'POOL_INSURANCE']
    plot_columnsarray(data, cat_array, sparsematrix=True, plot_type='crosstable')    


def plot_flag_variables(data):
    flag_array = ['BUYDOWN_FLAG',  'NEGATIVE_AMORTIZATION', 'PREPAY_PENALTY', 
             'IO_FLAG', 'CONVERTIBLE_FLAG', 'POOL_INSURANCE']
    plot_columnsarray(data, flag_array, sparsematrix=True, 
                  plot_type='numvarbyflag', measure='mean')


def plot_last_12_months_variables(data):
    red_array = ['LLMA2_HIST_LAST_12_MONTHS_MIS', 'LLMA2_C_IN_LAST_12_MONTHS',
       'LLMA2_30_IN_LAST_12_MONTHS', 'LLMA2_60_IN_LAST_12_MONTHS',
       'LLMA2_90_IN_LAST_12_MONTHS', 'LLMA2_FC_IN_LAST_12_MONTHS',
       'LLMA2_REO_IN_LAST_12_MONTHS']
    plot_columnsarray(data, red_array, sparsematrix=False, plot_type='crosstable')


def plot_nan_variables(data):
    nan_cont_array = [('CURRENT_INTEREST_RATE', 'CURRENT_INTEREST_RATE_nan'), 
             ('MBA_DAYS_DELINQUENT', 'MBA_DAYS_DELINQUENT_nan'),
             ('SCHEDULED_MONTHLY_PANDI', 'SCHEDULED_MONTHLY_PANDI_nan'),
             ('SCHEDULED_PRINCIPAL', 'SCHEDULED_PRINCIPAL_nan'),
             ('BACKEND_RATIO', 'BACKEND_RATIO_nan'), 
             ('FIRST_RATE_RESET_PERIOD', 'FIRST_RATE_RESET_PERIOD_nan'),
             ('LIFETIME_RATE_CAP', 'LIFETIME_RATE_CAP_nan'), 
             ('LIFETIME_RATE_FLOOR', 'LIFETIME_RATE_FLOOR_nan'), 
             ('MARGIN', 'MARGIN_nan'),
             ('PAY_RESET_FREQUENCY', 'PAY_RESET_FREQUENCY_nan'), 
             ('PERIODIC_RATE_CAP', 'PERIODIC_RATE_CAP_nan'),
             ('PERIODIC_RATE_FLOOR', 'PERIODIC_RATE_FLOOR_nan'), 
             ('RATE_RESET_FREQUENCY', 'RATE_RESET_FREQUENCY_nan'),
             ('SALE_PRICE', 'SALE_PRICE_nan')]
    plot_boolvariables(data, nan_cont_array, xaxis='TIME')
    

def plot_table_nan_variables(data):
    nan_array = ['CURRENT_INTEREST_RATE_nan', 'MBA_DAYS_DELINQUENT_nan',
       'SCHEDULED_MONTHLY_PANDI_nan', 'SCHEDULED_PRINCIPAL_nan',
       'BACKEND_RATIO_nan', 'FIRST_RATE_RESET_PERIOD_nan',
       'LIFETIME_RATE_CAP_nan', 'LIFETIME_RATE_FLOOR_nan', 'MARGIN_nan',
       'PAY_RESET_FREQUENCY_nan', 'PERIODIC_RATE_CAP_nan',
       'PERIODIC_RATE_FLOOR_nan', 'RATE_RESET_FREQUENCY_nan',
       'SALE_PRICE_nan']
    plot_table(data, 'MBA_DELINQUENCY_STATUS_next', nan_array, 'nan_Variables_table.png')


def plot_bool_variables(data):
    bool_array = ['LLMA2_APPVAL_LT_SALEPRICE', 'LLMA2_PRIME', 'LLMA2_SUBPRIME']
    columns = data.select_dtypes(include=['float32']).copy().columns.values
    lzip = [[(c,b) for c in columns] for b in bool_array]
    for l in lzip:
        plot_boolvariables(data, l, xaxis='TIME', name_fig=l[0][1])

def main(project_dir):
    """ Visualizing datasets: Continuous, Categorical and boolean variables.
    """
    logger = logging.getLogger(__name__)
    logger.info('Visualizing datasets: Continuous, Categorical and boolean variables...')
        


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    figures_dir = os.path.join(Path(os.getcwd()).parents[1], 'reports', 'figures', os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir)
