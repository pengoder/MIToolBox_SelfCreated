import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use('ggplot')

def read_data(load_path,v_sep):
    df = pd.read_csv(load_path, sep=v_sep)
    return df

def get_model_variable(df, var_nm, rm_ind=False):
    if rm_ind == False:
        var = df[var_nm]
    else:
        var = df.drop(df[var_nm], axis=1)
    return var

def y_x_plot_scatter(y, X):
    """get each of X vairables plot with y
    """
    plt.figure()
    sns.distplot(y, kde=True)
    var_n = len(X.columns)
    fig, axes = plt.subplots(var_n, 1, sharex=True, figsize=(5, 4*var_n))
    for i in range(var_n):
        axes[i].scatter(y, X.iloc[:, i])
        axes[i].legend()
    plt.show()
    
def descriptive_analysis(df):
#    sns.distplot(df.PCP_MEM_CNT, kde=True)
#    df.PCP_MEM_CNT.value_counts()
#    plt.hist(df.TTL_COST, bins=100)
#    plt.show()
#    df.boxplot(column='TTL_LIABILITY', by='HEDIS_POPULATION')
    pd.tools.plotting.scatter_matrix(df, figsize=(18,18), alpha=0.2, diagonal='kde')
    
def format_ytick_label(ax):
    """format y tick labels from number to percentage
    """
    y_axis = ax.get_yticks()
    ax.set_yticklabels('{:,.%}'.format(x) for x in y_axis)
	
