import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import pathlib
import math

plt.style.use('bmh') 

def statistical_summary(dataframe: pd.DataFrame):
    """ 
    This function provides a general statistical summary
    
    args: 
        dataframe (pd.DataFrame): target Dataframe
    
    output: 
        dataframe (pd.DataFrame): Dataframe descriptive statistics
    
    Example usage: 
        statistical_summary(dataframe = dataframe)
    """
    # Splitting numerical from categorical data
    num_attributes = dataframe.select_dtypes( include=['int64', 'float64','int32', 'float32'] )
    
    # Central Tendency - Mean, Median
    ct1 = pd.DataFrame( num_attributes.apply( np.mean ) ).T
    ct2 = pd.DataFrame( num_attributes.apply( np.median ) ).T
    
    # dispersion - std, min, max, range, skew, kurtosis
    d1 = pd.DataFrame( num_attributes.apply( np.std ) ).T 
    d2 = pd.DataFrame( num_attributes.apply( min ) ).T 
    d3 = pd.DataFrame( num_attributes.apply( max ) ).T 
    d4 = pd.DataFrame( num_attributes.apply( lambda x: x.max() - x.min() ) ).T 
    d5 = pd.DataFrame( num_attributes.apply( lambda x: x.skew() ) ).T 
    d6 = pd.DataFrame( num_attributes.apply( lambda x: x.kurtosis() ) ).T 
    
    # coefficient of variation
    cv1 = d1/ct1
    
    # Putting all together
    m = pd.concat( [d2, d3, d4, ct1, ct2, d1, d5, d6,cv1] ).T.reset_index()
    m.columns = ['attributes', 'min', 'max', 'range', 'mean', 'median', 'std_dev', 'skew', 'kurtosis', 'var_coef']
    return m; 

def correlation_matrix(dataframe:pd.DataFrame):
    """
    This function computes and plots a correlation matrix for all numerical columns
    
    Args: 
        dataframe (pd.DataFrame): Dataframe containing data of interest. 
    
    Returns: 
        heatmap based on correlation matrix
    
    Example usage: 
        correlation_matrix(dataframe = dataframe)
        
    """
    # Selection numerical columns
    num_cols = dataframe.select_dtypes( include=['int64', 'float64','int32', 'float32'] ).columns
    
    # Create correlation matrix
    corr_matrix = dataframe[num_cols].corr()

    # Create a mask to hide upper triangle of the plot
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', annot_kws={"size": 10})
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title('Correlation Matrix', fontsize=14)
    
    return plt.show();



def plot_categorical_features(dataframe: pd.DataFrame, 
                              ncols: int = 3, 
                              figsize: tuple = (20, 20), 
                              plot_type: str = 'countplot',
                              individual_plot: bool = False,
                              root_path: pathlib.Path = Path.cwd(),
                              feature_of_interest:str = None, 
                              file_name: str = None,
                              **kwargs):
    """
    Plots bar plots of categorical columns of a dataframe and save specific indivual plots of interest.

    Args:
        dataframe (pd.DataFrame): dataframe of interest
        ncols (int): number of columns in the plot grid (default = 3)
        figsize (tuple): size of the figure (default = (20, 12)) 
        plot_type (str): Type of plot to use (default = 'countplot')
        individual_plot (bool): flag to go from all figures to specific figure of interest (False as default)
        root_path (pathlib.Path): Root path for saving picture 
        feature_of_interest (str): Feature name of interest 
        file_name (str): name of the file to be created (not including extension, set on .png by default)
        **kwargs: Keyword arguments to be passed to the seaborn countplot/barplot function.
    
    Returns:
        plots :D 
    
    Example usage: 
        - all figures
        plot_categorical_features(dataframe=dataframe, 
                                  ncols=3, 
                                  figsize=(20, 20), 
                                  plot_type='countplot')
        - plot of interest 
        plot_categorical_features(dataframe = df, 
                                  root_path = root_path, 
                                  individual_plot = True, 
                                  plot_type = 'barplot',
                                  file_name = 'feature_of_interest',
                                  feature_of_interest = 'Reporting Semester')
    """
    # If not individual plot is requested, every possible figure is generated
    if not individual_plot: 
        # Keeping only categorical values
        cat_attributes = dataframe.select_dtypes(include=['object', 'category'])

        # Filter out columns with more than 10 unique categories
        cat_attributes = cat_attributes.loc[:, cat_attributes.nunique() <= 10]

        # Setting number of rows
        nrows = (cat_attributes.shape[1] - 1) // ncols + 1

        # Plotting data
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        fig.subplots_adjust(hspace=0.6, wspace=0.4)

        # Estimate number of figures
        print(f'[Info] Around {nrows*ncols} figures to be created')

        for i, col in tqdm(enumerate(cat_attributes)):
            ax = axes.flat[i]
            if plot_type == 'countplot':
                sns.countplot(x=col, data=dataframe, ax=ax, **kwargs)
            elif plot_type == 'barplot':
                sns.barplot(x=col, y='count', 
                            data=dataframe.groupby(col).size().reset_index(name='count'), 
                            ax=ax, **kwargs)
            ax.set_title(col, fontsize=12)
            ax.tick_params(axis='x', labelrotation=45, labelsize=10)

        # Removing empty figures
        if cat_attributes.shape[1] < nrows * ncols:
            for j in range(cat_attributes.shape[1], nrows * ncols):
                fig.delaxes(axes.flat[j])

        # Adjusting figure spacing
        plt.tight_layout()
        plt.show()
    
    # In case an individual figure is of interest 
    if individual_plot: 
        
        # It makes sure the path to save the figure indeed exists, if not it creates it. 
        figure_path = root_path / 'figures'
        if figure_path.exists(): 
            # print(f'[Info] {figure_path} already exists')
            pass
        else: 
            figure_path.mkdir(exist_ok=True, parents = True )
            # print(f'[Info] {figure_path} created sucessfully')
        
        # Handles the name of the file
        file_name = file_name +'.png'
        figure_name = figure_path / file_name
        
        # depending of the type of plot to be created
        if plot_type == 'countplot':
                # Creates the plot
                plot = sns.countplot(x = feature_of_interest, 
                                     data = dataframe, 
                                     **kwargs)
                # saves it
                plot = plot.get_figure()
                plot.savefig(figure_name, dpi=300, bbox_inches='tight', pad_inches=0.2)
                
        elif plot_type == 'barplot':
                # Creates the plot
                plot = sns.barplot(x = feature_of_interest, 
                                   y = 'count', 
                                   data = dataframe.groupby(feature_of_interest).size().reset_index(name='count'),
                                   **kwargs)
                # saves it
                plot = plot.get_figure()
                plot.savefig(figure_name, dpi=300, bbox_inches='tight', pad_inches=0.2)
        


def plot_categorical_features_percentage(dataframe: pd.DataFrame,
                                         ncols: int = 3,
                                         figsize: tuple = (20, 20),
                                         plot_type: str = 'countplot',
                                         individual_plot: bool = False,
                                         root_path: pathlib.Path = Path.cwd(),
                                         feature_of_interest:str = None, 
                                         file_name: str = None,
                                         **kwargs):
    """
    Plots bar plots of categorical columns of a dataframe and save specific indivual plots of interest.

    Args:
        dataframe (pd.DataFrame): dataframe of interest
        ncols (int): number of columns in the plot grid (default = 3)
        figsize (tuple): size of the figure (default = (20, 12)) 
        plot_type (str): Type of plot to use (default = 'countplot')
        individual_plot (bool): flag to go from all figures to specific figure of interest (False as default)
        root_path (pathlib.Path): Root path for saving figure (Set as current working directory, by default) 
        feature_of_interest (str): Feature name of interest 
        file_name (str): name of the file to be created (not including extension, set on .png by default)
        **kwargs: Keyword arguments to be passed to the seaborn countplot/barplot function.
    
    Returns:
        None
    
    Example usage: 
        - all figures
        plot_categorical_features_percentage(dataframe=dataframe, 
                                              ncols=3, 
                                              figsize=(20, 20), 
                                              plot_type='countplot')
        - plot of interest 
        plot_categorical_features_percentage(dataframe = df, 
                                              root_path = root_path, 
                                              individual_plot = True, 
                                              plot_type = 'barplot',
                                              file_name = 'feature_of_interest',
                                              feature_of_interest = 'Reporting Semester')
    """
    # If not individual plot is requested, every possible figure is generated
    if not individual_plot:
        # Keeping only categorical values
        cat_attributes = dataframe.select_dtypes(include=['object', 'category'])

        # Filter out columns with more than 10 unique categories
        cat_attributes = cat_attributes.loc[:, cat_attributes.nunique() <= 10]

        # Setting number of rows
        nrows = (cat_attributes.shape[1] - 1) // ncols + 1

        # Plotting data
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        fig.subplots_adjust(hspace=0.6, wspace=0.4)

        # Estimate number of figures
        print(f'[Info] Around {nrows*ncols} figures to be created')

        for i, col in tqdm(enumerate(cat_attributes)):
            ax = axes.flat[i]
            if plot_type == 'countplot':
                sns.countplot(x=col, data=dataframe, ax=ax, **kwargs)
                ax.set_ylabel('Percentage')
                total_count = len(dataframe)
                for p in ax.patches:
                    height = p.get_height()
                    percentage = (height / total_count) * 100
                    ax.annotate(f'{percentage:.1f}%', (p.get_x() + p.get_width() / 2, height),
                                ha='center', va='bottom')
            elif plot_type == 'barplot':
                sns.barplot(x=col, y='count',
                            data=dataframe.groupby(col).size().reset_index(name='count'),
                            ax=ax, **kwargs)
                ax.set_ylabel('Percentage')
                total_count = len(dataframe)
                for p in ax.patches:
                    height = p.get_height()
                    percentage = (height / total_count) * 100
                    ax.annotate(f'{percentage:.1f}%', (p.get_x() + p.get_width() / 2, height),
                                ha='center', va='bottom')
            ax.set_title(col, fontsize=12)
            ax.tick_params(axis='x', labelrotation=45, labelsize=10)

        # Removing empty figures
        if cat_attributes.shape[1] < nrows * ncols:
            for j in range(cat_attributes.shape[1], nrows * ncols):
                fig.delaxes(axes.flat[j])

        # Adjusting figure spacing
        plt.tight_layout()
        plt.show()
    # In case an individual figure is of interest 
    if individual_plot: 
        
        # It makes sure the path to save the figure indeed exists, if not it creates it. 
        figure_path = root_path / 'figures'
        if figure_path.exists(): 
            # print(f'[Info] {figure_path} already exists')
            pass
        else: 
            figure_path.mkdir(exist_ok=True, parents = True )
            # print(f'[Info] {figure_path} created sucessfully')
        
        # Handles the name of the file
        file_name = file_name +'.png'
        figure_name = figure_path / file_name
        
        # depending of the type of plot to be created
        if plot_type == 'countplot':
                # Creates the plot
                
            plot = sns.countplot(x = feature_of_interest, data = dataframe, **kwargs)
            plot.set_ylabel('Percentage')
            total_count = len(dataframe)
            for p in plot.patches:
                height = p.get_height()
                percentage = (height / total_count) * 100
                plot.annotate(f'{percentage:.1f}%', (p.get_x() + p.get_width() / 2, height),
                                ha='center', va='bottom')

            plot = plot.get_figure()
            plot.savefig(figure_name, dpi=300, bbox_inches='tight', pad_inches=0.2)
                
        elif plot_type == 'barplot':
            
            # Creates the plot           
            plot = sns.barplot(x=feature_of_interest, y='count',
                            data=dataframe.groupby(feature_of_interest).size().reset_index(name='count'),
                            **kwargs)
            plot.set_ylabel('Percentage')
            total_count = len(dataframe)
            for p in plot.patches:
                height = p.get_height()
                percentage = (height / total_count) * 100
                plot.annotate(f'{percentage:.1f}%', (p.get_x() + p.get_width() / 2, height),
                                ha='center', va='bottom')
            plot.set_title(feature_of_interest, fontsize=12)
            plot.tick_params(axis='x', labelrotation=45, labelsize=10)
            
            # saves it
            plot = plot.get_figure()
            plot.savefig(figure_name, dpi=300, bbox_inches='tight', pad_inches=0.2)
            
    return None;

def plot_categorical_heatmap_grid(dataframe: pd.DataFrame, 
                                  individual_plot: bool = False,
                                  root_path: pathlib.Path = Path.cwd(),
                                  feature_of_interest_1:str = None, 
                                  remove_categories_f1: list = [],
                                  feature_of_interest_2:str = None, 
                                  remove_categories_f2: list = [],
                                  file_name: str = None):
    """
    Creates a grid of heatmaps showing the counts of different categories for each pair of categorical variables.

    Args:
        dataframe (pd.DataFrame): Input dataframe containing the categorical variables to plot
        individual_plot (bool): (False by default)
        root_path (pathlib.Path): Root path for saving figure (Path.cwd() by default)
        feature_of_interest_1 (tr) = Feature name of interest (None by default)
        remove_categories_f1 (list) = categories within feature of interest 1 to be dropped ([] by default)
        feature_of_interest_2 (tr) = Feature name of interest (axis x) (None by default)
        remove_categories_f2 (list) = categories within feature of interest 2 to be dropped  ([] by default)
        file_name (str) = output file name (None by default)

    Returns:
        None
        
    Example usage: 
        - all figures
        plot_categorical_heatmap_grid(dataframe = dataframe)
    
        - plot of interest
        plot_categorical_heatmap_grid(dataframe = dataframe, 
                                      individual_plot = True,
                                      feature_of_interest_1 = 'var1', 
                                      feature_of_interest_2 = 'var2', 
                                      remove_categories_f2 = ['cat1', 'cat2'],
                                      file_name = file_name)
    """
    # If not individual plot is requested, every possible figure is generated
    if not individual_plot:
        
        # Keeping only categorical variables with 2-10 categories
        cat_vars = dataframe.select_dtypes(include=['object', 'category'])
        cat_vars = cat_vars.loc[:, cat_vars.nunique().between(2, 10)].columns.tolist()

        # Create all possible combinations of categorical variables
        combinations = list(itertools.combinations(cat_vars, 2))

        # Determine the number of variables and grid shape
        num_vars = len(cat_vars)
        num_cols = min(num_vars, 4)
        num_rows = (len(combinations) - 1) // (num_cols + 1)

        # Create the grid of heatmaps
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 4*num_rows))
        fig.tight_layout(pad=3.0)

        # Estimate number of figures
        print(f'[Info] Around {num_rows*num_cols} figures to be created')

        # Iterate over the combinations and plot the heatmaps
        for ax, (var1, var2) in tqdm(zip(axes.flat, combinations)):
            counts = dataframe.groupby([var1, var2]).size().unstack(fill_value=0)
            sns.heatmap(counts, annot=True, cmap='Blues', fmt='d', ax=ax)
            ax.set_title(f"{var1} vs {var2}")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')


        # Adjust spacing to avoid overlap with the figure
        plt.subplots_adjust(hspace=0.7, wspace=1)

        # Show the plot
        plt.show()
        
    # In case an individual figure is of interest 
    if individual_plot: 
        
        # Remove categories 
        dataframe = dataframe[~dataframe[feature_of_interest_1].isin(remove_categories_f1)]
        dataframe = dataframe[~dataframe[feature_of_interest_2].isin(remove_categories_f2)]
        
        # It makes sure the path to save the figure indeed exists, if not it creates it. 
        figure_path = root_path / 'figures'
        if figure_path.exists(): 
            # print(f'[Info] {figure_path} already exists')
            pass
        else: 
            figure_path.mkdir(exist_ok=True, parents = True )
            # print(f'[Info] {figure_path} created sucessfully')
        
        # Handles the name of the file
        file_name = file_name +'.png'
        figure_name = figure_path / file_name
        
        # Creating individual plot
        counts = dataframe.groupby([feature_of_interest_1, feature_of_interest_2]).size().unstack(fill_value=0)
        plot = sns.heatmap(counts, annot=True, cmap='Blues', fmt='.2f')
        plot.set_title(f"{feature_of_interest_1} vs {feature_of_interest_2}")


        
        # Saving plot
        plot = plot.get_figure()
        plot.savefig(figure_name, dpi=300, bbox_inches='tight', pad_inches=0.2)


def plot_clustered_bars(dataframe: pd.DataFrame, 
                        figsize: tuple = (20, 20)):
    """
    Creates a clustered bar chart for each combination of categorical variables in a pandas dataframe.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe with categorical variables
        figsize (tuple): figure size to use for the plot. (default = (20,20))
    
    Returns:
        None
        
    Example usage 
        plot_clustered_bars(dataframe = dataframe)
    """
    # Get categorical variables
    cat_vars = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not cat_vars:
        print("[Info] No categorical variables in input dataframe")
        return None 
    
    # Creating fig and axes
    num_vars = len(cat_vars)
    fig, axes = plt.subplots(nrows = num_vars, 
                             ncols = num_vars, 
                             figsize = figsize)
    
    # Going through possible interactions
    for i, var1 in enumerate(cat_vars):
        for j, var2 in enumerate(cat_vars):
            # Create a bar plot for the combination of variables
            if var1 == var2:
                sns.countplot(x = var1, 
                              data = dataframe, 
                              ax = axes[i,j])
                axes[i,j].set_xlabel('')
                axes[i,j].set_title(f"{var1}")
            else:
                sns.countplot(x = var1, 
                              hue = var2, 
                              data = dataframe, 
                              ax = axes[i,j])
                axes[i,j].legend_.remove()
                axes[i,j].set_title(f"{var1} vs {var2}")
    
    # Adjust spacing and layout
    plt.subplots_adjust(wspace=0.4, 
                        hspace=0.4)
    return plt.show(); 

def plot_clustered_bar_grid(dataframe: pd.DataFrame, 
                            cat_lenght: int = 10,
                            individual_plot: bool = False,
                            root_path: pathlib.Path = Path.cwd(),
                            feature_of_interest_1:str = None, 
                            feature_of_interest_2:str = None, 
                            file_name: str = None):
    """
    Creates a grid of clustered bar plots for every possible combination of categorical variables with less than 10 categories.
    Maximum 5 columns in the grid.

    Args:
        dataframe (pd.DataFrame): Input dataframe with categorical variables
        cat_lenght (int): Maximum number of categories in features to be included (10 by default)
        individual_plot (bool): flag to go from all figures to specific figure of interest (False by default)
        root_path (pathlib.Path): Root path for saving figure (current working directory by default)
        feature_of_interest_1 (str): Feature name of interest (axis x)
        feature_of_interest_2 (str): Feature name of interest
        file_name (str):  output file name 

    Returns:
        None
    
    Example Usage: 
        - all figures
        plot_clustered_bar_grid(dataframe = dataframe,
                                cat_lenght = 10)
                                
        - plot of interest
        plot_clustered_bar_grid(dataframe = df,
                                individual_plot = True,
                                feature_of_interest_1 = feature_of_interest_1, 
                                feature_of_interest_2 = feature_of_interest_2, 
                                file_name = file_name)
    """
    # If not individual plot is requested, every possible figure is generated
    if not individual_plot:

        # Select categorical variables with less than 10 categories
        cat_vars = dataframe.select_dtypes(include=['object', 'category'])
        cat_vars = cat_vars.loc[:, cat_vars.nunique() < cat_lenght].columns.tolist()

        # Create all possible combinations of categorical variables
        combinations = list(itertools.combinations(cat_vars, 2))

        # Determine the number of variables and grid shape
        num_vars = len(combinations)
        num_cols = min(num_vars, 4)
        num_rows = (num_vars - 1) // num_cols + 1



        # Create the grid of clustered bar plots
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 4*num_rows))
        fig.tight_layout(pad=3.0)

        # Estimate number of figures
        print(f'[Info] Around {num_rows*num_cols} figures to be created')

        # Iterate over the combinations and plot the clustered bar plots
        for ax, (var1, var2) in tqdm(zip(axes.flat, combinations)):
            total_counts = dataframe.groupby([var1, var2]).size().unstack().fillna(0)
            total_counts_percent = total_counts.div(total_counts.sum(axis=1), axis=0) * 100
            total_counts_percent.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f"{var1} vs {var2}")
            ax.legend(loc='upper right')
            ax.set_ylim([0, 100])  # Set the y-axis limit to 0-100 for percentages

            # Adjust rotation of x-axis labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Adjust spacing to avoid overlap with the figure
        plt.subplots_adjust(hspace=0.5, wspace=0.3)

        # Remove unused subplots
        if num_vars < num_rows * num_cols:
            for ax in axes.flat[num_vars:]:
                ax.set_visible(False)

        # Show the plot
        plt.show()
    # In case an individual figure is of interest 
    if individual_plot: 
        
        # It makes sure the path to save the figure indeed exists, if not it creates it. 
        figure_path = root_path / 'figures'
        if figure_path.exists(): 
            # print(f'[Info] {figure_path} already exists')
            pass
        else: 
            figure_path.mkdir(exist_ok=True, parents = True )
            # print(f'[Info] {figure_path} created sucessfully')
        
        # Handles the name of the file
        file_name = file_name +'.png'
        figure_name = figure_path / file_name
        
        # Creating individual plot
        total_counts = dataframe.groupby([feature_of_interest_1, feature_of_interest_2]).size().unstack().fillna(0)
        total_counts_percent = total_counts.div(total_counts.sum(axis=1), axis=0) * 100
        plot = total_counts_percent.plot(kind='bar', stacked=True)
        plot.set_title(f"{feature_of_interest_1} vs {feature_of_interest_2}")
        plot.legend(title = feature_of_interest_2, bbox_to_anchor = (1, 1))
        plot.set(ylabel='% of total')
        plot.set_ylim([0, 100])  # Set the y-axis limit to 0-100 for percentages

        # Adjust rotation of x-axis labels
        plot.set_xticklabels(plot.get_xticklabels(), rotation=45, ha='right')
        
        # Saving plot
        plot = plot.get_figure()
        plot.savefig(figure_name, dpi=300, bbox_inches='tight', pad_inches=0.2)

def plot_boxplot_grid(dataframe: pd.DataFrame,
                      cat_lenght: int = 10, 
                      plot_type: str = 'boxplot',
                      individual_plot: bool = False,
                      root_path: pathlib.Path = Path.cwd(),
                      numerical_feature:str = None, 
                      categorical_feature:str = None,
                      remove_categories: list = [],
                      file_name: str = None):
    """
    Creates a grid of boxplots for numerical variables across categorical features.
    Maximum 5 columns in the grid.

    Args:
        dataframe (pd.DataFrame): Input dataframe with numerical and categorical variables
        cat_lenght (int): Take variables with less than cat_lenght only (10 by default), 
        plot_type (str): Type of plot to be used, boxplot or violin (boxplot by default),
        individual_plot (bool): flag to go from all figures to specific figure of interest (False by default)
        root_path (pathlib.Path): Root path for saving figure (current working directory by default),
        numerical_feature (str): Numerical feature to display on individual plot (None by default), 
        categorical_feature (str) = Categorical feature to display on individual plot (None by default), 
        remove_categories (list) = categories to be removed on individual plot (empty list by default)
        file_name (str): output file name 
        
    Returns:
        None
    
    Example Usage: 
        - all figures
        plot_boxplot_grid(dataframe = dataframe,
                            cat_lenght = 10)
        - plot of interest   
        plot_boxplot_grid(dataframe = dataframe,
                          individual_plot = True,
                          numerical_feature = numerical_feature, 
                          categorical_feature = categorical_feature, 
                          remove_categories = ['cat1','cat2'],
                          file_name = file_name)
        
    """
    # If not individual plot is requested, every possible figure is generated
    if not individual_plot:

        # Select numerical and categorical variables
        num_vars = dataframe.select_dtypes(include=['int', 'float'])
        cat_vars = dataframe.select_dtypes(include=['object', 'category'])
        cat_vars = cat_vars.loc[:, cat_vars.nunique() < cat_lenght].columns.tolist()

        # Create all possible combinations of numerical and categorical variables
        combinations = list(itertools.product(num_vars.columns, cat_vars))

        # Determine the number of variables and grid shape
        num_cols = 4
        num_rows = (len(combinations) - 1) // num_cols + 1

        # Calculate the required figure size
        fig_width = 6 * num_cols
        fig_height = 4 * num_rows

        # Create the grid of boxplots with adjusted layout
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_width, fig_height))
        fig.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the figure layout to leave space for text

        # Estimate number of figures
        print(f'[Info] Around {num_rows*num_cols} figures to be created')

        # Iterate over the combinations and plot the boxplots
        for ax, (num_var, cat_var) in tqdm(zip(axes.flat, combinations)):
            if plot_type == 'violin':
                sns.violinplot(x = cat_var, 
                               y = num_var, 
                               data = dataframe, 
                               ax = ax)

            if plot_type == 'boxplot':
                sns.boxplot(x = cat_var, 
                            y = num_var, 
                            data = dataframe, 
                            ax = ax)

            ax.set_title(f"{num_var} by {cat_var}")

        # Adjust spacing to avoid overlap with the figure
        plt.subplots_adjust(hspace=0.5, wspace=0.3)

        # # Remove unused subplots
        # if num_vars < num_rows * num_cols:
        #     for ax in axes.flat[num_vars:]:
        #         ax.set_visible(False)

        # Show the plot
        plt.show()
    if individual_plot:
        # It makes sure the path to save the figure indeed exists, if not it creates it. 
        figure_path = root_path / 'figures'
        if figure_path.exists(): 
            # print(f'[Info] {figure_path} already exists')
            pass
        else: 
            figure_path.mkdir(exist_ok=True, parents = True )
            # print(f'[Info] {figure_path} created sucessfully')
        
        # Handles the name of the file
        file_name = file_name +'.png'
        figure_name = figure_path / file_name
        
        # Remove categories 
        dataframe = dataframe[~dataframe[categorical_feature].isin(remove_categories)]
        
        if plot_type == 'violin':
            plot = sns.violinplot(x = categorical_feature, 
                                  y = numerical_feature, 
                                  data = dataframe)

        if plot_type == 'boxplot':
            plot = sns.boxplot(x = categorical_feature, 
                               y = numerical_feature, 
                               data = dataframe)

        plot.set_title(f"{numerical_feature} by {categorical_feature}")
        
        # Saving plot
        plot = plot.get_figure()
        plot.savefig(figure_name, dpi=300, bbox_inches='tight', pad_inches=0.2)


def plot_feature_vs_target(dataframe: pd.DataFrame, 
                           target_variable: str):
    """
    Plot the relationship between each feature variable and the target variable.
    For variables with less than 50 unique values, create percentage bar plots.
    For variables with more than 50 unique values, create overlapped histograms.
    
    Parameters:
        dataframe (pd.DataFrame): The input dataframe containing the feature and target variables.
        target_variable (str): The name of the target variable column.
    
    Returns:
        None
    
    
    """
    num_features = len(dataframe.columns) - 1  # Subtract 1 to exclude the target column
    num_rows = math.ceil(num_features / 3)  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

    row = 0
    col = 0

    # Iterate over each feature column
    for i, feature_column in enumerate(dataframe.columns):
        if feature_column == target_variable:
            continue  # Skip the target variable column

        ax = axes[row, col] if num_rows > 1 else axes[col]  # Get the current axis

        if col == 2:
            col = 0
            row += 1
        else:
            col += 1

        unique_values = dataframe[feature_column].nunique()

        # Generate the appropriate plot based on the number of unique values
        if unique_values <= 50:
            sns.barboxplotplot(x = feature_column, 
                        y = target_variable, 
                        data = dataframe, 
                        estimator = lambda x: len(x) / len(dataframe) * 100, 
                        ax=ax)
            ax.set_title(f"{feature_column} vs {target_variable} (Percentage Bar Plot)")
            ax.set_ylabel("Percentage")
            plt.legend(frameon=False, ncol=2)
        else:
            sns.histplot(data = dataframe, 
                         x = feature_column, 
                         hue = target_variable, 
                         multiple = 'stack', 
                         kde = True,
                         ax = ax)
            ax.set_title(f"{feature_column} vs {target_variable} (Overlapped Histogram)")
            plt.legend(frameon=False, ncol=2)

    # Remove empty subplots
    if num_features % 3 != 0:
        for i in range(num_features % 3, 3):
            fig.delaxes(axes[row, i] if num_rows > 1 else axes[i])

    plt.tight_layout()
    plt.show()

def scatterplot_matrix_numeric(dataframe: pd.DataFrame, 
                               more_than_two_values: bool = True,
                               variables_of_interest:list = [], 
                               hue_variable:str = None,
                               remove_categories_hue: list = [],
                               individual_plot: bool = False,
                               root_path: pathlib.Path = Path.cwd(),
                               file_name: str = None):
    """
    Generates a distribution plot for one variable or a scatter plot for two variables in a dataframe,
    with an optional categorical variable as hue or color.

    Args:
        dataframe (pd.DataFrame): Input dataframe
        more_than_two_values (bool): Keep variables with at least two numerical values (True by default),
        individual_plot (bool): flag to go from all figures to specific figure of interest (False by default)
        variables_of_interest (list): Variables to be used for individual plot  ([] by default), 
        hue_variable (str): Variable to use as hue on individual plot(None by default),
        remove_categories_hue (list): Remove categories on hue variable ([] by default),
        root_path (pathlib.Path): Root path to save figure (Path.cwd() by default),
        file_name (str): output file name (None by default)

    Returns:
        None
    
    Example Usage: 
        - all figures
        scatterplot_matrix_numeric(dataframe = dataframe)
        
        - plot of interest
        scatterplot_matrix_numeric(dataframe = dataframe, 
                                   variables_of_interest = ['var1','var2'], 
                                   hue_variable='var3',
                                   remove_categories_hue = ['cat1','cat2'],
                                   individual_plot = True,
                                   file_name = file_name)
    """
    # If not individual plot is requested, every possible figure is generated
    if not individual_plot:
        # Get numerical variables
        num_vars = dataframe.select_dtypes(include = ['int64', 'float64','int32', 'float32']).columns.tolist()

        # Making sure there are numerical variables
        if not num_vars:
            print("No numerical variables in input dataframe")
            return None; 

        # Create scatterplot matrix
        sns.set(style='ticks')
        if more_than_two_values: 
            num_var_2 = [var for var in num_vars if len(list(dataframe[var].unique())) > 2]
            sns.pairplot(dataframe[num_var_2], 
                         diag_kind ='kde')
        else: 
            sns.pairplot(dataframe[num_vars], 
                        diag_kind ='kde')

        # Rotate x-axis and y-axis tick labels
        plt.xticks(rotation = 45)
        plt.yticks(rotation = 45)

        plt.show(); 
    
    if individual_plot:
        # It makes sure the path to save the figure indeed exists, if not it creates it. 
        figure_path = root_path / 'figures'
        if figure_path.exists(): 
            # print(f'[Info] {figure_path} already exists')
            pass
        else: 
            figure_path.mkdir(exist_ok=True, parents = True )
            # print(f'[Info] {figure_path} created sucessfully')
        
        # Handles the name of the file
        file_name = file_name +'.png'
        figure_name = figure_path / file_name
        
        # Remove categories 
        dataframe = dataframe[~dataframe[hue_variable].isin(remove_categories_hue)]

        # if only one variable is passed
        if len(variables_of_interest) == 1:
            variable = variables_of_interest[0]
            
            # Check the variable is indeed numeric
            if variable in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[variable]):
                
                # Creating the plot
                plot = sns.histplot(data=dataframe, x=variable, kde=True, hue=hue_variable)
                plt.xlabel(variable)
                plt.ylabel('Density')
                plt.title(f'Distribution of {variable}')
                
                # Saving plot
                plot = plot.get_figure()
                plot.savefig(figure_name, dpi=300, bbox_inches='tight', pad_inches=0.2)
            
            # If not return message
            else:
                print(f"Variable '{variable}' does not exist or is not numeric.")
        
        # If two variables are passed
        elif len(variables_of_interest) == 2:
            x_variable, y_variable = variables_of_interest
            
            # Check the variable is indeed numeric
            if x_variable in dataframe.columns and y_variable in dataframe.columns \
                    and pd.api.types.is_numeric_dtype(dataframe[x_variable]) \
                    and pd.api.types.is_numeric_dtype(dataframe[y_variable]):
                #  Creating the plot
                plot = sns.scatterplot(data=dataframe, x=x_variable, y=y_variable, hue=hue_variable)
                plt.xlabel(x_variable)
                plt.ylabel(y_variable)
                plt.title(f'Scatter Plot: {x_variable} vs {y_variable}')
                
                # Saving plot
                plot = plot.get_figure()
                plot.savefig(figure_name, dpi=300, bbox_inches='tight', pad_inches=0.2)
            else:
                print("Both variables should exist and be numeric.")
        else:
            print("Please provide either one variable for a distribution plot or two variables for a scatter plot.")

def kruskal_wallis_test(dataframe: pd.DataFrame, 
                        group_variable: str, 
                        numeric_variable: str,
                        groups: list, 
                        significance:float = 0.05):
    """
    Performs the Kruskal-Wallis test to compare means across multiple groups.

    Args:
        dataframe (pd.DataFrame): Input dataframe
        group_variable (str): Categorical variable representing the groups
        numeric_variable (str): Numeric variable for which means are compared
        groups (list): List of groups categories to include in the test
        significance (float): Statistical significance level (set default as 0.05)

    Returns:
        None

    Example Usage:
        kruskal_wallis_test(dataframe = dataframe, 
                            group_variable = group_variable, 
                            numeric_variable = numeric_variable,
                            groups =['gr1', 'gr2', 'gr3'])
    """
    # Group the data by the group variable
    dataframe = dataframe[dataframe[group_variable].isin(groups)]
    group_data = [dataframe[dataframe[group_variable] == group][numeric_variable] for group in groups]

    # Perform the Kruskal-Wallis test
    stat, p_value = stats.kruskal(*group_data)

    # Create a dictionary to store the results
    result = {'H-value': stat, 'p-value': p_value}

    if result['p-value'] < significance:
        print('Kruskal-Wallis test result\n')
        print(f"H-value: {result['H-value']}, p-value: {result['p-value']}\n")
        print(f'[Result] Statistically significant ({int(100*significance)}%) evidence suggesting means of {numeric_variable} across categories of {group_variable} are different.')
    else:
        print('Kruskal-Wallis test result\n')
        print(f"H-value: {result['H-value']}, p-value: {result['p-value']}\n")
        print(f'[Result] Statistically significant ({int(100*significance)}%) evidence suggesting means of {numeric_variable} across categories of {group_variable} are equal.')
