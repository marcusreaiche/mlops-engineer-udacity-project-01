from os.path import join as joinpath
import matplotlib.pyplot as plt
import seaborn as sns
from constants import IMG_SIZE, IMG_FILE_EXT


def create_eda_figs(df, fig_size=IMG_SIZE):
    """
    Helper function that creates EDA figures
    """
    figs_dict = {}
    # Churn hist
    figs_dict['churn_distribution'] = plt.figure(figsize=fig_size)
    df['Churn'].hist()
    # Customer_Age hist
    figs_dict['customer_age_distribution'] = plt.figure(figsize=fig_size)
    df['Customer_Age'].hist()
    # Marital_Status bar plot
    figs_dict['marital_status_distribution'] = plt.figure(figsize=fig_size)
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    # Total_Trans_Ct hist
    figs_dict['total_transaction_distribution'] = plt.figure(figsize=fig_size)
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    # Correlation heatmap
    figs_dict['heatmap'] = plt.figure(figsize=fig_size)
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    return figs_dict


def save_figs(figs_dict, fig_dir, fig_file_ext=IMG_FILE_EXT):
    """Helper function used to save figures"""
    for filename, fig in figs_dict.items():
        filepath = joinpath(fig_dir, f'{filename}.{fig_file_ext}')
        fig.savefig(filepath)
