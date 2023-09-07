from os.path import join as joinpath
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, plot_roc_curve
import joblib
from constants import (
    IMG_EDA_SIZE,
    IMG_FILE_EXT,
    IMG_ROC_CURVES_SIZE,
    IMG_CLASSIFICATION_REPORT_SIZE,
    ROC_CURVE_FILEPATH)


def create_eda_figs(df, fig_size=IMG_EDA_SIZE):
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


def _build_classification_report_image(y_train,
                                       y_test,
                                       y_train_preds,
                                       y_test_preds,
                                       model_name,
                                       filepath):
    """
    Builds and saves classification report image.
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions from generic model
            y_test_preds_lr: test predictions from generic model
            model_name: str (name of the model, e.g. 'Logistic Regression')
            filepath: str

    output:
             None
    """
    fig = plt.figure(figsize=IMG_CLASSIFICATION_REPORT_SIZE)
    # Classification report for train data
    plt.text(0.01,
             1.25,
             f'{model_name} Train',
             {'fontsize': 10},
             fontproperties = 'monospace')
    plt.text(0.01,
             0.05,
             str(classification_report(y_train, y_train_preds)),
             {'fontsize': 10},
             fontproperties = 'monospace')
    # Classification report for test data
    plt.text(0.01,
             0.6,
             f'{model_name} Test',
             {'fontsize': 10},
             fontproperties = 'monospace')
    plt.text(0.01,
             0.7,
             str(classification_report(y_test, y_test_preds)),
             {'fontsize': 10},
             fontproperties = 'monospace')
    plt.axis('off')
    # Fit plot within figure
    plt.tight_layout()
    # Save figure to disk
    fig.savefig(filepath)


def generate_roc_curves(models_lst,
                        X_test,
                        y_test,
                        filepath=ROC_CURVE_FILEPATH,
                        figsize=IMG_ROC_CURVES_SIZE):
    """
    Build and save ROC curves for list of models
    """
    fig, ax = plt.subplots(figsize=figsize)

    for model in models_lst:
        plot_roc_curve(model, X_test, y_test, ax=ax, alpha=0.8)
    fig.savefig(filepath)


def save_model(model, filepath):
    """Save model to disk"""
    joblib.dump(model, filepath)
