import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_model_results(results_df):

    plt.figure(figsize=(15, 8))
    
    feature_sets = results_df['Feature_Set'].drop_duplicates().values
    transitions = results_df['Transition'].unique()
    
    bar_width = 0.2
    positions = np.arange(len(feature_sets))
    
    for i, transition in enumerate(transitions):

        mask = results_df['Transition'] == transition
        scores = results_df[mask]['KFold_Mean_Score']
        std_devs = results_df[mask]['KFold_Std_Dev']
        
        plt.bar(positions + i*bar_width, 
                scores,
                bar_width,
                label=transition,
                alpha=0.7)
        
        plt.errorbar(positions + i*bar_width,
                    scores,
                    yerr=std_devs,
                    fmt='none',
                    color='black',
                    capsize=5)
    
    plt.title('Model Performance Comparison', fontsize=16)
    plt.xlabel('Feature Sets', fontsize=12)
    plt.ylabel('Mean Score', fontsize=12)
    
    feature_labels = [str(fs).strip('[]').replace("'", "") for fs in feature_sets]
    plt.xticks(positions + bar_width*(len(transitions)-1)/2, 
               feature_labels, 
               rotation=45, 
               ha='right')
    

    plt.legend(title='Transition Matrix Type')
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return plt.gcf()


