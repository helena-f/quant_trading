from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

def train_hmm_with_states(features, n_states=3, transmat=None, random_seed=42):

    try:
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=5000, random_state=random_seed, tol=1e-4, init_params="mc")

        if transmat is not None:
            transmat[np.isnan(transmat)] = 1.0 / n_states
            transmat[transmat.sum(axis=1) == 0] = 1.0 / n_states 
            transmat /= transmat.sum(axis=1, keepdims=True) 
            
            model.transmat_ = transmat

        model.fit(features.to_numpy())

        return model
    except Exception as e:
        raise Exception(f"Error training HMM: {e}")

def kfold_evaluate_model(features, transmat, n_splits=5, random_seed=42):

    if features.isna().any().any():
        features = features.fillna(0.0)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_scores = []
    best_model = None
    best_states = None
    
    for train_index, valid_index in kf.split(features):
        train_features = features.iloc[train_index]  
        valid_features = features.iloc[valid_index]
        
        scaler = StandardScaler()
        train_features = pd.DataFrame(scaler.fit_transform(train_features), columns=features.columns, index=train_features.index)
        valid_features = pd.DataFrame(scaler.transform(valid_features), columns=features.columns, index=valid_features.index)
        
        model = train_hmm_with_states(train_features, transmat=transmat, random_seed=random_seed)
        
        valid_score = model.score(valid_features)
        fold_scores.append(valid_score)

        states = model.predict(train_features)

        if best_model is None or valid_score > max(fold_scores):
            best_model = model
            best_states = states
    
    return best_model, best_states, np.mean(fold_scores), np.std(fold_scores)


if __name__ == "__main__":
    from stock_features import get_stock_data, get_key_features
    from market_states import get_transition_matrices

    stock_data = get_stock_data("AAPL", "2024-01-01", "2025-01-01", "1d")
    features = get_key_features(stock_data)

    transmat_options = get_transition_matrices(features)

    print("Training HMM using State-Based Transition Matrix:")
    best_model, best_states, mean_score, std_score = kfold_evaluate_model(features, transmat=transmat_options["State-Based"])

    print(f"HMM Training Completed!")
    print(f"📊 Mean Log-Likelihood Score: {mean_score:.4f}")
    print(f"📊 Score Standard Deviation: {std_score:.4f}")
