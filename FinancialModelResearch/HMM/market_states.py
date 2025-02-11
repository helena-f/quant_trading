import numpy as np

def define_market_states(data, bull_threshold=0.1, bear_threshold=-0.1):
    """
    Define market states: Bear Market (0), Sideways/Consolidation (1), Bull Market (2)
    """
    data['MA200'] = data['Close'].rolling(window=200).mean().ffill()
    data['Trend'] = data['Close'].diff().fillna(0)

    # Initialize all states as sideways/consolidation (1)
    states = np.ones(len(data), dtype=int)

    bull_condition = (data['Close'] > data['MA200']) & (data['Trend'] > bull_threshold)
    bear_condition = (data['Close'] < data['MA200']) & (data['Trend'] < bear_threshold)

    states[bull_condition] = 2  # Set bull market periods
    states[bear_condition] = 0  # Set bear market periods

    return states

def create_transition_matrix_from_states(states, n_states=3):
    transmat = np.zeros((n_states, n_states), dtype=float)

    for (current, next_) in zip(states[:-1], states[1:]):
        transmat[current, next_] += 1

    row_sums = transmat.sum(axis=1, keepdims=True)
    transmat = np.divide(transmat, row_sums, where=(row_sums != 0))

    return transmat

def get_transition_matrices(features_df, n_states=3):
    # Create a uniform transition matrix (equal probability for all transitions)
    uniform_transmat = np.full((n_states, n_states), 1.0 / n_states)

    # Create a persistent transition matrix (high probability to stay in the same state)
    persistent_transmat = np.array([
        [0.8, 0.15, 0.05],  
        [0.1, 0.8, 0.1],
        [0.05, 0.15, 0.8],  
    ])

    # Create a transition matrix based on observed state changes
    initial_states = define_market_states(features_df)
    state_based_transmat = create_transition_matrix_from_states(initial_states, n_states=3)

    state_based_transmat /= state_based_transmat.sum(axis=1, keepdims=True)

    return {
        "Default": None,
        "Uniform": uniform_transmat,
        "Persistent": persistent_transmat,
        "State-Based": state_based_transmat
    }



if __name__ == "__main__":
    from stock_features import get_stock_data, get_key_features

    stock_data = get_stock_data("AAPL", "2024-01-01", "2025-01-01", "1d")
    features = get_key_features(stock_data)

    print("\nLatest market states (last 10 days):")
    states = define_market_states(features)
    print(states[-10:])

    print("\nTransition Matrix Calculation:")
    transmat_options = get_transition_matrices(features)
    for key, matrix in transmat_options.items():
        print(f"\n{key} Transition Matrix:\n", matrix)