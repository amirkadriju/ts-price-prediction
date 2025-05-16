import jax
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from autobnn import estimators


# returns dataset of sliding windows --> all clmns are treated as features exceplt for "Label"
def create_sliding_window_dataset(df, window_size, target, target_threshold, target_lookahead):
    # calculate return based on target selection
    if target == 'high':
        df['Return'] = (df['High'].shift(-target_lookahead) - df['Close']) / df['Close']
    elif target == 'close':
        df['Return'] = df['Close'].pct_change().shift(-target_lookahead)
    else:
        raise ValueError(f'invalid target: {target}. Expected "high" or "close".')
    
    # calculate label
    df['Label'] = (df['Return'] > target_threshold).astype(int)
    
    # all columns are selected as features except for label
    feature_columns = [col for col in df.columns if col != "Label"]
    feature_data = df[feature_columns].values
    labels = df["Label"].values

    # check if there are enough samples in dataframe
    n_samples = len(df) - window_size
    if n_samples <= 0:
        raise ValueError('Not enough data to create one window.'
                         'Increase your dataset or decrease window_size.')

    X_list = []
    y_list = []

    for i in range(n_samples):
        # Take a window of shape (window_size, num_features)
        window = feature_data[i : i + window_size]
        label = labels[i + window_size - 1]  # label at end of window
        X_list.append(window)
        y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y


# walk forward backtest function for autoBNN
def walk_forward_autobnn(df, model, periods, window_size, train_size,
                         step_size, target, target_threshold, target_lookahead):

    # create X, y
    X, y = create_sliding_window_dataset(df, window_size, target, target_threshold, target_lookahead)
    
    # check to see if enough data for analysis
    n_total = len(X)
    if n_total < train_size + 1:
        raise ValueError('Not enough samples for walk-forward analysis.')

    # init empty lists to store results
    accuracies = []
    predictions = []
    prediction_indices = []

    # loop through each window and train / test model
    for i in range(0, n_total - train_size, step_size):
        # define train dataset
        X_train = X[i : i + train_size]
        y_train = y[i : i + train_size]
        
        # check if there is enough data for test
        test_index = i + train_size
        if test_index >= n_total:
            break

        # define test dataset
        X_test = X[test_index : test_index + 1]
        y_test = y[test_index : test_index + 1]

        # reshape data
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_test_reshaped  = X_test.reshape(X_test.shape[0], -1)
        
        # init scaler, fit on train dataset & transform test with same fit
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)

        # init estimator
        estimator = estimators.AutoBnnMapEstimator(
            model,
            likelihood_model="normal_likelihood_logistic_noise",
            seed=jax.random.PRNGKey(123),
            periods=periods
        )

        # fit model
        estimator.fit(X_train_scaled, y_train)

        # Predict test_value
        y_pred = estimator.predict(X_test_scaled)
        logit = y_pred[0, 0] if y_pred.ndim == 2 else y_pred[0]
        prob = 1.0 / (1.0 + np.exp(-logit))
        pred_class = int(prob >= 0.5)

        # calc accuracy
        acc = accuracy_score(y_test, [pred_class])
        accuracies.append(acc)
        
        # Save predictions and corresponding index
        df_index = i + window_size + train_size - 1
        prediction_indices.append(df_index)
        predictions.append(prob)

    return accuracies, prediction_indices, predictions

