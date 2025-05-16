import numpy as np
import pandas as pd
from datetime import datetime
from autobnn import operators, kernels
from sklearn.metrics import confusion_matrix

from walk_forward import walk_forward_autobnn


if __name__ == '__main__':
    # select dataset
    symbol = 'BTCUSDT'
    from_date = '2017-08-17'
    interval = '1d'

    # load data
    df = pd.read_feather(f'./data/{symbol.lower()}_from_{from_date}_interval_{interval}.feather')

    # define model & periods based on periodicBNN
    model = operators.Add(
        bnns=(
            kernels.PeriodicBNN(width=40, period=7.0),           # week
            kernels.PeriodicBNN(width=30, period=14.0),          # 2-week
            kernels.PeriodicBNN(width=20, period=30.0),          # month
            kernels.LinearBNN(width=10),
            kernels.MaternBNN(width=30),
        )
    )

    # define periods & start walk forward analysis
    periods = [7, 14, 30]
    accuracies, prediction_indices, predictions = walk_forward_autobnn(df,
                                                                       model,
                                                                       periods,
                                                                       window_size=30,
                                                                       train_size=50,
                                                                       step_size=5,
                                                                       target='high',
                                                                       target_threshold=0.02,
                                                                       target_lookahead=1)

    # print accuracies
    print("Accuracies:", accuracies)
    if accuracies:
        print("Mean Accuracy:", np.mean(accuracies))   

    # add predictions back to dataframe
    pred_column = pd.Series(data=np.nan, index=df.index)
    for idx, pred in zip(prediction_indices, predictions):
        if idx < len(pred_column):
            pred_column.iloc[idx] = pred

    df['Predicted_Prob'] = pred_column
    df['Predicted_Label'] = df['Predicted_Prob'].apply(lambda x: int(x >= 0.5) if pd.notnull(x) else np.nan)

    # print confusion matrix
    df_eval = df.dropna(subset=(['Label', 'Predicted_Label']))
    cm = confusion_matrix(df_eval["Label"], df_eval["Predicted_Label"])
    print("Confusion Matrix:")
    print(cm)

    # save dataframe
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_excel(f'./notebooks/{timestamp}_walk_forward_run.xlsx')