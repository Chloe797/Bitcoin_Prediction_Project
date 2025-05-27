#can now use these formulae in any notebook, reccomend to mount drive and load this file in directly
#from google.colab import drive
#mount
#drive.mount('/content/drive')

#copy this file to the Colab Directory - alternatively: just download and upload directly
#!cp '/content/drive/MyDrive/FILE_PATH/evaluation_metrics.py' . #replace FILE_PATH with actual file path.

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import statsmodels.api as sm

#MAE
def mae_calc(y_true, y_pred):
  """
  Calculates the Mean Absolute Error (MAE).
  """
  mae = mean_absolute_error(y_true, y_pred)
  return mae


#RMSE
def rmse_calc(y_true, y_pred):
  """
  Calculates the Root Mean Squared Error (RMSE).
  """
  rmse = np.sqrt(mean_squared_error(y_true, y_pred))
  return rmse


#sMAPE
def smape_calc(y_true, y_pred):
  """
  Calculates the Symmetric Mean Absolute Percentage Error (sMAPE).
  Handles cases where the denominator is zero by replacing it with zero
  """
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  #what to do if denom = 0 - very unlikely
  denom = (np.abs(y_true) + np.abs(y_pred)) / 2
  numer = np.abs(y_true - y_pred)


  #handling unlikely rare cases where true and predicted are 0
  with np.errstate(divide='ignore', invalid='ignore'):
    smape = np.where(denom == 0, 0.0, numer / denom )

  return np.mean(smape) *100

#creating cross val sepcifically for time series (i.e. to avoid leakage)
#resource used: https://subashpalvel.medium.com/understanding-time-series-cross-validation-1929c543d339
def rolling_cv(X, y, model, model_type='statsmodel', n_splits=5, custom_fit=None, custom_predict=None, return_preds=False):
    """
    Rolling Window Cross Validation that can be used with any model
    """
    
    roll_window = TimeSeriesSplit(n_splits=n_splits) # will likely always keep to five but better to leave it flexible
    #store results
    mae_results, rmse_results, smape_results = [], [], [] #store as lists
    y_true_all, y_pred_all = [], []
    
    
    for train_index, test_index in roll_window.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        
        #ensuring y_train is 1D array
        y_train = np.ravel(y_train)
        
        #fitting the model for various model types
        if model_type == 'statsmodel':
            X_train_const = sm.add_constant(X_train, has_constant = 'add')
            model_fit = model(y_train, X_train_const).fit()
            
            #add constant to test too
            X_test_const = sm.add_constant(X_test, has_constant='add')
            y_pred = model_fit.predict(X_test_const)
          
        elif model_type == 'custom':
            custom_fit(model, X_train, y_train)
            y_pred = custom_predict(model, X_test)

        elif model_type == 'autogluon':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test).to_numpy() #for further analysis

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        
        y_test = np.ravel(y_test) # 1d array
        y_pred = np.ravel(y_pred)
        
        mae_results.append(mae_calc(y_test, y_pred))
        rmse_results.append(rmse_calc(y_test, y_pred))
        smape_results.append(smape_calc(y_test, y_pred))


        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        
    results = {
        'mae': np.mean(mae_results),
        'rmse': np.mean(rmse_results),
        'smape': np.mean(smape_results)
        
    }
    
    if return_preds:
        results['y_true'] = np.array(y_true_all)
        results['y_pred'] = np.array(y_pred_all)
        
    return results
    
