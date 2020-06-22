import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error  # MSE
from sklearn.metrics import mean_absolute_error # MAE
from sklearn.metrics import median_absolute_error # MedAE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RANSACRegressor, SGDRegressor, HuberRegressor, TheilSenRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import seaborn as sn
import matplotlib.pyplot as plt
import random
import numpy as np
import plotly.graph_objects as go
import pickle
import json

def compare_models(X,Y):
  # Split data into training and validation set
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05,  shuffle=True, random_state=0) 
  print("Shapes: X_train: ", X_train.shape, "Y_train: ", Y_train.shape, "X_test: ", X_test.shape, "Y_test", Y_test.shape)
  print("Metric : negative mean square error (MSE)")

  # Scaling
  sc = StandardScaler()
  sc.fit(X_train)
  X_train = sc.transform(X_train)
  X_test = sc.transform(X_test)
  
  # PCA
  '''
  pc = PCA(n_components=0.95)
  pc.fit(X_train)
  X_train = pc.transform(X_train)
  X_test = pc.transform(X_test)
  print (pc.explained_variance_)
  print (pc.explained_variance_ratio_)
  '''
  # Polinomial degree
  '''
  poly = PolynomialFeatures(degree=2)
  poly.fit(X_train)
  X_train = poly.transform(X_train)
  X_test = poly.transform(X_test)
  '''

  # user variables to tune
  seed    = 5
  folds   = 10
  metric  = "neg_mean_squared_error"

  # hold different regression models in a single dictionary
  models = {}
  models["Linear"]        = LinearRegression()
  #models["RANSAC"]        = RANSACRegressor()
  models["Huber"]         = HuberRegressor(max_iter=1000)
  models["TheilSen"]      = TheilSenRegressor()
  #models["SGD"]           = SGDRegressor(max_iter=500,penalty=None, eta0=0.01, tol=0.00001)
  models["Ridge"]         = Ridge()
  models["Lasso"]         = Lasso()
  models["ElasticNet"]    = ElasticNet()
  models["KNN"]           = KNeighborsRegressor()
  models["DecisionTree"]  = DecisionTreeRegressor()
  models["SVR"]           = SVR()
  models["AdaBoost"]      = AdaBoostRegressor()
  models["GradientBoost"] = GradientBoostingRegressor()
  models["RandomForest"]  = RandomForestRegressor()
  models["ExtraTrees"]    = ExtraTreesRegressor()

  # 10-fold cross validation for each model
  model_results = []
  model_names   = []
  for model_name in models:
    model   = models[model_name]
    k_fold  = KFold(n_splits=folds, random_state=seed,shuffle=True)
    results = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring=metric)

    model_results.append(results)
    model_names.append(model_name)
    print("{}: {}, {}".format(model_name, round(results.mean(), 3), round(results.std(), 3)))

  fig = go.Figure()
  for name,res in zip(model_names,model_results):    
      fig.add_trace(go.Box(y=res,name=name, boxpoints='all'))
  #fig.show()
  return fig

def validate(Y_test,Y_pred,name):
  mse = mean_squared_error(Y_test,Y_pred)
  rmse = np.sqrt(mse)
  mae = mean_absolute_error(Y_test,Y_pred)
  medae = median_absolute_error(Y_test,Y_pred)
  print("[" + name + "]" + " MSE: ", round(mse,4), "RMSE: ", round(rmse,4), "MAE: ", round(mae,4), "MedAE: ", round(medae,4))

def train(X,Y,selected="Linear", modelName='best_model.sav'):
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10,  shuffle=True) 

  # Scaling
  sc = StandardScaler()
  sc.fit(X_train)
  X_train = sc.transform(X_train)
  X_test = sc.transform(X_test)

  # create and fit the best regression model
  seed =5
  models = {}
  models["Linear"]        = LinearRegression()
  #models["RANSAC"]        = RANSACRegressor()
  models["Huber"]         = HuberRegressor(max_iter=1000)
  models["TheilSen"]      = TheilSenRegressor()
  #models["SGD"]           = SGDRegressor(max_iter=500,penalty=None, eta0=0.01, tol=0.00001)
  models["Ridge"]         = Ridge()
  models["Lasso"]         = Lasso()
  models["ElasticNet"]    = ElasticNet()
  models["KNN"]           = KNeighborsRegressor()
  models["DecisionTree"]  = DecisionTreeRegressor()
  models["SVR"]           = SVR()
  models["AdaBoost"]      = AdaBoostRegressor()
  models["GradientBoost"] = GradientBoostingRegressor()
  models["RandomForest"]  = RandomForestRegressor()
  models["ExtraTrees"]    = ExtraTreesRegressor()
  
  best_model = models[selected]
  best_model.fit(X_train, Y_train)

  # Save model
  pickle.dump(best_model, open(modelName, 'wb'))

  # make predictions using the model (train and test)
  Y_test_pred = best_model.predict(X_test)
  Y_train_pred = best_model.predict(X_train)
  #print("[INFO] MSE : {}".format(round(mean_squared_error(Y_test, Y_test_pred), 3)))

  # R2 score coefficient of determination (quanto gli input influscono sulla predizione)
  # 0 male 1 bene
  validate(Y_train,Y_train_pred,name="Training")
  R2 = best_model.score(X_train, Y_train)
  print("[Training] R2 Score: ", round(R2,3))

  validate(Y_test,Y_test_pred,name="Test")
  R2 = best_model.score(X_test, Y_test)
  print("[Test] R2 Score: ", round(R2,3))

  fig_train = plot_fig([Y_train,Y_train_pred],["Train Real", "Train Predicted"])
  fig_test = plot_fig([Y_test,Y_test_pred],["Test Real","Test Predicted"])
  return fig_train,fig_test

def apply_model(X,modelName='best_model.sav'):
  loaded_model = pickle.load(open(modelName, 'rb'))
  y_hat = loaded_model.predict(X)
  return y_hat

def plot_fig(Ys, names):
  # Ys list of output to plot [Y_real, Y_pred]
  n = np.linspace(0,len(Ys[0]), len(Ys[0]), dtype=int)
  fig = go.Figure()
  for yh,nm in zip(Ys,names):
    fig.add_trace(go.Scatter(x=n, y=yh,
                      mode='lines',#mode='lines+markers',
                      name=nm))
  fig.update_layout(
      hovermode = "x",
      paper_bgcolor = "rgb(0,0,0)" ,
      plot_bgcolor = "rgb(10,10,10)" , 
      title=dict(
          x = 0.5,
          text = "Risultati",
          font=dict(
              size = 20,
              color = "rgb(255,255,255)"
          )
      )
  )
  return fig

