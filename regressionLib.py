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
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn import datasets


# 0 CARICA DATASET
def load_dataset(name="boston"):
    datasets_list = ["boston", "diabetes"] #,"tips"]
    chosen_dataset = name #"boston"

    boston = datasets.load_boston()
    df_boston = pd.DataFrame(boston.data,columns=boston.feature_names)
    df_boston["price"] = boston.target

    diabetes = datasets.load_diabetes()
    df_diabetes = pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
    df_diabetes["desease"] = diabetes.target
    df_diabetes = df_diabetes.round(decimals=5)

    #df_nuovo_dataset = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vSqHhx2kS9gCNmI04yksqTP2PRsT6ifTU2DLokKs3Y6KgcSGIAL7_4t_q_8kNhVkFA0xD2nt7hn_w-5/pub?output=csv")

    dict_datasets = {
        "boston": df_boston,
        "diabetes": df_diabetes,
    #    "nuovo" : df_nuovo_dataset
    }    
    df = dict_datasets[chosen_dataset]#.head(10)
    df = df.dropna()

    #print(df.head())
    X = df.iloc[:,0:-1].values
    Y = df.iloc[:,-1].values

    return X,Y

# 1 COMPARA I MODELLI
def compare_models(X,Y):
  # Split data into training and validation set
  #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01,  shuffle=True, random_state=0) 
  #print("Shapes: X_train: ", X_train.shape, "Y_train: ", Y_train.shape, "X_test: ", X_test.shape, "Y_test", Y_test.shape)
  #print("Metric : negative mean square error (MSE)")

  # Scaling
  sc = StandardScaler()
  sc.fit(X)
  X_train = sc.transform(X)
  Y_train = Y
  #X_test = sc.transform(X_test)
  
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

# 2 SCELGO IL MODELLO MIGLIORE
def train(X,Y,selected="Linear", modelName='best_model.sav'):
    max_val = -10000000
    for i in range(0,20):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10,  shuffle=True) 

        # Scaling
        #sc = StandardScaler()
        #sc.fit(X_train)
        #X_train = sc.transform(X_train)
        #X_test = sc.transform(X_test)

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

        # Logistic Regression
        pipeline = Pipeline([
            ("sc", StandardScaler()),
            #("pca", PCA(n_components=0.98)),
            ("reg", best_model),
        ])
        pipeline.fit(X_train, Y_train)
        
        #best_model.fit(X_train, Y_train)

        # make predictions using the model (train and test)
        Y_test_pred = pipeline.predict(X_test)
        Y_train_pred = pipeline.predict(X_train)
        print("[INFO] MSE : {}".format(round(mean_squared_error(Y_test, Y_test_pred), 3)))

        # R2 score coefficient of determination (quanto gli input influscono sulla predizione)
        # 0 male 1 bene
        #validate(Y_train,Y_train_pred,name="Training")
        R2_train = pipeline.score(X_train, Y_train)
        print("[Training] R2 Score: ", round(R2_train,3))

        #validate(Y_test,Y_test_pred,name="Test")
        R2_test = pipeline.score(X_test, Y_test)
        print("[Test] R2 Score: ", round(R2_test,3))

        if np.abs(R2_test)>max_val:
            # Save model
            pickle.dump(pipeline, open(modelName, 'wb'))
            max_val = np.abs(R2_test)
            fig_train = plot_fig([Y_train,Y_train_pred],["Train Real", "Train Predicted"])
            fig_test = plot_fig([Y_test,Y_test_pred],["Test Real","Test Predicted"])
            print( "Best: [Training] R2 Score: ", round(R2_train,3))
            print("Best: [Test] R2 Score: ", round(R2_test,3))

    return fig_train,fig_test

# 3. APPLICARE IL MODELL ALLENATO
def apply_model(X,modelName='best_model.sav'):
    loaded_model = pickle.load(open(modelName, 'rb'))
    y_hat = loaded_model.predict(X)
    return y_hat

# UTILIS
def validate(Y_test,Y_pred,name):
    mse = mean_squared_error(Y_test,Y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test,Y_pred)
    medae = median_absolute_error(Y_test,Y_pred)
    print("[" + name + "]" + " MSE: ", round(mse,4), "RMSE  : ", round(rmse,4), "MAE: ", round(mae,4), "MedAE: ", round(medae,4))

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


# PROCEDURE
def train_procedure():
    # 0 CARICA DATASET
    print("CARICA DATASET")
    X,Y = load_dataset(name="boston")

    # 1 COMPARA I MODELLI
    print("COMPARA MODELLI")
    fig_compare_models = compare_models(X,Y)

    # 2 SCELGO IL MODELLO MIGLIORE
    print("SCELGO MODELLO MIGLIORE")
    model_list = ["Linear", "Huber", "TheilSen","Ridge","Lasso","ElasticNet","KNN","DecisionTree","SVR","AdaBoost","GradientBoost","RandomForest","ExtraTrees"]
    chosen_model = "RandomForest"
    model_name = "test.sav"
    fig_train, fig_test = train(X,Y,selected=chosen_model,modelName=model_name)

def test_procedure():
    # 3 CARICO IL NUOVO DATASET
    print("CARICO NUOVO DATASET")
    X,Y = load_dataset(name="boston")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.80,  shuffle=True) 

    # 3bis. APPLICARE IL MODELL ALLENATO
    print("APPLICO MODELLO ALLENATO")
    Y_pred_pipe = apply_model(X_test,modelName="test.sav")
    fig_res = plot_fig([Y_test,Y_pred_pipe], ["test","pipe"])
 