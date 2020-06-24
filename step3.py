# coding=utf-8 

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_table
import regressionLib
import pandas as pd

# 0 CARICA DATASETprint("CARICA DATASET")
dataset_list = ["boston", "diabetes"]
X=Y = None #,df = regressionLib.load_dataset(name="boston")
button_pressed = False
#print("boston: " , X.shape)
#X,Y,df = regressionLib.load_dataset(name="diabetes")
#print("diabetes: " + str(X.shape))


# 2 SCELGO IL MODELLO MIGLIORE
print("SCELGO MODELLO MIGLIORE")
model_list = ["Linear", "Huber", "TheilSen","Ridge","Lasso","ElasticNet","KNN","DecisionTree","SVR","AdaBoost","GradientBoost","RandomForest","ExtraTrees"]
chosen_model = "RandomForest"
model_name = "test.sav"
#fig_train, fig_test = regressionLib.train(X,Y,selected=chosen_model,modelName=model_name)

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__) #, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    
        html.Div(
        [   
            dcc.Markdown('''
                # Regressione
                
                Questa dashboard permette di:

                1. Selezionare/Caricare il dataset 
                2. Selezionare l'algoritmo migliore
                3. Creare il modello
                4. Testare il modello

                '''),  
        ],className="twelve columns pretty_container", style={'text-align': 'center'}),
        
        html.Div([
            html.H3("1. Selezionare/Caricare il dataset"),
            dcc.Dropdown(
                id="dropdown-models-dataset",
                options=[{'label':nome, 'value':nome} for nome in dataset_list],
                value="boston",
                searchable=True,
                multi=False
            ),        
            dcc.Input(
                id="url",
                placeholder="Csv/Excel Link",
                style={
                    'text-align': 'center',
                    'background' : 'bottom',
                    'color' : 'white',
                    'margin-top': '1%'
                    }
            ),
            html.Div(id="html-box",style={'padding': '20px 20px 20px 20px', 'margin' : '10% 1% 1% 1%'}),      
        ], className="twelve columns pretty_container",style={'text-align': 'center'}),



        html.Div([
            html.H3("2. Selezionare l'algoritmo migliore: Cross Validation"),
            html.Button('Compare Models', id='compare-models', n_clicks=0),
            html.Div([
                dcc.Graph(id='figure-compare-models', figure=[])
            ],style={'padding': '20px 20px 20px 20px'}), 
        ], className="twelve columns pretty_container",style={'text-align': 'center'}),
    
        html.Div([
            html.H3("3. Creare il modello"),
            dcc.Dropdown(
                id="dropdown-models",
                options=[{'label':nome + " bla", 'value':nome} for nome in model_list],
                value="Linear",
                searchable=True,
                multi=False
            ),              
            dcc.Graph(id='fig-train',figure=[]),
            dcc.Graph(id='fig-test',figure=[]),
        ], className="twelve columns pretty_container",style={'text-align': 'center'}),
    ]
)


@app.callback(
    Output('figure-compare-models', 'figure'),
    [Input('compare-models', 'n_clicks')])
def update_output(n_clicks):
    # 1 COMPARA I MODELLI
    fig_compare_models = []
    if n_clicks>=1:
      global button_pressed
      if button_pressed==False:
        button_pressed = True
        fig_compare_models = []
        print("COMPARA MODELLI")
        global X,Y
        fig_compare_models = regressionLib.compare_models(X,Y)
        button_pressed =False

    return fig_compare_models


@app.callback(
    [Output('fig-train', 'figure'),Output('fig-test', 'figure')],
    [Input('dropdown-models', 'value')])
def upload_dataset(dropdown_models_value):
  print(dropdown_models_value)
  # Compare models
  fig_train = fig_test = []
  global X,Y
  if X is not None and Y is not None:
    print(X.shape)
    fig_train, fig_test = regressionLib.train(X,Y,selected=dropdown_models_value,modelName="test.sav")
  return  fig_train, fig_test


@app.callback(Output('html-box', 'children'),
              [Input('dropdown-models-dataset', 'value'),
              Input('url', 'value')])
def upload_dataset_(dropdown_models_value, url_value):
  print("entered")
  df = []
  correct = False
  try:
    print(url_value)
    df = pd.read_csv(url_value)
    print("dataset loaded")
    correct = True
  except:
    correct = False

  global X,Y
  if correct==False:   
    print(dropdown_models_value)
    X,Y,df = regressionLib.load_dataset(name=dropdown_models_value)
  else:
    print("dataset csv assignment")
    X = df.iloc[:,0:-1].values
    Y = df.iloc[:,-1].values
  
  children = [
      html.Div([
              dash_table.DataTable(
                  data=df.to_dict('records'),
                  columns=[{'name': i, 'id': i} for i in df.columns],
                  style_as_list_view=False,
                  #style_header={"display" : "none"},
                  style_cell={
                      'textAlign': 'center',
                      'backgroundColor': 'rgb(44, 44, 44)',
                      'color': 'white',
                      'minWidth': 95,
                      'width': 95, 
                      'maxWidth': 95,
                  }, 
                  fixed_rows={'headers': True},
                  #page_size=20,
                  style_table={'height': '600px', 'overflowY': 'auto'}
              ),
          ], style={'text-align': 'center'})
  ]
  return children 

if __name__ == '__main__':
    app.run_server(host="0.0.0.0",debug=False) #port=8900) #host="0.0.0.0", port=8900)
