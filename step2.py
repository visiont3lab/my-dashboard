# coding=utf-8 

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_table
import regressionLib

# 0 CARICA DATASETprint("CARICA DATASET")
X,Y = regressionLib.load_dataset(name="boston")

# 1 COMPARA I MODELLI
print("COMPARA MODELLI")
fig_compare_models = regressionLib.compare_models(X,Y)

# 2 SCELGO IL MODELLO MIGLIORE
print("SCELGO MODELLO MIGLIORE")
model_list = ["Linear", "Huber", "TheilSen","Ridge","Lasso","ElasticNet","KNN","DecisionTree","SVR","AdaBoost","GradientBoost","RandomForest","ExtraTrees"]
chosen_model = "RandomForest"
model_name = "test.sav"
fig_train, fig_test = regressionLib.train(X,Y,selected=chosen_model,modelName=model_name)

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
            html.H3("2. Selezionare l'algoritmo migliore: Cross Validation"),
            html.Div([
                dcc.Graph(id='figure-compare-models', figure=fig_compare_models)
            ],style={'padding': '20px 20px 20px 20px'}), 
        ], className="twelve columns pretty_container",style={'text-align': 'center'}),
    
        html.Div([
            html.H3("3. Creare il modello"),
            html.P("Testo "),
            
            dcc.Markdown('''
                  ### 3. Creare il Modello 
                  Testo
                  '''),
            dcc.RadioItems(
                options=[
                    {'label': 'New York City', 'value': 'NYC'},
                    {'label': 'Montréal', 'value': 'MTL'},
                    {'label': 'San Francisco', 'value': 'SF'}
                ],
                value='MTL'
            ),  
            dcc.Dropdown(
                id='demo-dropdown',
                options=[
                    {'label': 'New York City', 'value': 'NYC'},
                    {'label': 'Montreal', 'value': 'MTL'},
                    {'label': 'San Francisco', 'value': 'SF'}
                ],
                value='NYC'
            ),
            dcc.Checklist(
                options=[
                    {'label': 'New York City', 'value': 'NYC'},
                    {'label': 'Montréal', 'value': 'MTL'},
                    {'label': 'San Francisco', 'value': 'SF'}
                ],
                value=['NYC', 'MTL']
            ),  
            dcc.Graph(id='fig-train',figure=fig_train),
            dcc.Graph(id='fig-test',figure=fig_test),
        ], className="twelve columns pretty_container",style={'text-align': 'center'}),
    ]
)

        
if __name__ == '__main__':
    app.run_server(host="0.0.0.0",debug=False) #port=8900) #host="0.0.0.0", port=8900)
