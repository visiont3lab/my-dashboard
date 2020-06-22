import dash
import dash_html_components as html
import dash_core_components as dcc
#from flask_ngrok import run_with_ngrok
from dash.dependencies import Input, Output, State
import dash_table

import pandas as pd
from sklearn import datasets
import mylib 
import base64
import io 
import datetime

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xlsx' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    global X,Y
    X = df.iloc[:,0:-1].values
    Y = df.iloc[:,-1].values
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

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

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

processing = False
X = None
Y = None
datasets_list = ["reset","load","boston", "diabetes"] #,"tips"]
model_list = ["Linear", "Huber", "TheilSen","Ridge","Lasso","ElasticNet","KNN","DecisionTree","SVR","AdaBoost","GradientBoost","RandomForest","ExtraTrees"]


#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__) #, external_stylesheets=external_stylesheets)
server = app.server
#run_with_ngrok(server) 

app.layout = html.Div([
    
        html.Div(
        [   
            dcc.Markdown('''
                # Regressione
                
                Questa dashboard permette di:

                1. Selezionare il dataset 
                2. Selezionare l'algoritmo migliore
                3. Creare il modello
                4. Testare il modello

                '''),  
        ],className="twelve columns pretty_container", style={'text-align': 'center'}),
        
        html.Div([
            html.H3("1. Selezionare il dataset"),
            dcc.Dropdown(
                id="dropdown-datasets",
                options=[{'label':nome, 'value':nome} for nome in datasets_list],
                value="boston",
                searchable=True,
                multi=False
            ), 

            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Div(id='uploaded-dataset'),
            
        ], className="twelve columns pretty_container",style={'text-align': 'center'}),

        html.Div([
            html.H3("2. Selezionare l'algoritmo migliore: Cross Validation"),
            html.Button('Compare Models', id='process', n_clicks=0),
            html.Div([
                dcc.Graph(id='figure-compare-models', figure=[])
            ],style={'padding': '20px 20px 20px 20px'}), 
        ], className="twelve columns pretty_container",style={'text-align': 'center'}),
    
        html.Div([
            html.H3("3. Creare il modello"),
            dcc.Dropdown(
                id="dropdown-models",
                options=[{'label':nome, 'value':nome} for nome in model_list],
                value="Linear",
                searchable=True,
                multi=False
            ),  
            dcc.Graph(id='fig-train'),
            dcc.Graph(id='fig-test'),
        ], className="twelve columns pretty_container",style={'text-align': 'center'}),
    
    ]
)

@app.callback(
    [dash.dependencies.Output('fig-train', 'figure'),
    dash.dependencies.Output('fig-test', 'figure')],
    [dash.dependencies.Input('dropdown-models', 'value')])
def upload_dataset(dropdown_models_value):
    # Compare models
    global X,Y
    fig_train, fig_test = mylib.train(X,Y,selected=dropdown_models_value,modelName="test.sav")
    return fig_train,fig_test

@app.callback(
    dash.dependencies.Output('figure-compare-models', 'figure'),
    [dash.dependencies.Input('process', 'n_clicks')])
def update_image(n_clicks):
    fig_compare_models = []
    if n_clicks>=1:
        global X,Y
        fig_compare_models = mylib.compare_models(X,Y)
    return fig_compare_models


@app.callback(Output('uploaded-dataset', 'children'),
              [Input('dropdown-datasets', 'value'), Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def upload_dataset(dropdown_datasets_value,list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None and dropdown_datasets_value=="load":
        children = [
            parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]
        return children
    elif dropdown_datasets_value!="load" and dropdown_datasets_value!="reset":
        #global processing
        #if processing==False:
        #    processing = True
        # https://dash.plotly.com/datatable/height
        # https://dash.plotly.com/datatable
        # https://scikit-learn.org/stable/modules/cross_validation.html
        boston = datasets.load_boston()
        df_boston = pd.DataFrame(boston.data,columns=boston.feature_names)
        df_boston["price"] = boston.target

        diabetes = datasets.load_diabetes()
        df_diabetes = pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
        df_diabetes["desease"] = diabetes.target
        df_diabetes = df_diabetes.round(decimals=5)
        
        tips = pd.read_csv('https://frenzy86.s3.eu-west-2.amazonaws.com/fav/tips.csv')
        dict_datasets = {
            "boston": df_boston,
            "diabetes": df_diabetes,
            "tips": tips
        }    
        df = dict_datasets[dropdown_datasets_value]#.head(10)
        df = df.dropna()
        
        # Compare models
        global X,Y
        X = df.iloc[:,0:-1].values
        Y = df.iloc[:,-1].values
        #fig_compare_models = mylib.compare_models(X,Y)

        children = [
            html.Div([
                    html.H5(dropdown_datasets_value),
                    html.Hr(),  # horizontal line
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
        #    processing = False
        return children #, fig_compare_models
    else:
        return []

'''
@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children
'''
        
if __name__ == '__main__':
    #server.run()
    app.run_server(host="0.0.0.0",debug=False,port=8900) #host="0.0.0.0", port=8900)
