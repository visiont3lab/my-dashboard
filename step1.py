# coding=utf-8 
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_table
import regressionLib

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__) #, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    html.Div([
        
        html.Div([
               
                html.Div([
                    dcc.Markdown('''
                    Intro
                    ''')
                    ],
                    id="info",
                    className="four columns info_container "),

                html.Div([   
                      html.P("Table 1")
                      ], id='table-one-layout', className="four columns info_container "),

                html.Div([   
                      html.P("Table 2")
                      ], id='table-two-layout', className="four columns info_container "),

        ], className="row"),

        
        html.Div([
               
                html.Div([
                    dcc.Markdown('''
                    BLocco2
                    ''')
                    ],
                    id="fdfds",
                    className="six columns info_container "),

                html.Div([   
                      html.P("Blocco 2bis")
                      ], id='dsfds', className="six columns info_container "),

        ], className="row"),


        html.Div([
                html.Div([
                    html.H3("Analisi Nazionale"),
                ], className="six columns pretty_container"),
                html.Div([
                    html.H3("Andamento Nuovi Totali Positivi"),
                ], className="six columns pretty_container"),
            ], className="row"),
        html.Div([
                html.Div([
                    html.H3("Analisi Regionale"),
                ],className="six columns pretty_container"),
                html.Div([
                    html.H3("Mappa"),
                    ], className="six columns pretty_container"),
            ], className="row"), 
        ])
    ],id="main")

if __name__ == '__main__':
    app.run_server(host="0.0.0.0",debug=True) #, host="0.0.0.0", port=8800)
