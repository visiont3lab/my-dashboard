import dash
import dash_html_components as html
import dash_core_components as dcc
#from flask_ngrok import run_with_ngrok

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot_matplotlib(df,xx_string,yy_string,provincia, style='go--',mylabel="totale_casi"):
    with plt.style.context("dark_background"):
        xx = df[xx_string].values
        yy = df[yy_string].values
        plt.figure(figsize=(18,8))
        plt.plot()
        plt.plot(xx,yy,style,linewidth=2,label=mylabel)
        plt.legend()
        plt.title(provincia + "Analisi Matplotlib")
        plt.xlabel("data")
        plt.ylabel("totale casi")
        plt.savefig("assets/images/matplotlib.png", transparent=False)
        plt.show()

def plot_pandas(df,xx_string,yy_string, provincia):
    with plt.style.context("seaborn"):
        ax = df.set_index(xx_string).plot(y=yy_string,figsize=(18,8),title=provincia +" Analisi Pandas",grid=True)
        ax.figure.savefig("assets/images/pandas-matplotlib.png", transparent=False)
        plt.show()

def plot_plotly(df,xx_string,yy_string,provincia):
    xx = df[xx_string].values.tolist()
    yy = df[yy_string].values.tolist()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
            x = xx,
            y = yy,
            name= "totale_casi",
            mode="lines+markers",
        )
    )
    fig.update_layout(
        title=dict(
            text =provincia + " Analisi Plotly",
            y = 0.9,
            x = 0.5,
            xanchor = "center",
            yanchor = "top",
        ),
        legend=dict(
            y = 0.9,
            x = 0.03,
        ),
        xaxis_title="data",
        yaxis_title="totale casi",
        hovermode='x',  #['x', 'y', 'closest', False]
        plot_bgcolor = "rgb(10,10,10)",
        paper_bgcolor="rgb(0,0,0)"
    )
    return fig
    #fig.write_image("assets/images/plotly.png")

def plot_regione(df,regione):
    df_filt_reg = df[df["denominazione_regione"]==regione]

    fig = go.Figure()
    province = list(df_filt_reg["denominazione_provincia"].unique())
    province.remove("In fase di definizione/aggiornamento")
    for provincia in province:
        df_filt_prov = df_filt_reg[df_filt_reg["denominazione_provincia"]==provincia]
        xx = df_filt_prov["data"]
        yy = df_filt_prov["totale_casi"].values
        fig.add_trace(go.Scatter(x = xx,y = yy,name=provincia + ": totale casi" ,mode="lines+markers"))
        fig.update_layout(title=regione, hovermode="x",plot_bgcolor = "rgb(10,10,10)",
        paper_bgcolor="rgb(0,0,0)")
    return fig

df = pd.read_csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv")
df["data"] = pd.to_datetime(df["data"]).dt.date
provincia = "Ravenna"
df_filt = df[df["denominazione_provincia"]==provincia]

#plot_pandas(df_filt,"data","totale_casi", provincia)
#plot_matplotlib(df_filt,"data","totale_casi", provincia)
fig = plot_plotly(df_filt,"data","totale_casi", provincia)
fig_reg = plot_regione(df,"Emilia-Romagna")
#fig.show()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
#run_with_ngrok(server) 

app.layout = html.Div([
        html.Div(
        [   
            dcc.Markdown('''
                # Analisi Covid 19 (nuovo update) CddsHECKK  fdsfdsfdsfsa asd
                Andremo a eseguire un esercizio sul dataset covid.
                
                ## Esercizio n.1 
                >  Plottare l'andamento nel tempo dei contagiati della propria provincia.
                '''),  
        ]),
        html.Div(
        [   
            dcc.Graph(figure=fig),
        ]),
        html.Div(
        [   
            dcc.Markdown('''
                ## Esercizio n.2
                >  Plottare l'andamento nel tempo dei contagiati di tutte le pronvice dell'emilia romagna.
                '''),  
        ]),
        html.Div(
        [   
            dcc.Graph(figure=fig_reg),
        ]),
    ],style={"margin": "0", "padding": "0"})

if __name__ == '__main__':
    #server.run()
    app.run_server(host="0.0.0.0",debug=True,port=8900) #host="0.0.0.0", port=8900)
