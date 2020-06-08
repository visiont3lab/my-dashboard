{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "dashboard.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyM285KvdcJHv4oh6Mt/Zb21"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "UismT2ihBlOL",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 126
        },
        "outputId": "deacf427-ff41-4a3f-9c77-9c42e7f997f1"
      },
      "source": [
        "!git clone https://github.com/visiont3lab/my-dashboard.git"
      ],
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Cloning into 'my-dashboard'...\n",
            "remote: Enumerating objects: 74, done.\u001b[K\n",
            "remote: Counting objects: 100% (74/74), done.\u001b[K\n",
            "remote: Compressing objects: 100% (53/53), done.\u001b[K\n",
            "remote: Total 74 (delta 31), reused 32 (delta 12), pack-reused 0\u001b[K\n",
            "Unpacking objects: 100% (74/74), done.\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WNqXvus7B_FS",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "outputId": "a5c0dfe9-e657-491a-f4a1-517daf1f77f1"
      },
      "source": [
        "%cd my-dashboard/\n"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/content/my-dashboard\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pEeZ3EFhCR2m",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "%%capture\n",
        "!pip install -r requirements.txt"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "I92Mw_tuCjdm",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!pip install flask-ngrok"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LS_v7_zBC5vU",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "outputId": "972706c3-0da8-4119-b695-c08b9c9abd4f"
      },
      "source": [
        "%%writefile assets/typography.css\n",
        "body {\n",
        "    padding: 20px 300px 0px 300px;\n",
        "    margin : 0px 0px 0px  0px;\n",
        "    /*padding : 0px 0px 0px 0px;*/\n",
        "    background-color: red;\n",
        "}"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Overwriting assets/typography.css\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3YDpxNT4DPzz",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "outputId": "200c69c1-ba8c-4dd6-a6b2-6a7119d45cb8"
      },
      "source": [
        "import dash\n",
        "import dash_html_components as html\n",
        "import dash_core_components as dcc\n",
        "from flask_ngrok import run_with_ngrok\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import plotly.graph_objects as go\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "def plot_matplotlib(df,xx_string,yy_string,provincia, style='go--',mylabel=\"totale_casi\"):\n",
        "    with plt.style.context(\"dark_background\"):\n",
        "        xx = df[xx_string].values\n",
        "        yy = df[yy_string].values\n",
        "        plt.figure(figsize=(18,8))\n",
        "        plt.plot()\n",
        "        plt.plot(xx,yy,style,linewidth=2,label=mylabel)\n",
        "        plt.legend()\n",
        "        plt.title(provincia + \"Analisi Matplotlib\")\n",
        "        plt.xlabel(\"data\")\n",
        "        plt.ylabel(\"totale casi\")\n",
        "        plt.savefig(\"assets/images/matplotlib.png\", transparent=False)\n",
        "        plt.show()\n",
        "\n",
        "def plot_pandas(df,xx_string,yy_string, provincia):\n",
        "    with plt.style.context(\"seaborn\"):\n",
        "        ax = df.set_index(xx_string).plot(y=yy_string,figsize=(18,8),title=provincia +\" Analisi Pandas\",grid=True)\n",
        "        ax.figure.savefig(\"assets/images/pandas-matplotlib.png\", transparent=False)\n",
        "        plt.show()\n",
        "\n",
        "def plot_plotly(df,xx_string,yy_string,provincia):\n",
        "    xx = df[xx_string].values.tolist()\n",
        "    yy = df[yy_string].values.tolist()\n",
        "\n",
        "    fig = go.Figure()\n",
        "\n",
        "    fig.add_trace(go.Scatter(\n",
        "            x = xx,\n",
        "            y = yy,\n",
        "            name= \"totale_casi\",\n",
        "            mode=\"lines+markers\",\n",
        "        )\n",
        "    )\n",
        "    fig.update_layout(\n",
        "        title=dict(\n",
        "            text =provincia + \" Analisi Plotly\",\n",
        "            y = 0.9,\n",
        "            x = 0.5,\n",
        "            xanchor = \"center\",\n",
        "            yanchor = \"top\",\n",
        "        ),\n",
        "        legend=dict(\n",
        "            y = 0.9,\n",
        "            x = 0.03,\n",
        "        ),\n",
        "        xaxis_title=\"data\",\n",
        "        yaxis_title=\"totale casi\",\n",
        "        hovermode='x',  #['x', 'y', 'closest', False]\n",
        "        plot_bgcolor = \"rgb(10,10,10)\",\n",
        "        paper_bgcolor=\"rgb(0,0,0)\"\n",
        "    )\n",
        "    return fig\n",
        "    #fig.write_image(\"assets/images/plotly.png\")\n",
        "\n",
        "def plot_regione(df,regione):\n",
        "    df_filt_reg = df[df[\"denominazione_regione\"]==regione]\n",
        "\n",
        "    fig = go.Figure()\n",
        "    province = list(df_filt_reg[\"denominazione_provincia\"].unique())\n",
        "    province.remove(\"In fase di definizione/aggiornamento\")\n",
        "    for provincia in province:\n",
        "        df_filt_prov = df_filt_reg[df_filt_reg[\"denominazione_provincia\"]==provincia]\n",
        "        xx = df_filt_prov[\"data\"]\n",
        "        yy = df_filt_prov[\"totale_casi\"].values\n",
        "        fig.add_trace(go.Scatter(x = xx,y = yy,name=provincia + \": totale casi\" ,mode=\"lines+markers\"))\n",
        "        fig.update_layout(title=regione, hovermode=\"x\",plot_bgcolor = \"rgb(10,10,10)\",\n",
        "        paper_bgcolor=\"rgb(0,0,0)\")\n",
        "    return fig\n",
        "\n",
        "df = pd.read_csv(\"https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv\")\n",
        "df[\"data\"] = pd.to_datetime(df[\"data\"]).dt.date\n",
        "provincia = \"Ravenna\"\n",
        "df_filt = df[df[\"denominazione_provincia\"]==provincia]\n",
        "\n",
        "#plot_pandas(df_filt,\"data\",\"totale_casi\", provincia)\n",
        "#plot_matplotlib(df_filt,\"data\",\"totale_casi\", provincia)\n",
        "fig = plot_plotly(df_filt,\"data\",\"totale_casi\", provincia)\n",
        "fig_reg = plot_regione(df,\"Emilia-Romagna\")\n",
        "#fig.show()\n",
        "\n",
        "external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']\n",
        "\n",
        "app = dash.Dash(__name__, external_stylesheets=external_stylesheets)\n",
        "server = app.server\n",
        "run_with_ngrok(server) \n",
        "\n",
        "app.layout = html.Div([\n",
        "        html.Div(\n",
        "        [   \n",
        "            dcc.Markdown('''\n",
        "                # Analisi Covid 19 (nuovo update) Ciao  fdsfdsfdsfsa asd\n",
        "                Andremo a eseguire un esercizio sul dataset covid.\n",
        "                \n",
        "                ## Esercizio n.1 \n",
        "                >  Plottare l'andamento nel tempo dei contagiati della propria provincia.\n",
        "                '''),  \n",
        "        ]),\n",
        "        html.Div(\n",
        "        [   \n",
        "            dcc.Graph(figure=fig),\n",
        "        ]),\n",
        "        html.Div(\n",
        "        [   \n",
        "            dcc.Markdown('''\n",
        "                ## Esercizio n.2\n",
        "                >  Plottare l'andamento nel tempo dei contagiati di tutte le pronvice dell'emilia romagna.\n",
        "                '''),  \n",
        "        ]),\n",
        "        html.Div(\n",
        "        [   \n",
        "            dcc.Graph(figure=fig_reg),\n",
        "        ]),\n",
        "    ],style={\"margin\": \"0\", \"padding\": \"0\"})\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    server.run()\n",
        "    #app.run_server(host=\"0.0.0.0\") #,debug=True,port=8900) #host=\"0.0.0.0\", port=8900)\n"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            " * Serving Flask app \"__main__\" (lazy loading)\n",
            " * Environment: production\n",
            "\u001b[31m   WARNING: This is a development server. Do not use it in a production deployment.\u001b[0m\n",
            "\u001b[2m   Use a production WSGI server instead.\u001b[0m\n",
            " * Debug mode: off\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            " * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            " * Running on http://b38bcdd9dba6.ngrok.io\n",
            " * Traffic stats available on http://127.0.0.1:4040\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "127.0.0.1 - - [08/Jun/2020 17:49:50] \"\u001b[37mGET / HTTP/1.1\u001b[0m\" 200 -\n",
            "127.0.0.1 - - [08/Jun/2020 17:49:50] \"\u001b[37mGET /assets/typography.css?m=1591638495.9949372 HTTP/1.1\u001b[0m\" 200 -\n",
            "127.0.0.1 - - [08/Jun/2020 17:49:50] \"\u001b[37mGET /_dash-component-suites/dash_renderer/polyfill@7.v1_2_2m1591638339.7.0.min.js HTTP/1.1\u001b[0m\" 200 -\n",
            "127.0.0.1 - - [08/Jun/2020 17:49:50] \"\u001b[37mGET /_dash-component-suites/dash_renderer/react@16.v1_2_2m1591638339.8.6.min.js HTTP/1.1\u001b[0m\" 200 -\n",
            "127.0.0.1 - - [08/Jun/2020 17:49:50] \"\u001b[37mGET /_dash-component-suites/dash_renderer/react-dom@16.v1_2_2m1591638339.8.6.min.js HTTP/1.1\u001b[0m\" 200 -\n",
            "127.0.0.1 - - [08/Jun/2020 17:49:50] \"\u001b[37mGET /_dash-component-suites/dash_renderer/prop-types@15.v1_2_2m1591638339.7.2.min.js HTTP/1.1\u001b[0m\" 200 -\n",
            "127.0.0.1 - - [08/Jun/2020 17:49:50] \"\u001b[37mGET /_dash-component-suites/dash_core_components/dash_core_components.v1_8_1m1591638341.min.js HTTP/1.1\u001b[0m\" 200 -\n",
            "127.0.0.1 - - [08/Jun/2020 17:49:51] \"\u001b[37mGET /_dash-component-suites/dash_core_components/dash_core_components-shared.v1_8_1m1591638341.js HTTP/1.1\u001b[0m\" 200 -\n",
            "127.0.0.1 - - [08/Jun/2020 17:49:51] \"\u001b[37mGET /_dash-component-suites/dash_html_components/dash_html_components.v1_0_2m1591638341.min.js HTTP/1.1\u001b[0m\" 200 -\n",
            "127.0.0.1 - - [08/Jun/2020 17:49:51] \"\u001b[37mGET /_dash-component-suites/dash_renderer/dash_renderer.v1_2_2m1591638339.min.js HTTP/1.1\u001b[0m\" 200 -\n",
            "127.0.0.1 - - [08/Jun/2020 17:49:51] \"\u001b[37mGET /_dash-dependencies HTTP/1.1\u001b[0m\" 200 -\n",
            "127.0.0.1 - - [08/Jun/2020 17:49:51] \"\u001b[37mGET /_favicon.ico?v=1.9.1 HTTP/1.1\u001b[0m\" 200 -\n",
            "127.0.0.1 - - [08/Jun/2020 17:49:51] \"\u001b[37mGET /_dash-layout HTTP/1.1\u001b[0m\" 200 -\n",
            "127.0.0.1 - - [08/Jun/2020 17:49:51] \"\u001b[37mGET /_dash-component-suites/dash_core_components/async-graph.v1_8_1m1582838719.js HTTP/1.1\u001b[0m\" 200 -\n",
            "127.0.0.1 - - [08/Jun/2020 17:49:51] \"\u001b[37mGET /_dash-component-suites/dash_core_components/async-markdown.v1_8_1m1582838719.js HTTP/1.1\u001b[0m\" 200 -\n",
            "127.0.0.1 - - [08/Jun/2020 17:49:52] \"\u001b[37mGET /_dash-component-suites/dash_core_components/async-plotlyjs.v1_8_1m1582838719.js HTTP/1.1\u001b[0m\" 200 -\n",
            "127.0.0.1 - - [08/Jun/2020 17:49:52] \"\u001b[37mGET /_dash-component-suites/dash_core_components/async-highlight.v1_8_1m1582838719.js HTTP/1.1\u001b[0m\" 200 -\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "H_gHOOPNCOEW",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 254
        },
        "outputId": "e4e1137e-a8d5-4172-de6d-adfc05ceedff"
      },
      "source": [
        "!python app.py"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            " * Serving Flask app \"app\" (lazy loading)\n",
            " * Environment: production\n",
            "\u001b[31m   WARNING: This is a development server. Do not use it in a production deployment.\u001b[0m\n",
            "\u001b[2m   Use a production WSGI server instead.\u001b[0m\n",
            " * Debug mode: off\n",
            " * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)\n",
            " * Running on http://26754b7a6fe5.ngrok.io\n",
            " * Traffic stats available on http://127.0.0.1:4040\n",
            "Error in atexit._run_exitfuncs:\n",
            "Traceback (most recent call last):\n",
            "  File \"/usr/local/lib/python3.6/dist-packages/matplotlib/_pylab_helpers.py\", line 76, in destroy_all\n",
            "    gc.collect(1)\n",
            "KeyboardInterrupt\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kIBd5MLPCPlG",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 217
        },
        "outputId": "ae2cb806-9ed5-4812-f31b-760aac05348c"
      },
      "source": [
        "!git status\n"
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "On branch master\n",
            "Your branch is up to date with 'origin/master'.\n",
            "\n",
            "Changes not staged for commit:\n",
            "  (use \"git add <file>...\" to update what will be committed)\n",
            "  (use \"git checkout -- <file>...\" to discard changes in working directory)\n",
            "\n",
            "\t\u001b[31mmodified:   app.py\u001b[m\n",
            "\t\u001b[31mmodified:   assets/typography.css\u001b[m\n",
            "\n",
            "no changes added to commit (use \"git add\" and/or \"git commit -a\")\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "R84AsnL6Drrm",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!git add app.py assets/typography.css"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Uq-Ep7MFEELd",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!git config --global user.email \"visiont3lab@gmail.com\""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hS5GDjVcELiW",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!git config --global user.name \"visiont3lab\""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "5Wkh1TBMEPnU",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 54
        },
        "outputId": "cdd80dc6-8e3b-4094-8422-a2d1840fd6ca"
      },
      "source": [
        "!git commit -m \"Change colab\""
      ],
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[master d8fc484] Change colabÇ\n",
            " 2 files changed, 8 insertions(+), 8 deletions(-)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yqUfPmWvERvX",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "outputId": "024ed90d-1216-4643-ac7e-9e8bd050fbc2"
      },
      "source": [
        "!git push -u origin master"
      ],
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "fatal: could not read Username for 'https://github.com': No such device or address\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SMTjOyCTEVhs",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}