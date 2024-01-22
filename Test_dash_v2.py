# -*- coding: utf-8 -*-
#01/22/2024: created
 
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

df = pd.read_csv('data/20220512.000000-20220809.000000_m5_stats.csv')

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='M5 Data Dashboard', style={'textAlign':'center'}),
    dcc.Tabs([
        dcc.Tab(
            label='LINE PLOT',
            children=[html.Label('Variable A',style={'font-weight':'bold'}),
            dcc.Dropdown(df.columns, 'Sonic_x_clean_119m [m/s]', id='ln_var1'),
            html.Br(),
            html.Label('Variable B',style={'font-weight':'bold'}),
            dcc.Dropdown(df.columns, 'Sonic_x_corr_119m [m/s]', id='ln_var2'),
            dcc.Graph(id='line-plot')],
            style={'font-weight': 'bold'}),
        
        dcc.Tab(
            label='SCATTER PLOT',
            children=[html.Label('Variable A',style={'font-weight':'bold'}),
            dcc.Dropdown(df.columns, 'Sonic_u_corr_119m [m/s]', id='sc_var1'),
            html.Br(),
            html.Label('Variable B',style={'font-weight':'bold'}),
            dcc.Dropdown(df.columns, 'Sonic_uu_corr_119m [m^2/s^2]', id='sc_var2'),
            dcc.Graph(id='scatter-plot')],
            style={'font-weight': 'bold'})
    ]),
])

@callback(
    Output('line-plot', 'figure'),
    Input('ln_var1', 'value'),
    Input('ln_var2', 'value')
)
def update_graph_ln(ln_var1,ln_var2):
    fig=px.line(df, x='Time (UTC)', y=[ln_var1,ln_var2],color_discrete_sequence=['black', 'green'])
    return fig


@callback(
    Output('scatter-plot', 'figure'),
    Input('sc_var1', 'value'),
    Input('sc_var2', 'value')
)
def update_graph_sc(sc_var1,sc_var2):
    fig=px.scatter(x=df[sc_var1],y=df[sc_var2])
    fig.update_xaxes(title=sc_var1)
    fig.update_yaxes(title=sc_var2)
    return fig

if __name__ == '__main__':
    app.run(debug=True)
