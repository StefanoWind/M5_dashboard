# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:30:20 2024

@author: sletizia
"""
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, callback, Output, Input
import plotly.graph_objs as go
import pandas as pd
from PIL import Image
import base64
import re
import numpy as np

#%% Inputs
source_data_raw='data/20220601.000000-20220701.000000.nwtc.m5_20hz.a0.csv'
source_data_qc='data/20220601.000000-20220701.000000.nwtc.m5_20hz.b0.csv'
source_image='data/M5.png'
source_logo='data/NREL-logo.png'
source_fc='data/FC.png'
source_m5_pic='data/M5_pic.png'

N_bins=10

#%% Initialization
Data_raw= pd.read_csv(source_data_raw)
Data_qc = pd.read_csv(source_data_qc)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

logo = base64.b64encode(open(source_logo, 'rb').read()).decode('ascii')


FC_pic = base64.b64encode(open(source_fc, 'rb').read()).decode('ascii')

m5_pic=dbc.Container([dcc.Graph(id='m5_pic')])

tower_ln= dbc.Container([dcc.Graph(id='tower_ln',style={'height': '95vh',"width":"60vh"})])
tower_sc= dbc.Container([dcc.Graph(id='tower_sc',style={'height': '95vh',"width":"60vh"})])
            
dropdown_ln=dbc.Container([
        html.Label('Variable',style={'font-weight':'bold'}),
        dcc.Dropdown(Data_qc.columns, 'Sonic_u_119m [m/s]', id='ln_var')])

dropdown_sc1=dbc.Container([
        html.Label('Variable A',style={'font-weight':'bold'}),
        dcc.Dropdown(Data_qc.columns, 'Sonic_u_119m [m/s]', id='sc_var1')])

dropdown_sc2=dbc.Container([
        html.Label('Variable B',style={'font-weight':'bold'}),
        dcc.Dropdown(Data_qc.columns, 'Sonic_u_61m [m/s]', id='sc_var2')])

line_plot=   dbc.Container([dcc.Graph(id='line_plot',   style={'height':'35vh'})],style={'margin-top': '2vh'})

hist=        dbc.Container([dcc.Graph(id='hist_plot',   style={'height':'35vh'})],style={'margin-top': '2vh'})

scatter_plot=dbc.Container([dcc.Graph(id='scatter_plot',style={'height':'35vh'})],style={'margin-top': '2vh'})

pcolor_plot= dbc.Container([dcc.Graph(id='pcolor_plot', style={'height':'35vh'})],style={'margin-top': '2vh'})


#%% Main
app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dbc.Navbar([
        dbc.Container([
            dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Img(src='data:image/png;base64,{}'.format(logo),height="60vw", width="auto")
                    ],style={"textAlign": "left"}),
                dbc.Col([
                    html.H2('M5 Data Dashboard',style={'color': 'white'})
                    ],md=10)
                ],style={"width":"200vh"}),
            
            dbc.Row([
                dbc.Col([
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Home", href="/")),
                        dbc.NavItem(dbc.NavLink("Time series", href="/page-1")),
                        dbc.NavItem(dbc.NavLink("Capture matrix", href="/page-2")),
                        dbc.NavItem(dbc.NavLink("Documentation", href="/page-3")),
                        dbc.NavItem(dbc.NavLink("About", href="/page-4")),
                        ]),
                    ]),
                ],style={"width":"200vh"}),
               
            ]),
            ],fluid=True)
        ],
        color=f'rgb({0}, {121}, {194})',
        dark=True,
        style={"textAlign": "center",'font':'Cambria'},
    ),
    dbc.Container(id='page-content'),
 ],fluid=True)

@callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/':
        content=    dbc.Container([
                    dbc.Row([
                        dbc.Col([html.H2("The M5 tower: a state-of-the-art network for atmospheric mesurements"),
                                 html.Img(src='data:image/png;base64,{}'.format(FC_pic), height="750px"),
                                 html.H6("View of the Flatirons campus with zoomed picture of M5 tower"),
                                 html.H5("The 135-m M5 metereological tower provides unique high-quality measurements \
                                         of wind and thermodynamic properties of the atmospheric boundary layer.\
                                         The objectives of the 135-m meteorological mast installation project include:",
                                         style={"text-align": "justify",'font-size': '1.25rem'}),
                                 html.Ul([
                                    html.Li("Measure the inflow to utility scale turbines with high spatial and temporal resolution,"),
                                    html.Li("Record data that can be used to quantify 3-dimensional turbulence and atmospheric stability across the atmospheric boundary layer,"),
                                    html.Li(" Provide high-quality data to researchers within and outside NREL and the wind energy community.")
                                ],style={"text-align": "justify",'font-size': '1.25rem','font-weight':'500'})
                                ],
                                md=12,style={"text-align": "center"} ),
                        ]),
                    ])
    elif pathname == '/page-1':
        content=    dbc.Row([
                    dbc.Col(tower_ln,md=4),
                    dbc.Col([
                    dropdown_ln,
                    line_plot,
                    hist])
                    ])
        
    elif pathname == '/page-2':
        content=dbc.Row([
                dbc.Col(tower_sc,md=4),
                dbc.Col([
                    dropdown_sc1,
                    dropdown_sc2,
                    scatter_plot,
                    pcolor_plot
                ])
        ])
    else:
        content=None
     
    return content
            
@callback(
    Output('m5_pic', 'figure'),
    Input('url', 'pathname'),
)
def draw_m5(active_tab):
    if active_tab=='/':
        fig=go.Figure()
        img = Image.open(source_m5_pic)
        fig.add_layout_image(
            dict(
                source=img,
                xref="x",
                yref="y",
                x=0,
                y=140,
                sizex=70,
                sizey=140))
        fig.update_xaxes(range=[0,75],showticklabels=False,showgrid=False, zeroline=False)  # Set x-axis limits from 2 to 4
        fig.update_yaxes(range=[-2, 145],showticklabels=False,showgrid=False, zeroline=False)

    return fig

            
@callback(
    Output('tower_ln', 'figure'),
    Input('url', 'pathname'),
    Input('ln_var', 'value'),
)
def draw_tower_ln(active_tab,ln_var):
    if active_tab=='/page-1':
        corr={3:1.25,38:-1,87:2.2,122:1,15:0.5,41:-0.5,61:-0.4,74:1.8,100:-2,119:-0.25}
        fig=go.Figure()
        
        img = Image.open(source_image)
        fig.add_layout_image(
            dict(
                source=img,
                xref="x",
                yref="y",
                x=0,
                y=145,
                sizex=75,
                sizey=147,
                sizing="stretch",
                opacity=1,
                layer="below",
                )
        )
        
        fig.update_xaxes(range=[0,75],showticklabels=False,showgrid=False, zeroline=False)  # Set x-axis limits from 2 to 4
        fig.update_yaxes(range=[-2, 145],showticklabels=False,showgrid=False, zeroline=False)
    
        z1=np.float64(re.search(r'\d+', ln_var).group())
        if 'Sonic' in ln_var:
            x1=57.5
        else:
            x1=37.2
                
        z1=z1+corr[int(z1)]
        fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=x1-5, y0=z1-5, x1=x1+5, y1=z1+5,
            line_color="black",line_width=3,
        )
    else:
        fig=None
    
    return fig

@callback(
    Output('tower_sc', 'figure'),
    Input('url', 'pathname'),
    Input('sc_var1', 'value'),
    Input('sc_var2', 'value'),
)
def draw_tower_sc(active_tab,sc_var1,sc_var2):
    if active_tab=='/page-2':
        corr={3:1.25,38:-1,87:2.2,122:1,15:0.5,41:-0.5,61:-0.4,74:1.8,100:-2,119:-0.25}
        fig=go.Figure()
        
        img = Image.open(source_image)
        fig.add_layout_image(
            dict(
                source=img,
                xref="x",
                yref="y",
                x=0,
                y=145,
                sizex=75,
                sizey=147,
                sizing="stretch",
                opacity=1,
                layer="below")
        )
        fig.update_xaxes(range=[0,75],showticklabels=False,showgrid=False, zeroline=False)  # Set x-axis limits from 2 to 4
        fig.update_yaxes(range=[-2, 145],showticklabels=False,showgrid=False, zeroline=False)
    
        z1=np.float64(re.search(r'\d+', sc_var1).group())
        z2=np.float64(re.search(r'\d+', sc_var2).group())
        if 'Sonic' in sc_var1:
            x1=57.5
        else:
            x1=37.2
        if 'Sonic' in sc_var2:
            x2=57.5
        else:
            x2=37.2
           
        z1=z1+corr[int(z1)]
        fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=x1-5, y0=z1-5, x1=x1+5, y1=z1+5,
            line_color="black",line_width=3,
        )
    
        z2=z2+corr[int(z2)]
        fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=x2-4, y0=z2-4, x1=x2+4, y1=z2+4,
            line_color="black",line_width=3,
        )
    else:
        fig=None
        
    return fig

@callback(
    Output('line_plot', 'figure'),
    Input('ln_var', 'value'),
)
def update_graph_ln(ln_var):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=Data_qc['Time (UTC)'], y=Data_raw[ln_var],mode='lines',line=dict(color='red'),name='Raw data'))
    fig.add_trace(go.Scatter(x=Data_qc['Time (UTC)'], y=Data_qc[ln_var],mode='lines',line=dict(color='green'),name='QC data'))
    fig.update_layout(
    paper_bgcolor=f'rgb({0}, {121}, {194})',
    margin=dict(l=0, r=3, t=2, b=0),
    legend=dict(yanchor="top",y=0.99,xanchor="right", x=0.99),
        font_family="Cambria",
        font_color="white",)
    fig.update_xaxes(title='Time (UTC)',showgrid=True,gridcolor='lightgrey',linecolor='black',mirror=True)
    fig.update_yaxes(title=ln_var,showgrid=True,gridcolor='lightgrey',linecolor='black',mirror=True)
    return fig


@callback(
    Output('hist_plot', 'figure'),
    Input('ln_var', 'value'),
)
def update_graph_hist(ln_var):
    fig=go.Figure()
    fig.add_trace(go.Histogram(x=Data_raw[ln_var],nbinsx=100,name='Raw data',marker_color='red'))
    fig.add_trace(go.Histogram(x=Data_qc[ln_var],nbinsx=100,name='QC data',marker_color='green'))
    fig.update_layout(
    paper_bgcolor=f'rgb({0}, {121}, {194})',
    margin=dict(l=0, r=3, t=2, b=0),
    legend=dict(yanchor="top",y=0.99,xanchor="right", x=0.99),
    font_family="Cambria",
    font_color="white",)
    fig.update_xaxes(title=ln_var,showgrid=True,gridcolor='lightgrey',linecolor='black',mirror=True)
    fig.update_yaxes(title='Number of occurences',showgrid=True,gridcolor='lightgrey',linecolor='black',mirror=True)
    return fig


@callback(
    Output('scatter_plot', 'figure'),
    Input('sc_var1', 'value'),
    Input('sc_var2', 'value')
)
def update_graph_sc(sc_var1,sc_var2):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=Data_qc[sc_var1], y=Data_qc[sc_var2],mode='markers', marker=dict(color='black',opacity=0.5)))
    fig.update_xaxes(title=sc_var1,showgrid=True,gridcolor='lightgrey',linecolor='black',mirror=True)
    fig.update_yaxes(title=sc_var2,showgrid=True,gridcolor='lightgrey',linecolor='black',mirror=True)
    fig.update_layout(
    paper_bgcolor=f'rgb({0}, {121}, {194})',
    margin=dict(l=0, r=3, t=2, b=0),
    font_family="Cambria",
    font_color="white",)
    x=Data_qc[sc_var1]
    y=Data_qc[sc_var2]
    x_bin_edges=np.linspace(np.nanpercentile(x, 5),np.nanpercentile(x, 95),N_bins)
    y_bin_edges=np.linspace(np.nanpercentile(y, 5),np.nanpercentile(y, 95),N_bins)
    x_bins = (x_bin_edges[1:] + x_bin_edges[:-1]) / 2
    y_bins = (y_bin_edges[1:] + y_bin_edges[:-1]) / 2

    xmat,ymat = np.meshgrid(x_bin_edges, y_bin_edges, indexing='ij') # for plotting
    xmat_cc,ymat_cc = np.meshgrid(x_bins, y_bins, indexing='ij') # for labeling
    
    return fig


@callback(
    Output('pcolor_plot', 'figure'),
    Input('sc_var1', 'value'),
    Input('sc_var2', 'value')
)
def update_graph_cm(sc_var1,sc_var2):
    x=Data_qc[sc_var1]
    y=Data_qc[sc_var2]
    x_bin_edges=np.linspace(np.nanpercentile(x, 5),np.nanpercentile(x, 95),N_bins)
    y_bin_edges=np.linspace(np.nanpercentile(y, 5),np.nanpercentile(y, 95),N_bins)
    x_bins = (x_bin_edges[1:] + x_bin_edges[:-1]) / 2
    y_bins = (y_bin_edges[1:] + y_bin_edges[:-1]) / 2

    xmat,ymat = np.meshgrid(x_bin_edges, y_bin_edges, indexing='ij') # for plotting
    xmat_cc,ymat_cc = np.meshgrid(x_bins, y_bins, indexing='ij') # for labeling
   
    alldata = pd.DataFrame({
        'xdata': x,
        'ydata': y
    })
    alldata['x_bins'] = pd.cut(alldata['xdata'], x_bin_edges, right=False)
    alldata['y_bins'] = pd.cut(alldata['ydata'], y_bin_edges, right=False)
    counts = alldata.groupby(['x_bins','y_bins']).count()['xdata'].unstack()
    counts.replace(0, np.nan, inplace=True)
    counts = counts.values
    
    fig=go.Figure()
    heatmap = go.Heatmap(x=x_bins, y=y_bins, z=counts, colorscale='hot',colorbar_title='Number of data points')

    fig.add_trace(heatmap)
    fig.update_xaxes(title=sc_var1,showgrid=True,gridcolor='lightgrey',linecolor='black',mirror=True)
    fig.update_yaxes(title=sc_var2,showgrid=True,gridcolor='lightgrey',linecolor='black',mirror=True)
    fig.update_layout(
    paper_bgcolor=f'rgb({0}, {121}, {194})',
    margin=dict(l=0, r=3, t=2, b=0),
    font_family="Cambria",
    font_color="white",)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)