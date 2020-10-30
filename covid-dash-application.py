# -*- coding: utf-8 -*-
from flask import Flask
import os
import dash_auth
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_table
from dash_table.Format import Format, Scheme, Group, Sign, Symbol
import plotly.graph_objs as go
import datetime
from datetime import timedelta
from datetime import datetime as dt
import gunicorn
import base64

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.title = 'COVID-19'

app.config['suppress_callback_exceptions']=True

time1 = datetime.datetime.now()
print('TIME1:', time1)

dat1_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
dat1_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
dat1_recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
dat2 = pd.read_csv('https://raw.githubusercontent.com/RRighart/covid-19/master/countries.csv')
dat2['Country/Region'] = dat2['Country/Region'].astype('object')


def preproc(dat, countries):
    """
    dat=input dataset
    countries=countries where input data is provinces
    """
    dat['Country/Region'] = dat['Country/Region'].astype('object')
    dat['Country/Region'] = dat['Country/Region'].replace({'Korea, South':'South Korea', 'Taiwan*':'Taiwan'})
    temp = pd.DataFrame(dat[dat['Country/Region'].isin([countries])].groupby('Country/Region').sum().reset_index()) #sum Lat,Long is useless
    temp.insert(loc=0, column='Province/State', value= np.nan)
    dat = dat.append(temp) # Add summed area for country split in provinces
    return(dat)

dat1_confirmed = preproc(dat1_confirmed, 'China')
dat1_recovered = preproc(dat1_recovered, 'China')
dat1_deaths = preproc(dat1_deaths, 'China')

##############
# LAYOUT
##############

# dropdown menus

preselected_countries= ['Germany', 'France', 'Switzerland', 'Netherlands', 'Belgium'] #list(dat2['Country/Region'].unique()) 

unique_countries = list(dat2['Country/Region'].unique()) # all countries present in second dataset

country_list=np.array([{'label': i, 'value': i} for i in sorted(dat2['Country/Region'].unique())])

baseline_list = np.array([{'label': i, 'value': i} for i in ['Date', 'Days']])

time_list = np.array([{'label': i, 'value': i} for i in ['Weeks', 'Days']])

yscale_list = np.array([{'label': i, 'value': i} for i in ['Linear', 'Logarithmic']])

entry_list = np.array([{'label': i, 'value': i} for i in ['Confirmed', 'Recovered', 'Fatality']])

pop_list = np.array([{'label': i, 'value': i} for i in ['Absolute number', 'Per 1 million inhabitants', 'Per 10000 km2 area']])

days_list = np.array([{'label': i, 'value': i} for i in ['Current', 'Change since 1 day', 'Change since 3 days', 'Change since 7 days']])

start_date = dt(2020, 1, 1)
end_date = dt(2020, 12, 31)



navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(height='34px')), # src=image_filename, 
                    dbc.Col(dbc.NavbarBrand("COVID-19 DATA | A VISUALIZATION PER COUNTRY, POPULATION SIZE AND AREA", className="ml-3")),
                ],
                align="center",
                no_gutters=True,
            ),
        ),
    ],
    color="grey", #grey, darkgreen, brown, darkred
    dark=True,
)


body = html.Div([ #4 
    dcc.Tabs([ #3

        # TAB CHARTS
        dbc.Tab([ #2
            dbc.Container([ #1
                dbc.Container([
                    dbc.Row([
                        #dbc.Col([ # CHARTS
                            dbc.Col([
                                dcc.Graph(id='graph1'),
                                ], width=10),


                            dbc.Col([
                                dbc.Col([
                                    html.Div('Select date', style={'color': 'grey', 'fontSize': 16, 'font-weight':'bold'}),
                                    dcc.DatePickerRange(id='my-date-picker', minimum_nights=0, min_date_allowed=dt(2020, 1, 1), max_date_allowed=dt(2020, 12, 31), initial_visible_month=dt(2020, 3, 20), start_date=dt(2020, 3, 10), end_date=dt(2020, 12, 31), disabled=False, style={'width': '100%'}, with_full_screen_portal=True)
   	                            ]),

                                dbc.Col([
                                    html.Div('Baseline', style={'color': 'grey', 'fontSize': 16, 'font-weight':'bold'}),
                                    dcc.RadioItems(id='baseline-radioitems', options=baseline_list, value='Date'),
                                    ]),

                                dbc.Col([
                                    html.Div('Scale', style={'color': 'grey', 'fontSize': 16, 'font-weight':'bold'}),
                                    dcc.RadioItems(id='yscale-radioitems', options=yscale_list, value='Linear'),
                                    ]),

                                ], width=2), #row
                            ]),

                    ], fluid=True, className="content-box"), # container

                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.Div('Confirmed infections / Fatality', style={'color': 'grey', 'fontSize': 16, 'font-weight':'bold'}),
                            dcc.Dropdown(id='cases-dropdown', options=entry_list, value='Confirmed', clearable=False),
                            ], width=3),

                        dbc.Col([
                            html.Div('Country', style={'color': 'grey', 'fontSize': 16, 'font-weight':'bold'}),
                            dcc.Dropdown(id='country-dropdown', options=country_list, value=preselected_countries, multi=True, clearable=False),
                            ], width=3),

                        dbc.Col([
                            html.Div('Population size / Area', style={'color': 'grey', 'fontSize': 16, 'font-weight':'bold'}),
                            dcc.Dropdown(id='pop-dropdown', options=pop_list, value='Absolute number', clearable=False),
                            ], width=3),

                        ]), #row
                    ], fluid=True, className="content-box"), #container

                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='graph2'),
                            ], width=10),
                            
                        dbc.Col([
                            dbc.Col([  
                                html.Div('Number of bars', style={'color': 'grey', 'fontSize': 16, 'font-weight':'bold'}),
                                dcc.Input(id='nbars', type="number", value=16),
                                ]),
                            dbc.Col([  
                                html.Div('Value displayed', style={'color': 'grey', 'fontSize': 16, 'font-weight':'bold'}),
                                dcc.Dropdown(id='days_change', options=days_list, value='Current', clearable=False),
                                ]),

                            ], width=2), #col
                        ]), #row
                    ], fluid=True, className="content-box"), #container

                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='graph3'),
                            ], width=10),
                            
                        dbc.Col([
                            dbc.Col([  
                                html.Div('Country', style={'color': 'grey', 'fontSize': 16, 'font-weight':'bold'}),
                                dcc.Dropdown(id='country-dropdown2', options=country_list, value='France', multi=False, clearable=False),
                                #], width=3),
                                ]),

                            dbc.Col([
                                html.Div('Time scale', style={'color': 'grey', 'fontSize': 16, 'font-weight':'bold'}),
                                dcc.RadioItems(id='time-radioitems', options=time_list, value='Weeks'),
                                ]),

                            #dbc.Col([  
                            #    html.Div('Value displayed', style={'color': 'grey', 'fontSize': 16, 'font-weight':'bold'}),
                            #    dcc.Dropdown(id='days_change', options=days_list, value='Current', clearable=False),
                            #    ]),

                            ], width=2), #col
                        ]), #row
                    ], fluid=True, className="content-box"), #container


                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.Div('Suggestions, questions, or bug reports:'),
                            html.Div('rrighart@googlemail.com', style={'color': 'blue'}), 
                            dcc.Link('https://www.rrighart.com', href='https://www.rrighart.com'),
                            html.Div('Covid-19 data are kindly provided by:', style={'color': 'black', 'fontSize': 14, 'font-weight':'normal'}),
                            dcc.Link('https://github.com/CSSEGISandData/COVID-19/', href='https://github.com/CSSEGISandData/COVID-19/'),
                            html.Div('Population and area data were obtained at Wikipedia and Worldometers', style={'color': 'black', 'fontSize': 14, 'font-weight':'normal'}),
                            ], width=8), #col
                        ]), #row
                    ], fluid=True, className="content-box"), #container


                #], fluid=True, className="content-box"), 
            ], fluid=True, className="content-box"), # 1. container

        ], label="", label_style={"color": "#00AEF9"}), # 2. dbc.Tab, label_style={"color": "#0032f9"}

    ], className="nav-fill") # 3. dbc.Tabs

], style={"min-width":"320px"}) # 4.

#    return slayout

app.layout = html.Div([navbar, body])

##############
# CALLBACKS
##############


# functions

def absolute_change(dat, days):
    """
    dat=dataset, for ex dat.
    days=number of days to look back, for ex 3.
    """
    t1 = dat.loc[(dat['Country/Region'].isin(unique_countries)) & (dat['date']==dat['date'].max()-timedelta(days)), ['Country/Region', 'new_metric']].sort_values(by='new_metric', ascending=False).reset_index(drop=True)
    t1 = t1.rename({'new_metric':'t1'}, axis=1)
    t2 = dat.loc[(dat['Country/Region'].isin(unique_countries)) & (dat['date']==dat['date'].max()), ['Country/Region', 'new_metric']].sort_values(by='new_metric', ascending=False).reset_index(drop=True)
    t2 = t2.rename({'new_metric':'t2'}, axis=1)
    ndat = pd.merge(t1, t2, on=['Country/Region'])
    ndat['difference'] = ndat['t2'].subtract(ndat['t1'], fill_value=None)
    ndat = round(ndat.sort_values(by='difference', ascending=False),2)
    #print('NDAT:', ndat)
    return(ndat)

# TAB1

def linegraph(t):
    @app.callback(
        Output('graph1', 'figure'),
        [Input('cases-dropdown', 'value'),
        Input('country-dropdown', 'value'),
        Input('my-date-picker', 'start_date'),
        Input('my-date-picker', 'end_date'),
        Input('baseline-radioitems', 'value'),
        Input('yscale-radioitems', 'value'),
        Input('pop-dropdown', 'value')]) # ['Absolute number of cases', 'Per 1 million inhabitants', 'Per 10000 km2 area']
     
    def update_graph(filename, select_country, start_date, end_date, base, yscale, pop):
        if filename == 'Fatality': fn=dat1_deaths
        elif filename == 'Confirmed': fn=dat1_confirmed
        elif filename == 'Recovered': fn=dat1_recovered
        else: print('Wrong input for filename')

        dat1 = fn[(fn['Country/Region'].isin(unique_countries)) & (fn['Province/State'].isin([np.nan*len(unique_countries)]))]
        dat1 = pd.melt(dat1.drop(['Lat', 'Long', 'Province/State'], axis=1), id_vars = ['Country/Region'])
        dat1 = dat1.rename({'variable':'date', 'value':'metric'}, axis=1)
        dat1['date']=pd.to_datetime(dat1['date'])
        print('DAT1:', dat1)
        dat = pd.merge(dat1, dat2, on=['Country/Region'])
        dat = dat.sort_values(by='date', ascending=True)

        if pop == 'Per 1 million inhabitants': 
            dat['F_inhabitants'] = dat['inhabitants'].div(1000000)
            dat['new_metric'] = dat['metric'].div(dat['F_inhabitants'])
        elif pop == 'Per 10000 km2 area':
            dat['F_area'] = dat['area'].div(10000)
            dat['new_metric'] = dat['metric'].div(dat['F_area'])
        elif pop == 'Absolute number':
            dat['new_metric'] = dat['metric'].copy()
        else: print('Normal values are')

        countries_ranked_lastdate = list(dat.loc[(dat['date']==dat['date'].max()), ['Country/Region', 'new_metric']].sort_values(by='new_metric', ascending=False)['Country/Region'])
       
        rsc = [x for x in countries_ranked_lastdate if x in select_country]

        if base=='Days': # baseline_list = np.array([{'label': i, 'value': i} for i in ['Date', 'Days from start']])
            dat = dat[(dat['metric'] != 0)]
            dat = dat.sort_values(['Country/Region', 'date'], ascending=[True, True])
            dat['date_rank'] =dat.groupby(['Country/Region']).apply(lambda x: x['date'].astype('category').cat.codes+1).values
            print('dat:', dat[dat['Country/Region']=='France'])
            xaxis = 'date_rank'
            start_date=0
            end_date=100
        elif base=='Date':
            xaxis= 'date'
        else: print('No baseline shift')

        if yscale=='Linear':
            yaxis_scale = 'linear'
        elif yscale=='Logarithmic':
            yaxis_scale = 'log'
        else: print('No yscale value given')

        colors = [x.replace('US','blue').replace('Germany','blueviolet').replace('France','grey').replace('Belgium', 'darkkhaki').replace('Netherlands', 'orange').replace('Switzerland', 'chocolate').replace('Italy', 'darkblue').replace('Spain', 'purple').replace('United Kingdom', 'darkslateblue').replace('Portugal', 'powderblue').replace('Austria', 'midnightblue').replace('Ireland', 'greenyellow').replace('China', 'green').replace('South Korea', 'darkred').replace('Iran', 'magenta').replace('Luxembourg', 'lightskyblue').replace('Poland', 'lightsteelblue').replace('Turkey', 'slategray').replace('Greece', 'palegreen').replace('Brazil', 'limegreen').replace('Mexico', 'maroon').replace('Sweden', 'gainsboro').replace('Denmark', 'silver').replace('Finland', 'lightblue').replace('Norway', 'indianred').replace('Taiwan', 'navy').replace('India', 'sienna').replace('Vietnam', 'cyan').replace('Indonesia', 'moccasin') for x in rsc]
        print('rsc:', rsc)
        print('colors:', colors)


        trace = []

        for i,j in zip(list(rsc), colors):
            dfn = dat[(dat['Country/Region'] == i) & (dat[xaxis] >= start_date) & (dat[xaxis] <= end_date)]
            trace.append(go.Scatter(x=dfn[xaxis], y=round(dfn['new_metric'],2), mode='lines+markers', line=dict(width=3, color=j), name=str(i) )) # name=str(i), text=dfn['metric'], textposition='auto', 
        
        layout = dict(
            title=dict(text=filename+' cases over time', font=dict(size=20, color='grey', weight='bold')),
            xaxis=dict(title=base, showgrid=True, fixedrange=True),
            yaxis=dict(title='Cases ('+pop+')', showgrid=True, title_standoff = 50, fixedrange=True, type=yaxis_scale), # 'yaxis': {'type': 'log', 'linear'}}
            margin=dict(l=40, b=40, t=40, r=0, pad=-10),
            height=400,
            #width=1600,
            autosize=True,
            showlegend=True,
            hovermode='closest'
            )
   
        fig = dict(data=trace, layout=layout)
        return fig
        
linegraph("graph1")

def bargraph(t):
    @app.callback(
        Output('graph2', 'figure'),
        [Input('cases-dropdown', 'value'),
        Input('nbars', 'value'),
        Input('days_change', 'value'),
        Input('pop-dropdown', 'value')])

    def update_graph(filename, nbars, days_change, pop):
        if filename == 'Fatality': fn=dat1_deaths
        elif filename == 'Confirmed': fn=dat1_confirmed
        elif filename == 'Recovered': fn=dat1_recovered
        else: print('Wrong input for filename')
        dat1 = fn[(fn['Country/Region'].isin(unique_countries)) & (fn['Province/State'].isin([np.nan*len(unique_countries)]))]
        dat1 = pd.melt(dat1.drop(['Lat', 'Long', 'Province/State'], axis=1), id_vars = ['Country/Region'])
        dat1 = dat1.rename({'variable':'date', 'value':'metric'}, axis=1)
        dat1['date']=pd.to_datetime(dat1['date'])
        dat = pd.merge(dat1, dat2, on=['Country/Region'])

        print('POP', pop)
        if pop == 'Per 1 million inhabitants': 
            dat['F_inhabitants'] = dat['inhabitants'].div(1000000)
            dat['new_metric'] = round(dat['metric'].div(dat['F_inhabitants']),2)
        elif pop == 'Per 10000 km2 area':
            dat['F_area'] = dat['area'].div(10000)
            dat['new_metric'] = round(dat['metric'].div(dat['F_area']),2)
        elif pop == 'Absolute number':
            dat['new_metric'] = dat['metric'].copy()
        else: print('Normal values are taken')

        trace=[]

        if days_change=='Current':
            ndat = dat.loc[(dat['Country/Region'].isin(unique_countries)) & (dat['date']==dat['date'].max()), ['Country/Region', 'new_metric']].sort_values(by='new_metric', ascending=False).head(nbars)
            trace = go.Bar(x=ndat['Country/Region'], y=ndat['new_metric'], text=ndat['new_metric'], textposition='auto')
        elif days_change=='Change since 1 day':
            ndat = absolute_change(dat, 1)
            trace = go.Bar(x=ndat['Country/Region'], y=ndat['difference'], text=ndat['difference'], textposition='auto')
        elif days_change=='Change since 3 days':
            ndat = absolute_change(dat, 3)
            trace = go.Bar(x=ndat['Country/Region'], y=ndat['difference'], text=ndat['difference'], textposition='auto')
        elif days_change=='Change since 7 days':
            ndat = absolute_change(dat, 7)
            trace = go.Bar(x=ndat['Country/Region'], y=ndat['difference'], text=ndat['difference'], textposition='auto')
        else: print('Error')

        layout = dict(
            title=dict(text=filename+' cases per country', font=dict(size=20, color='grey', style='bold')),
            xaxis=dict(title='', showgrid=True, showticklabels=True, fixedrange=True),
            yaxis=dict(title=days_change, showgrid=True, fixedrange=True),
            font=dict(size=10),
            margin=dict(l=80, b=100, t=40, r=0),
            height=500,
            #width=1600,
            autosize=True,
            showlegend=False,
            hovermode='closest'
            )
   
        fig = dict(data=[trace], layout=layout)
        return fig
        
bargraph("graph2")


def newcasesgraph(t):
    @app.callback(
        Output('graph3', 'figure'),
        [Input('cases-dropdown', 'value'),
        Input('country-dropdown2', 'value'),
        Input('time-radioitems', 'value')])

    def update_graph(filename, countries, time):
        if filename == 'Fatality': fn=dat1_deaths
        elif filename == 'Confirmed': fn=dat1_confirmed
        elif filename == 'Recovered': fn=dat1_recovered
        else: print('Wrong input for filename')
        dat1 = fn[(fn['Country/Region'].isin(unique_countries)) & (fn['Province/State'].isin([np.nan*len(unique_countries)]))]
        dat1 = pd.melt(dat1.drop(['Lat', 'Long', 'Province/State'], axis=1), id_vars = ['Country/Region'])
        dat1 = dat1.rename({'variable':'date', 'value':'metric'}, axis=1)
        dat1['date']=pd.to_datetime(dat1['date'])
        dat = pd.merge(dat1, dat2, on=['Country/Region'])

        dat = dat.sort_values(by='date', ascending=True)
        dat = dat[dat['Country/Region'] == countries]
        dat['week'] = pd.DatetimeIndex(dat['date']).week
        dat['added_cases'] = dat['metric'].diff()
        dat['added_cases'].fillna('0', inplace=True)
        dat['added_cases'] = dat['added_cases'].astype('int')
        weektable = dat.groupby(['week']).aggregate('added_cases').sum()
        weektable = weektable.reset_index(drop=False)

        trace=[]

        if time == 'Weeks':        
            trace = go.Bar(x=weektable['week'], y=weektable['added_cases'], marker_color='indianred')
        elif time == 'Days':
            trace = go.Bar(x=dat['date'], y=dat['added_cases'], marker_color='orange')

        layout = dict(
            title=dict(text='Evolution of new cases for '+countries, font=dict(size=20, color='grey', style='bold')),
            xaxis=dict(title=time, showgrid=True, showticklabels=True, fixedrange=True),
            yaxis=dict(title='Number of new cases', showgrid=True, fixedrange=True),
            font=dict(size=10),
            margin=dict(l=80, b=100, t=40, r=0),
            height=500,
            #width=1600,
            autosize=True,
            showlegend=False,
            hovermode='closest'
            )
   
        fig = dict(data=[trace], layout=layout)
        return fig
        
newcasesgraph("graph3")


time2 = datetime.datetime.now()
print('TIME2:', time2)
print('DIFF:', (time2-time1))

if __name__ == '__main__':
    app.run_server(debug=True)






