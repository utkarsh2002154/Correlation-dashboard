#!/usr/bin/env python

# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import statsmodels.tsa.stattools as ts 
import yfinance as yf
import dash
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import dash_bootstrap_components as dbc
import numpy as np
import datetime
from statsmodels.tsa.stattools import coint

# Initialize the app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.Div(children='Correlation Based Trading Opportunities among different Products'),
    html.Hr(),
    dbc.Row([dcc.Dropdown(options=['Energy', 'Meats', 'Grains', 'Index', 'Currency', 'Metals'], value='Energy', id='controls-and-radio-item', style={'width': '50vh', 'height': '4vh'}),
    dcc.Dropdown(options=['All', 'Energy', 'Meats', 'Grains', 'Index', 'Currency', 'Metals'], value='Index', id='controls-and-radio-item2', style={'width': '50vh', 'height': '4vh'})]),
    dcc.Dropdown(options=['5m', '15m','30m', '1H', '1D'], value='1D', id='controls-and-radio-item3',style={'width': '50vh', 'height': '4vh'}),
    dcc.Input(id='number-input', type='number', value=0,style={'width': '8vh', 'height': '2vh'}),
    html.Br(),
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=datetime.datetime(2024,2,1).strftime('%Y-%m-%d'),
        end_date=datetime.datetime(2024,3,1).strftime('%Y-%m-%d'),
        display_format='YYYY-MM-DD'
    ),
    dcc.Graph(figure={}, id='controls-and-graph6', style={'width': '18*1000vh', 'height': '90vh'}),
    dcc.Graph(figure={}, id='controls-and-graph7', style={'width': '18*1000vh', 'height': '90vh'}),
    html.Div(children = "t-stat < -2.89 and p-value < 0.05 for series to be stationary"),
    html.Div(id = 'Cointegration')
])

months = {'F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'}

prod = {('CL=F', 'US_Oil'), ('HO=F','Heating_oil'), ('RB=F', 'Gasoline'), ('NG=F', 'Natural_Gas'), ('BZ=F', 'Brent'), ('LE=F', 'Live_Cattle'), 
        ('GF=F', 'Feeder_Cattle'), ('HE=F', 'Lean_Hogs'), ('KE=F', 'Wheat'), ('ZC=F', 'Corn'), ('ZS=F', 'Soyabean'), 
        ('^GSPC', 'S&P500'), ('^DJI', 'Dow_Jones'), ('NQ=F', 'Nasdaq100'), ('^RUT', 'Russell2000'), ('DX=F', 'USD'), 
        ('6B=F', 'British_Pound'), ('6J=F', 'Yen'), ('6E=F', 'EuroFX'), ('GC=F', 'Gold'), ('SI=F', 'Silver'),('HG=F', 'Copper'), ('PL=F', 'Platinum')}

prod2 = {('CL', 'US_Oil', 'NYM'), ('HO','Heating_oil', 'NYM'), ('RB', 'Gasoline', 'NYM'), ('NG', 'Natural_Gas', 'NYM'), ('BZT', 'Brent', 'NYM'), ('LE', 'Live_Cattle', 'NYM'), 
        ('GF', 'Feeder_Cattle', 'NYM'), ('HE', 'Lean_Hogs', 'NYM'), ('KE', 'Wheat', 'NYM'), ('ZC', 'Corn', 'NYM'), ('ZS', 'Soyabean', 'NYM'), 
        ('ES', 'S&P500', 'CME', 'NYM'), ('^DJI', 'Dow_Jones', 'NYM'), ('NQ=F', 'Nasdaq100', 'NYM'), ('^RUT', 'Russell2000', 'NYM'), ('DX', 'USD', 'NYM'), 
        ('6B', 'British_Pound', 'NYM'), ('6J', 'Yen', 'NYM'), ('6E', 'EuroFX', 'NYM'), ('GC', 'Gold', 'CMX'), ('SI', 'Silver', 'CMX'),('HG', 'Copper', 'CMX'), ('PL', 'Platinum', 'CMX')}

# Add controls to build the interaction
@callback(
    Output('controls-and-graph6', 'figure'),
    Input(component_id='controls-and-radio-item', component_property='value'),
    Input(component_id='controls-and-radio-item2', component_property='value'),
    Input(component_id='controls-and-radio-item3', component_property='value'),
    Input(component_id='number-input', component_property='value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)

def update_graph(value, value2, interval, fil, start_date, end_date):
    df_sel = pd.DataFrame()
    for (i, k) in prod:
        pr = yf.Ticker(i)
        hist = pr.history(start = start_date, end = end_date, interval = interval)
        df_sel[k] = hist['Close']
    df_Energy = df_sel[['US_Oil', 'Heating_oil', 'Gasoline', 'Natural_Gas', 'Brent']]
    df_Meat = df_sel[['Live_Cattle', 'Feeder_Cattle', 'Lean_Hogs']]
    df_Grain = df_sel[['Wheat', 'Corn', 'Soyabean']]
    df_Index = df_sel[['S&P500', 'Nasdaq100', 'Dow_Jones', 'Russell2000']]
    df_Curr = df_sel[['USD', 'British_Pound', 'Yen', 'EuroFX']]
    df_Metal = df_sel[['Gold', 'Silver', 'Copper', 'Platinum']]
    if value2 == "All":
        dfc = df_sel.corr()*100
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Energy" and value2 == "Energy":
        dfc = df_Energy.corr()*100
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(df_Energy.corr()*100, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Energy" and value2 == "Meats":
        df_com = pd.concat([df_Energy, df_Meat], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Meat.columns, columns=df_Energy.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        dfc = dfc[dfc > fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Energy" and value2 == "Grains":
        df_com = pd.concat([df_Energy, df_Grain], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Grain.columns, columns=df_Energy.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Energy" and value2 == "Index":
        df_com = pd.concat([df_Energy, df_Index], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Index.columns, columns=df_Energy.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Energy" and value2 == "Currency":
        df_com = pd.concat([df_Energy, df_Curr], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Curr.columns, columns=df_Energy.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Energy" and value2 == "Metals":
        df_com = pd.concat([df_Energy, df_Metal], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Metal.columns, columns=df_Energy.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Meats" and value2 == "Meats":
        dfc = df_Meat.corr()*100
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Meats" and value2 == "Energy":
        df_com = pd.concat([df_Energy, df_Meat], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Energy.columns, columns=df_Meat.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Meats" and value2 == "Grains":
        df_com = pd.concat([df_Meat, df_Grain], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Grain.columns, columns=df_Meat.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Meats" and value2 == "Index":
        df_com = pd.concat([df_Meat, df_Index], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Index.columns, columns=df_Meat.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Meats" and value2 == "Currency":
        df_com = pd.concat([df_Meat, df_Curr], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Curr.columns, columns=df_Meat.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Meats" and value2 == "Metals":
        df_com = pd.concat([df_Meat, df_Metal], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Metal.columns, columns=df_Meat.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Grains" and value2 == "Grains":
        dfc = df_Grain.corr()*100
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Grains" and value2 == "Meats":
        df_com = pd.concat([df_Grain, df_Meat], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Meat.columns, columns=df_Grain.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Grains" and value2 == "Energy":
        df_com = pd.concat([df_Grain, df_Energy], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Energy.columns, columns=df_Grain.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Grains" and value2 == "Index":
        df_com = pd.concat([df_Grain, df_Index], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Index.columns, columns=df_Grain.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Grains" and value2 == "Currency":
        df_com = pd.concat([df_Grain, df_Curr], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Curr.columns, columns=df_Grain.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Grains" and value2 == "Metals":
        df_com = pd.concat([df_Grain, df_Metal], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Metal.columns, columns=df_Grain.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Index" and value2 == "Index":
        dfc = df_Index.corr()*100
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Index" and value2 == "Meats":
        df_com = pd.concat([df_Energy, df_Meat], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Meat.columns, columns=df_Index.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Index" and value2 == "Grains":
        df_com = pd.concat([df_Energy, df_Grain], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Grain.columns, columns=df_Index.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Index" and value2 == "Energy":
        df_com = pd.concat([df_Index, df_Energy], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Energy.columns, columns=df_Index.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Index" and value2 == "Currency":
        df_com = pd.concat([df_Index, df_Curr], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Curr.columns, columns=df_Index.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Index" and value2 == "Metals":
        df_com = pd.concat([df_Index, df_Metal], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Metal.columns, columns=df_Index.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Currency" and value2 == "Currency":
        dfc = df_Curr.corr()*100
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Currency" and value2 == "Meats":
        df_com = pd.concat([df_Curr, df_Meat], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Meat.columns, columns=df_Curr.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Currency" and value2 == "Grains":
        df_com = pd.concat([df_Curr, df_Grain], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Grain.columns, columns=df_Curr.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Currency" and value2 == "Energy":
        df_com = pd.concat([df_Curr, df_Energy], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Energy.columns, columns=df_Curr.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Currency" and value2 == "Index":
        df_com = pd.concat([df_Curr, df_Index], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Index.columns, columns=df_Curr.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Currency" and value2 == "Metals":
        df_com = pd.concat([df_Curr, df_Metal], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Metal.columns, columns=df_Curr.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Metals" and value2 == "Metals":
        dfc = df_Metal.corr()*100
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Metals" and value2 == "Meats":
        df_com = pd.concat([df_Metal, df_Meat], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Meat.columns, columns=df_Metal.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Metals" and value2 == "Grains":
        df_com = pd.concat([df_Metal, df_Grain], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Grain.columns, columns=df_Metal.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Metals" and value2 == "Energy":
        df_com = pd.concat([df_Metal, df_Energy], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Energy.columns, columns=df_Metal.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    elif value == "Metals" and value2 == "Currency":
        df_com = pd.concat([df_Metal, df_Curr], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Curr.columns, columns=df_Metal.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    else:
        df_com = pd.concat([df_Metal, df_Index], axis=1)
        dfc = df_com.corr()*100
        dfc.drop(index=df_Index.columns, columns=df_Metal.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'yellow','green'], range_color=[-100, 100])
    return fig

@callback(
    Output('controls-and-graph7', 'figure'),
    [Input('controls-and-graph6', 'clickData'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    Input(component_id='controls-and-radio-item3', component_property='value'),
    Input(component_id='number-input', component_property='value')]
)

def update_graph2(clickData, start_date, end_date, interval, fil):
    if clickData is not None and 'points' in clickData:
        p1 = clickData['points'][0]['x']
        p2 = clickData['points'][0]['y']
        
        pn1 = [t[0] for t in prod2 if t[1]==p1]
        pn2 = [t[0] for t in prod2 if t[1]==p2]
        pe1 = [t[2] for t in prod2 if t[1]==p1]
        pe2 = [t[2] for t in prod2 if t[1]==p2]
        
        df_p1 = pd.DataFrame()
        df_p2 = pd.DataFrame()
        
        for i in months:
            tick1 = pn1[0] + i +"24"+ "."+ pe1[0]
            tick2 = pn2[0] + i +"24"+"."+ pe2[0]
            
            try:
                hist1 = yf.Ticker(tick1)
                hist2 = yf.Ticker(tick2)
                pf1 = hist1.history(start = start_date, end = end_date, interval = interval)
                pf2 = hist2.history(start = start_date, end = end_date, interval = interval)
                df_p1[tick1] = pf1['Close']
                df_p2[tick2] = pf2['Close']
            except Exception as e:
                print(f"Error fetching data for {tick1} or {tick2}: {e}")
        df_p1 = df_p1.dropna(axis=1, how='all')
        df_p2 = df_p2.dropna(axis=1, how='all')
        dfx = pd.concat([df_p1, df_p2], axis=1)
        dfc = dfx.corr() * 100
        dfc.drop(index=df_p2.columns, columns=df_p1.columns, inplace=True)
        dfc = dfc[np.abs(dfc) >= fil]
        fig2 = px.imshow(dfc, text_auto=True, aspect="auto", color_continuous_scale=['red', 'white', 'green'], range_color=[-100, 100])
    else:
        fig2 = px.line()
    return fig2

@callback(
    Output('Cointegration', 'children'),
    [Input('controls-and-graph7', 'clickData'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    Input(component_id='controls-and-radio-item3', component_property='value')]
)

def printCoint(clickData, start_date, end_date, interval):
    if clickData is not None and 'points' in clickData:
        p1 = clickData['points'][0]['x']
        p2 = clickData['points'][0]['y']
        
        Ticker1 = p1
        Ticker2 = p2
        
        df_p1 = pd.DataFrame()
        hist1 = yf.Ticker(Ticker1)
        hist2 = yf.Ticker(Ticker2)
        pf1 = hist1.history(start = start_date, end = end_date, interval = interval)
        pf2 = hist2.history(start = start_date, end = end_date, interval = interval)
        df_p1[p1] = pf1['Close']
        df_p1[p2] = pf2['Close']
        df_p1.dropna(inplace=True)
        result = ts.coint(df_p1[p1], df_p1[p2])
        coin = "t-stat: "+str(result[0]) + "       p-value: "+str(result[1]) + "  " + str(Ticker1) + "  "+ str(Ticker2) 
    else:
        coin = " "
    return coin
    
    
# Run the app
if __name__ == '__main__':
    app.run(debug=True)

