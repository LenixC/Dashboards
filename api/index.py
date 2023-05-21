from flask import Flask, render_template, request
from scipy import optimize
from prophet import Prophet
from datetime import datetime, date, timedelta

import pandas as pd
import numpy as np
import plotly.express as px

import os
import json
import plotly
import requests 
import sqlite3


# Create Home Page Route
app = Flask(__name__)


energy_names = {'COL': 'Coal',
                'NG' : 'Natural Gas',
                'NUC': 'Nuclear',
                'OIL': 'Oil',
                'SUN': 'Solar',
                'WAT': 'Hydroelectric',
                'WND': 'Wind'}


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def load_data(energy_source):
    connection = sqlite3.connect('api/EnergySources.db')
    query = """select  
                   period, {} 
                from  
                    EnergySources 
                order by 
                    period asc""".format(energy_source)
    df = pd.read_sql_query(query, connection)
    df = df.rename(columns={'period': 'ds', energy_source: 'y'})
    connection.close()
    
    return df


def pull_if_needed():
    EIA_API = os.environ.get('EIA_API')
    connection = sqlite3.connect('api/EnergySources.db')
    query = """select  
                   period 
                from  
                    EnergySources 
                order by 
                    period desc
                limit
                    1"""
    df = pd.read_sql_query(query, connection)
    yesterday = date.today() - timedelta(days=1)
    last_pull = datetime.strptime(df['period'].iloc[0], '%Y-%m-%d').date()
    next_pull = last_pull + timedelta(days=1)

    if not last_pull == yesterday:
        energy_sources = ['COL', 'NG', 'NUC', 'OIL', 'SUN', 'WAT', 'WND']
        new_sources = pd.DataFrame(columns=['period'])

        for source in energy_sources:
            route_daily_source = 'https://api.eia.gov/v2/electricity/rto/daily-fuel-type-data/data/'
            query_daily_source = '&frequency=daily&data[0]=value&facets[fueltype][]={}&facets[respondent][]=CAL&facets[timezone][]=Pacific&start={}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'

            r_source = requests.get(
                route_daily_source + '?api_key=' + EIA_API + query_daily_source.format(source, next_pull)
            )

            x_source = r_source.json()
            df_source = pd.read_json(json.dumps(x_source["response"]["data"]))[['period', 'value']]
            df_source = df_source.rename(columns={'value': source})

            new_sources = pd.merge(new_sources, df_source, on='period', how="outer")
        new_sources[energy_sources] = new_sources[energy_sources].astype(int)
        new_sources['period'] = new_sources['period'].astype(str)
        new_sources.to_sql('EnergySources', connection, if_exists='append', index=False)
        connection.commit()
    connection.close()


def get_todays_energy():
    EIA_API = os.environ.get('EIA_API')
    today = date.today()
    route_today = 'https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/'
    query_today = '&frequency=hourly&data[0]=value&facets[respondent][]=CAL&start={}T00&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'

    r_source = requests.get(
        route_today + '?api_key=' + EIA_API + query_today.format(today)
    )

    x_source = r_source.json()
    df_today = pd.read_json(json.dumps(x_source["response"]["data"]))[['fueltype', 'value']]
  
    fig = px.pie(df_today, values='value', names='fueltype',
                 width=150, height=150)
    fig.update_layout(margin=dict(t=0,
                                  b=0,
                                  l=0,
                                  r=0,),
                      showlegend=False,
                      paper_bgcolor='rgba(0, 0, 0, 0)')
    fig.update_traces(textinfo='none')
    
    return df_today['value'].sum(), fig


def sin_plot(x, amp, per, phase, vert, growth):
  return (((amp)*np.sin(per*(phase+x)))+vert) + (growth*x)


def harmonic_sine(x, b_1, b_2, b_3, b_4, period, phase, vert, amp, growth):
  sin1 = b_1*np.sin(period*(phase+x))
  sin2 = b_2*np.sin(2*period*(phase+x))
  sin3 = b_3*np.sin(3*period*(phase+x))
  sin4 = b_4*np.sin(4*period*(phase+x))
  return amp*(sin1 + sin2 + sin3 + sin4) + (growth*x) + vert


def add_prophet(fig, dated, extension):
    m = Prophet()
    m.fit(dated)
    prophet_forecast = m.predict(extension)
    fig.add_scatter(x=extension['ds'], y=prophet_forecast['yhat'],
                    name="Prophet Forecast",
                    line=dict(width=1)
                    )
    return fig


def add_sine(fig, dated, extension):
    params, params_covariance = optimize.curve_fit(sin_plot, 
                                                   dated.index, dated['y'],
                                                   p0=[40000, .0172, 0.25, 90000, 10],
                                                   )

    fig.add_scatter(x=extension['ds'], y=sin_plot(extension.index, 
                                                      params[0], params[1], 
                                                      params[2], params[3], 
                                                      params[4]),
                    name="Sine Forecast",
                    line=dict(width=4)
                    )
    return fig


def add_harmonic_sine(fig, dated, extension):
    params, params_covariance = optimize.curve_fit(harmonic_sine, 
                                                   dated.index, dated['y'],
                                                   p0=[3, 1, 1, 1, 
                                                       0.0172, 0, 0,
                                                       0, 0],
                                                   )

    fig.add_scatter(x=extension['ds'], y=harmonic_sine(extension.index, 
                                                       params[0], params[1], 
                                                       params[2], params[3], 
                                                       params[4], params[5],
                                                       params[6], params[7],
                                                       params[8]),
                    name="Stacked Sine Forecast",
                    line=dict(width=4)
                    )

    return fig


def render_data(energy_source, prediction_days):
    dated = load_data(energy_source).copy()
    dated['ds'] = pd.to_datetime(dated['ds'])
    dated = dated[['ds', 'y']]
    
    last_date = dated['ds'].iloc[-1]
    new_index = pd.date_range(start=last_date, periods=prediction_days, freq='D')
    new_date_range = pd.DataFrame(new_index, columns=['ds'])
    new_date_range['y'] = np.NaN
    extension = pd.concat([dated, new_date_range]).reset_index()
    extension = extension.drop(columns=['index'])
    fig = px.scatter()

    fig.add_scatter(x=extension['ds'], y=extension['y'], mode="markers",
                    marker=dict(size=3),
                    name="Ground Truth"
                    )

    fig = add_prophet(fig, dated, extension)
    if energy_source in ["NG", "COL"]:
        fig = add_harmonic_sine(fig, dated, extension)
    elif energy_source in ["NUC", "OIL", "WAT"]:
        pass
    else:
        fig = add_sine(fig, dated, extension)

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type="date",
            showgrid=False,
        ),
        yaxis=dict(
            showgrid=False,
        ),
        margin=dict(t=0,
                    b=0,
                    l=0,
                    r=0,
                    ),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        legend=dict(yanchor="top", y=1.25, xanchor="left", x=0.75,
                    bgcolor='rgba(255, 255, 255, .6)')
    )
    return fig


@app.route('/california_dashboard', methods=['GET', 'POST'])
def california_dashboard():
    pull_if_needed()
    energy_today, energy_pie = get_todays_energy()
    energy_today = human_format(energy_today)
    pieJSON = json.dumps(energy_pie, cls=plotly.utils.PlotlyJSONEncoder)
    context = {'graphJSON': None,
               'pieJSON': pieJSON,
               'energy_type': "Solar",
               'energy_today': energy_today}
    fig = None;
    graphJson=None;
    if request.method == 'POST':
        energy_type = request.form['energy_source']
        prediction = int(request.form['prediction'])
        fig = render_data(energy_type, prediction)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        context['graphJSON'] = graphJSON
        context['energy_type'] = energy_names.get(energy_type)
    else:
        fig = render_data("SUN", 180)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        context['graphJSON'] = graphJSON
        context['energy_type'] = energy_names.get("SUN")

    return render_template('calidash.html', context=context)


@app.route('/')
def bar_with_plotly():
    return "<a href=\"california_dashboard\">California Energy Production Dashboard</a>"


if __name__ == '__main__':
    app.run(debug=True)
