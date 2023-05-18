from flask import Flask, render_template, request
from scipy import optimize
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

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


# def pull_data():
#    EIA_API = os.getenv('EIA_API')
#
#    route_daily = 'https://api.eia.gov/v2/electricity/rto/daily-fuel-type-data/data/'
#    query_daily = '&frequency=daily&data[0]=y&facets[fueltype][]=SUN&facets[respondent][]=CAL&facets[timezone][]=Pacific&sort[0][column]=ds&sort[0][direction]=desc&offset=0&length=5000'
#
#    r = requests.get(
#        route_daily + '?api_key=' + EIA_API + query_daily
#    )
#    x = r.json()
#    data = x["response"]["data"]
#
#    df = pd.read_json(json.dumps(data))
#    return df


def load_data(energy_source):
    connection = sqlite3.connect('EnergySources.db')
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


def render_data(energy_source):
    dated = load_data(energy_source).copy()
    dated['ds'] = pd.to_datetime(dated['ds'])
    dated = dated[['ds', 'y']]
    
    last_date = dated['ds'].iloc[-1]
    new_index = pd.date_range(start=last_date, periods=180, freq='D')
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
    if energy_source in ["NUC", "OIL", "WAT"]:
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
        )
    )
    return fig


@app.route('/california_dashboard', methods=['GET', 'POST'])
def california_dashboard():
    if request.method == 'POST':
        energy_type = request.form['energy_source']
        print(energy_type)
        fig = render_data(energy_type)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('calidash.html', graphJSON=graphJSON)
    else:
        fig = render_data("SUN")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('calidash.html', graphJSON=graphJSON)


@app.route('/')
def bar_with_plotly():
    return "<a href=\"california_dashboard\">California Energy Production Dashboard</a>"


if __name__ == '__main__':
    app.run(debug=True)
