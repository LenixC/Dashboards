from flask import Flask, render_template
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


def solar_data():
    m = Prophet()
    #try:
    dated = load_data("SUN").copy()
    dated['ds'] = pd.to_datetime(dated['ds'])
    dated = dated[['ds', 'y']]
    
    last_date = dated['ds'].iloc[-1]
    new_index = pd.date_range(start=last_date, periods=180, freq='D')
    new_date_range = pd.DataFrame(new_index, columns=['ds'])
    new_date_range['y'] = np.NaN
    extension = pd.concat([dated, new_date_range]).reset_index()
    extension = extension.drop(columns=['index'])
    fig = px.scatter(extension, x="ds", y="y")

    print(dated)
    print(extension)

    params, params_covariance = optimize.curve_fit(sin_plot, 
                                                   dated.index, dated['y'],
                                                   p0=[40000, .0172, 0.25, 90000, 10],
                                                   )

    fig.add_scatter(x=extension['ds'], y=sin_plot(extension.index, 
                                                      params[0], params[1], 
                                                      params[2], params[3], 
                                                      params[4]))

    return fig
    #except:
    #    return "OOPS"


@app.route('/cali')
def california_dashboard():
    fig = solar_data()
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('calidash.html', graphJSON=graphJSON)


@app.route('/')
def bar_with_plotly():
    return "<a href=\"cali\">California Energy Production Dashboard</a>"


if __name__ == '__main__':
    app.run(debug=True)
