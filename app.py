from flask import Flask, render_template
from scipy import optimize

import pandas as pd
import numpy as np
import plotly.express as px

import os
import json
import plotly
import requests 


# Create Home Page Route
app = Flask(__name__)


def pull_data():
    EIA_API = os.getenv('EIA_API')

    route_daily = 'https://api.eia.gov/v2/electricity/rto/daily-fuel-type-data/data/'
    query_daily = '&frequency=daily&data[0]=value&facets[fueltype][]=SUN&facets[respondent][]=CAL&facets[timezone][]=Pacific&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000'

    r = requests.get(
        route_daily + '?api_key=' + EIA_API + query_daily
    )
    x = r.json()
    data = x["response"]["data"]

    df = pd.read_json(json.dumps(data))
    return df


def sin_plot(x, amp, per, phase, vert, growth):
  return (((amp)*np.sin(per*(phase+x)))+vert) + (growth*x)


def solar_data():
    dated = pull_data().copy()
    dated['period'] = pd.to_datetime(dated['period'])
    dated = dated.loc[::-1].reset_index(drop=True)[['period', 'value']]
    
    last_date = dated['period'].iloc[-1]
    new_index = pd.date_range(start=last_date, periods=180, freq='D')
    new_date_range = pd.DataFrame(new_index, columns=['period'])
    new_date_range['value'] = np.NaN
    extension = pd.concat([dated, new_date_range]).reset_index()
    fig = px.scatter(extension, x="period", y="value")


    params, params_covariance = optimize.curve_fit(sin_plot, 
                                                   dated.index, dated['value'],
                                                   p0=[40000, .0172, 0.25, 90000, 10],
                                                   )

    fig.add_scatter(x=extension['period'], y=sin_plot(extension.index, 
                                                      params[0], params[1], 
                                                      params[2], params[3], 
                                                      params[4]))
    return fig


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
