import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import techie


# BBands plot
def plot_BBANDS(data, column, n):
    """
    Plot Bollinger Bands Plotly interactive plot in Jupyter notebook
    :param data: DataFrame
    :param column: Column name in dataframe for price.
    :param n: Bollinger Band period
    """
    df = data.copy()
    df.columns = [x.upper() for x in list(df.columns)]
    df = df.rename(columns={column.upper(): 'Close', 'DATE': 'date', 'TIMESTAMP': 'date'})
    df = techie.BBANDS(df, n)

    # plotting
    upper = go.Scatter(dict(
        x = df['date'],
        y = df['BollingerU_' + str(n)],
        line = {'color': 'red', 'dash': 'dash'},
        name = 'upper'
    ))
    lower = go.Scatter(dict(
        x=df['date'],
        y=df['BollingerL_' + str(n)],
        line={'color': 'red', 'dash': 'dash'},
        name='lower'
    ))
    ma = go.Scatter(dict(
        x = df['date'],
        y = df['MA_' + str(n)],
        line = {'color': 'orange'},
        name = 'MA'
    ))
    close = go.Scatter(dict(
        x = df['date'],
        y = df['Close'],
        name = column,
        line = {'color': 'blue'},
    ))
    iplot([upper, lower, ma, close])