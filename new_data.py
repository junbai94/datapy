# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:13:57 2017

@author: junbai

New Data computation class
"""
import sys
sys.path.append("C:/Users/j291414/my algorithms")

import datetime
import pandas as pd
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from risk_engine import misc
import new_regression as nr
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 


DATABASE = "C:/Users/j291414/Desktop/market_data.db"


# rewrite get_data function
def get_data(ID, name=None, table='spot_daily', frm=None, to=None, conversion=None, rolling=None, db=DATABASE):
    with sqlite3.connect(db) as conn:
        # select from database
        if table in ['spot_daily', 'spot_index']:
            sql = "SELECT date, close FROM %s WHERE spotID='%s'" % (table, ID)
        elif table in ['fut_daily']:
            sql = "SELECT date, close FROM %s WHERE instID='%s'" % (table, ID)
        elif table in ['_fut_daily']:
            sql = "SELECT datetime, close FROM %s WHERE instID = '%s'" % (table, ID)
        elif table in ['fx_daily']:
            sql = "SELECT date, rate FROM %s WHERE tenor='0W' AND ccy='%s'" % (table, ID)
        else:
            raise ValueError("TABLE NOT SUPPORTED")
        df = pd.read_sql_query(sql, conn)
        df.columns = ['date', 'close']
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
        
        # FX conversion
        if conversion:
            def get_fx(fromCurrency, toCurrency):
                # currency weight
                weights = {'EUR':10, 'USD':5, 'CNY': 1}
                if weights[fromCurrency] >= weights[toCurrency]:
                    ccy = "/".join([fromCurrency, toCurrency])
                    fx = get_data(ccy, 'fx', 'fx_daily')
                else:
                    ccy = "/".join([toCurrency, fromCurrency])
                    fx = get_data(ccy, 'fx', 'fx_daily')
                    fx['fx'] = fx['fx'].apply(lambda x: 1.0/x)
                return fx
            
            fromCurrency, toCurrency = conversion
            fx = get_fx(fromCurrency, toCurrency)
            group = merge_data(df, fx)
            group['close'] = group['close'] * group['fx']
            df = group[['date', 'close']]
            
        # rolling average
        if rolling:
            df['close'] = pd.rolling_mean(df['date'], window=rolling)
        
        # date range selection
        if frm or to:
            df = date_range(df, frm, to)
        
        # rename price column
        if name:
            df = df.rename(columns={'close':name, 'rate':name})
    
    return df
        

#def get_data(ID, table='spot_daily', name=None, frm=None, to=None, currency=None, rolling=None, db=DATABASE):
#    conn = sqlite3.connect(db)
#    if table == 'fx_daily':
#        sql = "select date, rate from fx_daily where tenor='0W' and ccy='{}'".format(ID)
#        temp = pd.read_sql_query(sql, conn)
#    elif table == 'spot_daily':
#        sql = "select date, close from spot_daily where spotID = '{}'".format(ID)
#        temp = pd.read_sql_query(sql, conn)
#    elif table == 'fut_daily':
#	if not 'hc' in ID: 
#            sql = "select date, close from fut_daily where instID = '{}'".format(ID)
#	else:
#	    sql = "select date, close from fut_daily where instID = '{}' and exch='SHFE'".format(ID)
#        temp = pd.read_sql_query(sql, conn)
#    elif table == 'spot_index':
#        sql = "select date, close from spot_index where code = '{}'".format(ID)
#        temp = pd.read_sql_query(sql, conn)
#    elif table == '_fut_daily':
#        sql = "select timestamp, close from _fut_daily where instID = '{}'".format(ID)
#        temp = pd.read_sql_query(sql, conn)
#        temp = temp.rename(columns={'timestamp':'date'})
#    else:
#        raise ValueError('FUNCTION NOT SUPPORT THIS TABLE')
#    
#    temp['date'] = pd.to_datetime(temp['date'], format='%Y-%m-%d %H:%M:%S')
#    
#    if currency:
#        fx = get_data(currency, 'fx_daily')
#        temp = divide_fx(temp, fx)
#        temp = temp[['date', 'result']]
#
#    if rolling:
#	temp.iloc[:,1] = pd.rolling_mean(temp.iloc[:,1], window=rolling)
#
#    if name:
#        temp.columns = ['date', name]
#
#    if frm or to:
#	temp = date_range(temp, frm, to)
#
#    conn.close()
#    return temp


def get_cont_contract(ticker, n, frm, to, rolling_rule='-30b', \
                      freq='d', need_shift=False, name=None, to_USD=False):
    frm = datetime.strptime(frm, "%Y-%m-%d").date()
    to = datetime.strptime(to, "%Y-%m-%d").date()
    temp = misc.nearby(ticker, n, frm, to, rolling_rule, freq, need_shift)
    temp['date'] = pd.to_datetime(temp.index, format='%Y-%m-%d %H:%M:%S')
    temp = temp[['date', 'close', 'contract']]
    
    if to_USD:
        fx = get_data('USD/CNY', 'rate', table='fx_daily')
        temp = temp.merge(fx, on='date')
        temp['result'] = temp['close'].divide(temp['rate'])
        temp = temp[['date', 'result', 'contract']]
        
    if name:
       temp.columns = ['date', name, 'contract']
    
    temp.index = range(len(temp))  
    return temp


def get_date(df):
    """
    Return the column name of datetimes. This function is for normalizing purpose.
    """
    columns = list(df.select_dtypes(include=['datetime']).columns)
    if len(columns) >= 2:
        raise ValueError("More than 1 column of datetime objects")
    if len(columns) == 0:
        raise ValueError("No column is of datetime class")
    return columns[0]


def merge_data(*args):
    temp = args[0]
    for i in range(1, len(args)):
        temp = temp.merge(args[i], on='date')
    temp = temp.dropna()
    return temp

def cut_data(df, *args, **kwargs):
    temp = df.copy()
    output = list()
    
    if 'n' in kwargs.keys():
        n = kwargs['n']
        for i in range(n):
            output.append(temp.iloc[i*len(temp)/n:(i+1)*len(df)/n])
    elif args:
        for i, date in enumerate(args):
            if i == 0:
                output.append(temp[temp['date']<=date])
                continue
            output.append(temp[(temp['date']<=date)&(temp['date']>=args[i-1])])
        output.append(temp[temp['date']>=args[-1]])   
    else:
        n = 2
        for i in range(n):
            output.append(temp.iloc[i*len(temp)/n:(i+1)*len(df)/n])
    
    return output

def date_range(df, frm=None, to=None):
    temp = df
    try:
        if frm:
	    temp = temp[temp['date'] >= frm]
	if to:
	    temp = temp[temp['date'] <= to]
        return temp

    except KeyError:
        raise KeyError("NAME YOUR DATE COLUMN TO BE 'date'")  
        
def monthly_avg(df):
    return df.resample("M", how ='mean', on='date')

def times_fx(df, fx):
    temp = df.merge(fx, on='date')
    if 'result' not in temp.columns:
        temp['result'] = temp.iloc[:,1].multiply(temp.iloc[:,2])
        return temp
    else:
        raise ValueError('Do not choose result as column name')
        
def divide_fx(df, fx):
    temp = df.merge(fx, on='date')
    if 'result' not in temp.columns:
        temp['result'] = temp.iloc[:,1].divide(temp.iloc[:,2])
        return temp
    else:
        raise ValueError('Do not choose result as column name')
        
def plot_data(df_list, enlarge=True, interactive=False):
    if not interactive:
        if enlarge:
            plt.figure(figsize=(10, 5))
        for df in df_list:
            label = df.columns[1]
            plt.plot(df['date'], df[label], label=label)
        plt.xticks(rotation='vertical')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
    else:
        from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
        import plotly.graph_objs as go
        init_notebook_mode(connected=True)
        data = list()
        for df in df_list:
            date_label, price_label = tuple(df.columns)
            trace = {'x': df[date_label], 'y': df[price_label], 'name': price_label}
            data.append(trace)
        iplot(data)


def colored_scatter(dep, indep, df, reg=True):
    points = plt.scatter(df[dep], df[indep], c=df['date'], s=20, cmap='jet')
    cb = plt.colorbar(points)
    cb.ax.set_yticklabels([str(x.date()) for x in df['date'][::len(df)//5]])
    if reg:
        sns.regplot(dep, indep, df, scatter=False, color='.1')


def quick_regression(dep_tpl, indep_tpl):
    if type(dep_tpl) == type(tuple()):
        dep = get_data(dep_tpl[0], dep_tpl[1], name=dep_tpl[2])
        indep = get_data(indep_tpl[0], indep_tpl[1], indep_tpl[2])
	name_dep = dep_tpl[2]
	name_indep = indep_tpl[2]
    else:
	dep = dep_tpl
	indep = indep_tpl
	name_dep = dep.columns[1]
	name_indep = indep.columns[1]
    merged = merge_data(dep, indep)
    reg = nr.Regression(merged, name_dep, [name_indep,])
    reg.run_all()
    return reg

def quick_regression_analysis(dep_tpl, indep_tpl):
    if type(dep_tpl) == type(tuple()):
        dep = get_data(dep_tpl[0], dep_tpl[1], name=dep_tpl[2])
        indep = get_data(indep_tpl[0], indep_tpl[1], indep_tpl[2])
	name_dep = dep_tpl[2]
	name_indep = indep_tpl[2]
    else:
	dep = dep_tpl
	indep = indep_tpl
	name_dep = dep.columns[1]
	name_indep = indep.columns[1]
    merged = merge_data(dep, indep)
    reg = nr.Regression(merged, name_dep, [name_indep,])
    reg.summarize_all()
    return reg


