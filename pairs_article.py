
#    written by jim leahy   3/29/20    v 1.1
# this script follows the procedure for finding pairs as outlined at
# https://www.quantopian.com/lectures/introduction-to-pairs-trading,
# used with permission. the code is a combination of a script to get historical
# data from IEX cloud and some code snipets provided in this lecture.
#
# this script uses the IEX Cloud API (https://iexcloud.io/docs/api) as the source of
# historical daily stock prices. The company has free and paid subscriptions, but the
# free subscription is sufficient for pairs trade analysis using the approach outlined
# here. It does require an online registration and account, but no credit card is
# required for the free plan. This plan gives you 50,000 messages per month. A message
# is the basic unit of data measurement, with daily close-only data requiring 2 messages
# per day, per symbol. This means analyzing 5 stocks with 3 months of data  would require
# 620 IEX messages each time you run the script.

# Once python is installed, you will have to also install a few libraries used in
# the script. Not all the libraries used in the script are installed with the language,
# but it's easy to install a library with pip (ex: pip install <library>).  To run the
# script just type "python find_pairs.py". But before you run it you have to change a
# few lines in the script.

# you can change the symbols by editing the symbols array between the ########### lines below
# don't put in large numbers of symbols. 
# this will use historical data from the past3 months.
# ex:        symbols = ['intc', 'nvda', 'mu', 'spy', 'amd', 'xlnx']
#
# The script defaults to 3 months (~62 days) of historical data, but the user can increase
# or decrease this number. However, using less data makes it harder to determine proper
# cointegration and correlation values and using a much longer time-frame can exhibit
# false cointegration since stocks can go into and out of cointegration over time.
# Go to https://iexcloud.io/cloud-login#/register to register for an account. Once you
# have an account you'll be given a public key and a sandbox key. These keys have to
# be entered in this script for the variables labeled "pub_token" and "test_token",
# respectively in the script as shown below. 

# ex:    test_token = "your sandbox key goes here"    # test token
# ex:    pub_token = "your public key goes here"   # public token

# The sandbox token allows you to test running the script with randomly generated
# historical data. The sandbox is used to verify the script setup without using any of
# the alloted messages. To change the script to use the sandbox uncomment the line
# using the sandbox key and add a comment character, "#", before the line in the script
# using the public key as shown below.

#  ex:    url = sandbox_url + sym + plot_range + "?token=" + test_token + close_only 
#  ex:    #   url = pub_url + sym + plot_range + "?token=" + pub_token + close_only 

# the script will plot the graphs for the first cointegrated pair (index starting
# at 0) that it finds. if there are more than one cointegrated pairs found, you can
# change the index from 0 to 1, 2, or whatever other pair you want to plot in the
# line with the variable "first_pair", as shown below.

#  ex:                  first_pair = pairs[0]
# 
# when you're done with the setup, to run the script type "python find_pairs.py"


#####   START OF SCRIPT  #########



import requests # get stuff from the web
import os  #  file system operations
import re  # regular expressions
import pandas as pd # pandas... data analysis library
import numpy as np
import datetime as dt # date and time functions
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import io
import time
import matplotlib.pyplot as plt
from matplotlib import style
import json
from pprint import pprint
from pandas.io.json import json_normalize
from collections import OrderedDict
import seaborn
import statsmodels.tsa.stattools as ts

style.use('ggplot')
style.use('dark_background')

def find_cointegrated_pairs(stocks_frame):
    n = len(stocks_frame.columns)
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            series_a = stocks_frame.iloc[:,i].values   # get array
            series_b = stocks_frame.iloc[:,j].values   # get array
            coint_result = coint(series_a, series_b)
            score = coint_result[0]
            pvalue = coint_result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((stocks_frame.columns[i], stocks_frame.columns[j]))
    return score_matrix, pvalue_matrix, pairs

 # z score function
def zscore(series):
    return (series - series.mean()) / np.std(series)

def find_correlated_pairs(cframe):
    # loop through correlation matrix and flag correlated pairs
    n = len(cframe.columns)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            val = cframe.iloc[i,j]
            if val > .95:
                pairs.append((cframe.columns[i], cframe.index[j]))
             #   print(round(val,5), cframe.columns[i], cframe.index[j])
    return pairs

def parse_json(json_data):
    close_array = []
    date_array = []
    for key in json_data:
        close_array.append(float( key['close']))   # create a list
        date_array.append(key['date'])
    close_df = pd.DataFrame(close_array).astype(float)  # convert list to dataframe
    date_df = pd.DataFrame(date_array)  # convert list to dataframe

    new_df = pd.DataFrame()    # create enpty dataframe
    new_df = pd.DataFrame(data=[date_array,close_array]).T  # concatenate arrays into
                                                            # dataframe
    new_df.columns=['date', 'close']   #  add column names
    new_new_df = new_df.set_index('date')  # make date column the index
    new_new_df.sort_values(by=['date'], inplace=True, ascending=True)
    return new_new_df
 
#####################################################################
#####################################################################
#   put symbols here - be careful not to duplicate symbols.

symbols = ['bac', 'gs', 'c', 'ms', 'xlf', 'spy']
#symbols = ['intc', 'nvda', 'mu', 'spy', 'amd', 'xlnx']

#####################################################################
#####################################################################

test_token = "YOUR TEST TOKEN GOES HERE"    # test token
pub_token = "YOUR PUB TOKEN GOES HERE"   # public token
plot_range = "/chart/3m" ;   # 1m, 3m, 6m, 1y, 2y, 5y
close_only = "&chartCloseOnly=true"   ### limit message cost to 2 per sample
pub_url = "https://cloud.iexapis.com/stable/stock/"
sandbox_url = "https://sandbox.iexapis.com/stable/stock/"

print('getting data ... ')

#   loop through the symbol list getting historical data and create
#   a single dataframe with only the adjusted close prices

n = len(symbols)
for i in range(n):
    sym = symbols[i]
    url = sandbox_url + sym + plot_range + "?token=" + test_token + close_only 
 #   url = pub_url + sym + plot_range + "?token=" + pub_token + close_only 
    print (symbols[i])
    data = requests.get(url)   # get data
    buf = io.StringIO(data.text) # create a buffer
    json_data = json.loads(data.text,object_pairs_hook=OrderedDict)    # parse json data
    # create a dataframe - parse json data
    df = parse_json(json_data)
    if i == 0:
          df_combined = df[['close']].copy().astype(float)
          # rename a column
          df_combined = df_combined.rename(columns={"close":sym})
    else:
        df_close1 = df[['close']].copy().astype(float)
        df_combined['close'] = df_close1
        df_combined = df_combined.rename(columns={"close":sym})

print('\n')
print(df_combined.head(10))
fig = plt.figure("pairs plots", figsize=(11,7))
fig.tight_layout()
col_len = len(df_combined.columns)
print ('dataframe columns = ' + str(col_len))
print ('dataframe length = ' + str(len(df_combined.index)))
scores, pvalues, pairs = find_cointegrated_pairs(df_combined)
plt.subplot(2,2,3)
seaborn.heatmap(pvalues, xticklabels=symbols, yticklabels=symbols,
                cmap='RdYlGn_r' , #  
                mask = (pvalues >= 0.95)
                )
print ("\n")
print ('*******************************************************')
print ('  these are determined to be cointegrated pairs:       ')
print  (pairs)
print ('-------------------------------------------------------')
print ('the first pair is used for z-score and relative perf. plot')
print ('change the pairs[0] index to plot results for a different pair')

if not pairs:
    print ("no cointegrated pairs found")
    quit()
####################################################################
###########   change index to plot z score for different       #####
###########   cointegrated pairs                               #####
####################################################################

first_pair = pairs[0]

####################################################################
####################################################################
#print("first",pairs[0])
#print('fs=',first_pair[0])
#plt.subplot(2,2,3)
plt.title("Cointegrated Pairs")
plt.plot()
plt.subplot(2,2,2)

# do correlation
### dataframe.corr parameters: dataframe.corr(method='',min_periods=1)
### method: {'pearson', 'kendall', 'spearman'} or callable
print("\nCorrelation Matrix")
#print('spearman correlation')
#print(df_combined.corr(method='spearman'))
df_corr = df_combined.corr()
print(df_corr)
print('\n')
print('\ncorrelated pairs: ')
corr_pairs = find_correlated_pairs(df_corr)
print (corr_pairs)

#import seaborn as sns
corr = df_combined.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
seaborn.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, cmap='RdYlGn', mask = mask)
plt.title("Correlation Matrix") 
plt.plot()
plt.rcParams['xtick.labelsize'] = 8 
plt.subplot(2,2,4)
# now plot the difference of the pairs
series_a = df_combined[first_pair[0]]
series_b = df_combined[first_pair[1]]

series_a= sm.add_constant(series_a.values)
results = sm.OLS(series_b, series_a).fit()  # ordinary least squares
series_a = df_combined[first_pair[0]]
ols_df = pd.read_html(results.summary().tables[1].as_html(),header=0,index_col=0)[0]
print('\nOLS dataframe')
print(ols_df)
a=ols_df['coef'].values[1]
b=ols_df['coef'].values[0]
print('\n results of ordinary least squares ')
print('slope =  '+ str(a))  # slope (or multiplier from linear regression) mult series_a by this value
print('intercept =  '+ str(b))  # intercept


series_a = df_combined[first_pair[0]]
series_b = df_combined[first_pair[1]]

ols_result = sm.OLS(series_a, series_b).fit() 

print('\n dickey-fuller test')
print( ts.adfuller(ols_result.resid))
print('\n')

series_a = df_combined[first_pair[0]]
series_b = df_combined[first_pair[1]]
##diff_series = (series_a) - (series_b)
diff_series = ((series_a) * a) - (series_b)  ## mult series_a by ols slope
# plot the z score of the difference
zscore(diff_series).plot()
plt.axhline(zscore(diff_series).mean(), color='brown')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.title("Z-score of difference of "  + first_pair[0] + " minus  " + first_pair[1]) 
plt.ylabel("z-score")
plt.grid('on', linestyle='--', alpha=0.5)

norm_df = df_combined.copy()
print(norm_df.head(10))
norm_df = norm_df.divide(norm_df.iloc[0])
print('\n')
print(norm_df.head(10))
series_a = norm_df[first_pair[0]]
series_b = norm_df[first_pair[1]]
plt.subplot(2,2,1)
plt.grid('on', linestyle='--', alpha=0.5)
series_a.plot()
series_b.plot()
plt.title("relative performance of "  + first_pair[0] + " and  " + first_pair[1]) 
plt.legend()
#####   save graph to a png file
#plt.savefig("coint_corr.png")   # save graph to png file
plt.show()

#### plot the difference of the normalized series

diff_series = (series_a)  - (series_b)
# plot just the difference
plt.figure("diff plot", figsize=(11,7))
diff_series.plot(color='yellow')
plt.axhline(0.0, color='green', linestyle='--')
plt.title("difference of normalized "  + first_pair[0] + " minus  " + first_pair[1]) 
plt.ylabel("diff")
plt.grid('on', linestyle='--', alpha=0.5)
plt.show()

