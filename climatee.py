

import pandas as pd
import numpy as np
from scipy.stats import norm
import sympy as sym
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from statistics import mean
from statistics import median


## Question 1 

def dplus(A,a_vol, D, T):
    return (np.log(A/D) + (r+(a_vol**2))*T)/(a_vol*np.sqrt(T))
def dminus(A,a_vol, D, T):
    return (np.log(A/D) + (r-(a_vol**2))*T)/(a_vol*np.sqrt(T))
def probDefault(A,a_vol,D, T):
    return norm.cdf(-((np.log(A/D) + (r-(a_vol**2))*T)/(a_vol*np.sqrt(T))))


dfstock = pd.read_excel("climate_transition_risk.xlsx", sheet_name=0)
dfcompany = pd.read_excel("climate_transition_risk.xlsx", sheet_name=1)
dfclimate = pd.read_excel("climate_transition_risk.xlsx", sheet_name=2)
dfstock.columns = ['date', 'price']
dfclimate.columns = ['date', 'scenario', 'price']

r = 0.05
T = 1

yearsReversed = sorted(dfcompany['Year'].tolist())
years = dfcompany['Year'].tolist()
def split_years(dt):
    dt['year'] = dt['date'].dt.year
    return [dt[dt['year'] == y] for y in dt['year'].unique()]

e_vol_1y = []

## For merton formulation we need standard deviation of log returns
for i in range(len(years)):
    mask = (dfstock['date'] > str(years[i])) & (dfstock['date'] <= str(years[i]+1))
    dfstock1year = dfstock.loc[mask].copy()
    
    ## Getting equity volatility from yearly standard deviation of log returns.
    dfstock1year['pctChange'] = dfstock.price.pct_change()    
    equity_vol = dfstock1year["pctChange"].std()
    e_vol_1y.append(equity_vol * np.sqrt(dfstock1year.size))



plt.plot(yearsReversed, e_vol_1y[::-1])
plt.ylabel('Equity Volatility')
plt.title('Yearly Equity Volatility 2012-2021')
plt.xlabel('Year')
plt.tight_layout()
plt.show()


a_vol_1y = []
As = []
Eq = []
Debt = []
def myFunction(z):
    A = z[0]
    vol_A = z[1]

    F = np.empty((2))
    F[0] = A*norm.cdf(dplus(A,vol_A, D, T))-np.exp(-r)*D*norm.cdf(dminus(A,vol_A, D, T))-E
    F[1] = A*vol_A*norm.cdf(dplus(A,vol_A, D, T))-E*e_vol_1y[i]
    return F

for i in range(len(years)):
    mask = (dfstock['date'] > str(years[i])) & (dfstock['date'] <= str(years[i] + 1))
    dfstock1year = dfstock.loc[mask].copy()
    last_price = dfstock1year["price"].iloc[-1]
    last_outst = dfcompany["Shares Outstanding (Millions)"].iloc[i]
    E = last_outst*1000000 * last_price
    D = 0.5*1000000* dfcompany["Long Term Debt (Millions)"].iloc[i] + 1000000* dfcompany["Short Term Debt (Millions)"].iloc[i]
    asset = (E + D)
    zGuess = np.array([asset,e_vol_1y[i]])
    z = fsolve(myFunction,zGuess)
    A = z[0]
    print(z)
    vol_A = z[1]
    As.append(z[0])
    a_vol_1y.append(z[1])
    Eq.append(E)
    Debt.append(D)

plt.plot(yearsReversed, a_vol_1y[::-1])
plt.ylabel('Asset Volatility')
plt.title('Yearly Asset Volatility 2012-2021')
plt.xlabel('Year')
plt.tight_layout()
plt.show()

AsBil = []
for i in As:
    AsBil.append(i/1000000000)

plt.plot(yearsReversed, AsBil[::-1])
plt.ylabel('Assets (Billion $)')
plt.title('Yearly Assets 2012-2021')
plt.xlabel('Year')
plt.tight_layout()
plt.show()


## Adding features to new Dataframe with the newly calculated calues for each year. This makes calculated the relative probability of defaults for each year easy.
dfcompany['assets'] = As
dfcompany['assetVol'] = a_vol_1y
dfcompany['equities'] = Eq
dfcompany['equityVol'] = e_vol_1y
dfcompany['debt'] = Debt
## Calculating probability of default
dfcompany['probDefault'] = norm.cdf(-dminus(dfcompany['assets'], dfcompany['assetVol'],dfcompany['debt'], T))
probDefault = dfcompany['probDefault']
dfcompany['probDefault (%)'] = probDefault * 100
probDefaultP = []
for i in probDefault:
    probDefaultP.insert(0, i * 100)


dfcompany.to_csv('newDf.csv')


plt.plot(yearsReversed, probDefaultP)
plt.yscale("log")
plt.ylabel('Probability of Default (%)')
plt.title('Yearly Historical Probability of Default (%) 2012-2021')
plt.xlabel('Year')
plt.tight_layout()
plt.show()



### Question 2
scope1reset = 112000000 
scope3reset = 650000000


scenarioYears = ['2025', '2030', '2035', '2040', '2045', '2050']
scenarioYearsplotfull = yearsReversed + scenarioYears

# carbonPrice = 517.1212734
def epsilon(scope, carbonprice):
    return (scope * carbonprice)
def scenarioProbDefault(A,a_vol,D, T, carbonprice):
    return norm.cdf(-((np.log(A/D) - (a_vol**2)*T)/(a_vol*np.sqrt(T))))

#### Question 1.2.1

tempScenarioScope1 = []
tempScenarioScope3 = []
scenariosScope1 = []
scenariosScope3 = []
carbonPrices = dfclimate['price']
print(carbonPrices[10])
# scenarios = dfclimate['scenario']
for i in range(3):
    scope1 = scope1reset
    scope3 = scope3reset
    for j in range(6):
        carbonPrice = carbonPrices[i*6 + j]
        scenariosScope1.append(scenarioProbDefault(np.maximum(As[0] - epsilon(scope1, carbonPrice), 0), a_vol_1y[0] , Debt[0], T, carbonPrice))
        scenariosScope3.append(scenarioProbDefault(np.maximum(As[0] - epsilon(scope3 + scope1, carbonPrice), 0), a_vol_1y[0], Debt[0], T, carbonPrice))
        # scenariosScope1.append(np.mean(tempScenarioScope1))
        # scenariosScope3.append(np.mean(tempScenarioScope3))
        # tempScenarioScope1 = []
        # tempScenarioScope3 = []
        # scope1 = scope1 - scope1reset/5

scenariospercentScope1 = []
for i in scenariosScope1:
    scenariospercentScope1.append(i * 100)

scenariospercentScope3 = []
for i in scenariosScope3:
    scenariospercentScope3.append(i * 100)


scenarioNCRScope1 = scenariospercentScope1[0:6]  #Getting the 6 scenarios related to index i=0 (NCR) of for loop --> this is in order of dates 2025 - 2050
scenarioDelayedScope1 = scenariospercentScope1[6:12] # Getting the 6 scenarios related to index i=i (delayed) of for loop 
scenarioNetZeroScope1 = scenariospercentScope1[12:18] # Net Zero


scenarioNCRScope3 = scenariospercentScope3[0:6]  #Getting the 6 scenarios related to index i=0 (NCR) of for loop --> this is in order of dates 2025 - 2050
scenarioDelayedScope3 = scenariospercentScope3[6:12] # Getting the 6 scenarios related to index i=i (delayed) of for loop 
scenarioNetZeroScope3 = scenariospercentScope3[12:18] # Net Zero

## Here we are adding in the 2012-2021 value of PD to plot historical pd + simulated pd for time series
scenarioNCRfullScope1 = probDefaultP + scenarioNCRScope1
scenarioDelayedfullScope1 = probDefaultP + scenarioDelayedScope1
scenarioNetZerofullScope1 = probDefaultP + scenarioNetZeroScope1

scenarioNCRfullScope3 = probDefaultP + scenarioNCRScope3
scenarioDelayedfullScope3 = probDefaultP + scenarioDelayedScope3
scenarioNetZerofullScope3 = probDefaultP + scenarioNetZeroScope3

for i in range(0, len(scenarioYearsplotfull)):
    scenarioYearsplotfull[i] = int(scenarioYearsplotfull[i])
dfresultsscope1 = pd.DataFrame({'dates': scenarioYearsplotfull, 'PD - NCR': scenarioNCRfullScope1, 'PD - Delayed': scenarioDelayedfullScope1, 'PD - Net Zero': scenarioNetZerofullScope1})
dfresultsscope3 = pd.DataFrame({'dates': scenarioYearsplotfull, 'PD - NCR': scenarioNCRfullScope3, 'PD - Delayed': scenarioDelayedfullScope3, 'PD - Net Zero': scenarioNetZerofullScope3})
dfresultsscope1.to_csv('ap.csv')
dfresultsscope3.to_csv('ap3.csv')

# print(dfresults)
plt.plot('dates', 'PD - NCR', data=dfresultsscope1[dfresultsscope1['dates'] < 2025], color='grey', linewidth=2, label = 'Historical PD')
plt.plot('dates', 'PD - NCR', data=dfresultsscope1[dfresultsscope1['dates'] > 2020], color='green', linewidth=2, linestyle = '--')
plt.plot('dates', 'PD - Delayed', data=dfresultsscope1[dfresultsscope1['dates'] > 2020], color='orange', linewidth=2, linestyle = '--')
plt.plot('dates', 'PD - Net Zero', data=dfresultsscope1[dfresultsscope1['dates'] > 2020], color='red', linewidth=2, linestyle = '--')
plt.yscale("log")
plt.title('Probability of Default (%) Forecasts with Scope 1 & 2')
# plt.title('Probability of Default (%) Forecasts with Scope 1 & 2 \n Assuming Linear Emissions')
plt.ylabel('Probability of Default (%)')
plt.xlabel('Year')
plt.legend()
plt.tight_layout()
plt.show()


plt.plot('dates', 'PD - NCR', data=dfresultsscope3[dfresultsscope3['dates'] < 2025], color='grey', linewidth=2, label = 'Historical PD')
plt.plot('dates', 'PD - NCR', data=dfresultsscope3[dfresultsscope3['dates'] > 2020], color='green', linewidth=2, linestyle = '--')
plt.plot('dates', 'PD - Delayed', data=dfresultsscope3[dfresultsscope3['dates'] > 2020], color='orange', linewidth=2, linestyle = '--')
plt.plot('dates', 'PD - Net Zero', data=dfresultsscope3[dfresultsscope3['dates'] > 2020], color='red', linewidth=2, linestyle = '--')
plt.yscale("log")
plt.title('Probability of Default (%) Forecasts with Scope 1, 2 & 3')
# plt.title('Probability of Default (%) Forecasts with Scope 1, 2 & 3 \n Assuming Linear Emissions')
plt.ylabel('Probability of Default (%)')
plt.xlabel('Year')
plt.legend()
plt.tight_layout()
plt.show()