### this creates a python package for COMM475 - Investment policy
# reference: https://www.tidy-finance.org/python/fama-macbeth-regressions.html #

### The following functions are included
# 0. clean_test_asset_returns(test_assets,risk_factors)
# 1. fama_macbeth_timeseries_estimate_beta(test_assets,risk_factors,returns_monthly)
# 2. fama_macbeth_crosssection_estimate_premium(test_assets,risk_factors,beta_monthly)
# 3. fama_macbeth_crosssection_premium_stat(risk_premiums)

# Additionally, a consolidated function for fama macbeth regression consolidates the above 4 steps
# 4. fama_macbeth_regression(test_assets,risk_factors)


# load package for dataframe operation
import pandas as pd
import numpy as np
from datetime import datetime

from itertools import product
from joblib import Parallel, delayed, cpu_count
from statsmodels.regression.rolling import RollingOLS
import statsmodels.formula.api as smf

# define parameters

# window size is the number of months (rolling OLS) used to estimate beta
# minimuim observation is defined to filter the test asset ids filter the test asset ids for beta estimation (to make sure there is enough data sample for each test id included to estimate beta

window_size = 60 #number of months to estimate beta
min_obs = 48 #minimum number of months for teach test asset id to be included in the estimation


############### step 0. Clean test asset monthly returns ###########
def clean_test_asset_returns(test_assets,risk_factors):
    
    """_summary_
    this function clean test assets returns and align the test assets data with risk factor data

    Returns:
      clearned test returns data (merged with risk factors)

    data format:
    test_id is the name of each test asset

    """

    # test_ids that have data records > window_size are included as valid_test_ids
    valid_test_ids = (test_assets
    .groupby("test_id")["test_id"]
    .count()
    .reset_index(name="counts")
    .query(f"counts > {window_size}+1")
    )

  # define the test period (first_date, last_date) for each valid test id
    test_id_information = (test_assets
    .merge(valid_test_ids, how="inner", on="test_id")
    .groupby(["test_id"])
    .aggregate(first_date=("date", "min"),
                last_date=("date", "max"))
    .reset_index()
    )

    unique_test_id = test_assets["test_id"].unique()
    unique_month = risk_factors["date"].unique()

    all_combinations = pd.DataFrame(
    product(unique_test_id, unique_month),
    columns=["test_id", "date"]
    )

  # format dates to be datetime format
    test_id_information.loc[:, 'first_date'] = pd.to_datetime(test_id_information['first_date'], errors='coerce')
    test_id_information.loc[:, 'last_date'] = pd.to_datetime(test_id_information['last_date'], errors='coerce')
    all_combinations.loc[:, 'date'] = pd.to_datetime(all_combinations['date'], errors='coerce')
    all_combinations['date'] = all_combinations['date'].astype('datetime64[ns]')
    test_assets.loc[:, 'date'] = pd.to_datetime(test_assets['date'], errors='coerce')

  # add a date variable in 'yyyymm' format, this facilitates the merge process because the date sometimes are displayed differently: e.g., the start of a month, or the end of a month
    test_id_information.loc[:, 'first_date_yyyymm'] = test_id_information['first_date'].dt.strftime('%Y%m')
    test_id_information.loc[:, 'last_date_yyyymm'] = test_id_information['last_date'].dt.strftime('%Y%m')
    all_combinations.loc[:, 'date_yyyymm'] = all_combinations['date'].dt.strftime('%Y%m')
    test_assets.loc[:,'date_yyyymm'] = test_assets['date'].dt.strftime('%Y%m')

  # filter test assets monthly returns to only include test_ids that have enough data records
    returns_monthly = (all_combinations
    .merge(test_assets.get(["test_id", "date_yyyymm", "ret_excess"]),
            how="left", on=["test_id", "date_yyyymm"])
    .merge(test_id_information, how="left", on="test_id")
    .query("(date_yyyymm >= first_date_yyyymm) & (date_yyyymm <= last_date_yyyymm)")
    .drop(columns=["first_date_yyyymm", "last_date_yyyymm"])
    )

  # format risk factors
    risk_factors['date'] = pd.to_datetime(risk_factors['date'], errors='coerce')
    risk_factors.loc[:,'date_yyyymm'] = risk_factors['date'].dt.strftime('%Y%m')

  # merge monthly returns with risk factors
    returns_monthly = (returns_monthly.drop(columns =['date'])
                                    .merge(risk_factors, how = "left", on = "date_yyyymm"))

    return returns_monthly


####################### 1. Fama-MacBeth time-series analysis of returns to estimate beta ###############
def fama_macbeth_timeseries_estimate_beta(returns_monthly,risk_factors):
    
    """
    summary: this function estimates beta on a rolling basis for each risk factors
    input: risk factors
    return: rolling beta for each risk factor
    """

    # define the keys for risk factors, this will be used to name the beta estimation for each risk factor
    risk_factors_keys = list(risk_factors.keys())

    # remove 'date' and 'date_yyyymm' from the list using a loop to only keep risk factor names to faciliate OLS regression, naming convention, etc.
    for key in ['date', 'date_yyyymm']:
        if key in risk_factors_keys:
            risk_factors_keys.remove(key)
         
            
    # group returns_monthly by test id, this prepares for the Fama-MacBeth regression by test_id
    permno_groups = (returns_monthly.groupby("test_id", group_keys=False)
        )

    def roll_beta_estimation_for_joblib(test_id, group):

        """Calculate rolling beta estimation using joblib.
          test_id: test asset id (e.g., permno for single stock; industry name for Fama-French industry portfolio)
          group: the timeseries returns for each test_id
        """

        # Drop rows with NaN values in the 'returns' column and risk factors columns
        group = group.dropna(subset=['ret_excess'])
        group = group.dropna(subset=risk_factors_keys)

        # If the group is empty after dropping NaNs or if the group does not have enough data records, skip it;
        if group.empty or len(group) <= window_size:
            return None

        group = group.sort_values(by="date")

        beta_values = (RollingOLS.from_formula(
          formula = f"ret_excess ~ {' + '.join(risk_factors_keys)}",
          data=group,
          window=window_size,
          min_nobs=min_obs,
          missing="drop"
        )
        .fit()
        .params
        )

        result = pd.DataFrame(beta_values)
        result.columns = ['Intercept'] + risk_factors_keys
        result["date"] = group["date"].values
        result["test_id"] = test_id

        return result

    # estimate betas based on rolling stock returns.
    n_cores = cpu_count()-1

    beta_monthly = (
    pd.concat(Parallel(n_jobs=n_cores)
      (delayed(roll_beta_estimation_for_joblib)(name, group)
      for name, group in permno_groups)
    )
    .dropna()
    )

    # tag "beta_" in the beta estimation if the column name is in risk_factors_keys
    beta_monthly = beta_monthly.rename(columns=lambda col: 'beta_' + col if col in risk_factors_keys else col)
    # format the date to prepare for the merge with test asset returns
    beta_monthly.loc[:,'date_yyyymm'] = beta_monthly['date'].dt.strftime('%Y%m')

    return beta_monthly

        
###################### 2. Fama-MacBeth cross-sectional analysis of returns ###########################
def fama_macbeth_crosssection_estimate_premium(test_assets, risk_factors, beta_monthly):
    
    """
    input: risk_factors
    return: estimated risk premium for each risk factor on a rolling basis
    """
    
    ### 2.1 Merge test assets with beta estimation ###
    data_fama_macbeth = pd.DataFrame()

    # Calculate time-varying risk premium
    data_fama_macbeth = (test_assets.drop(columns=['date'])
      .merge(beta_monthly,
              how="left",
              on=["date_yyyymm","test_id"])
      .sort_values(["date_yyyymm", "test_id"])
    )

    data_fama_macbeth_lagged = (data_fama_macbeth
    .assign(date=lambda x: x["date"].dt.to_period("M").dt.to_timestamp()-pd.DateOffset(months=1))
    .get(["test_id", "date", "ret_excess"])
    .rename(columns={"ret_excess": "ret_excess_lead"})
    )

    data_fama_macbeth_lagged.loc[:,"date"] = pd.to_datetime(data_fama_macbeth_lagged["date"], errors='coerce')
    data_fama_macbeth_lagged.loc[:,"date_yyyymm"] = data_fama_macbeth_lagged["date"].dt.strftime('%Y%m')

    data_fama_macbeth = (data_fama_macbeth.drop(columns=['date'])
    .merge(data_fama_macbeth_lagged, how="left", on=["test_id", "date_yyyymm"])
    )

    ### 2.2 Estimate risk premium by date ###
    # calculate risk premium
    risk_factors_beta_keys = list(beta_monthly.keys())

    # Remove 'date' and 'date_yyyymm' from the list using a loop
    for key in ['date','date_yyyymm','test_id','Intercept']:
        if key in risk_factors_beta_keys:
            risk_factors_beta_keys.remove(key)

    risk_factors_beta_keys.append('ret_excess_lead')
    data_fama_macbeth.dropna(subset=risk_factors_beta_keys, inplace=True)
    risk_factors_beta_keys.remove('ret_excess_lead')

    # initiate a dataframe for risk premiums
    risk_premiums = pd.DataFrame()

    risk_premiums = (data_fama_macbeth
    .groupby("date_yyyymm")
    .apply(lambda x: smf.ols(
      formula=f"ret_excess_lead ~ {' + '.join(risk_factors_beta_keys)}",
      data=x
    ).fit().params if len(x) > 0 else pd.Series())
    .reset_index()
    )

    # rename risk premium factor key names (remove tag "beta_")
    risk_premiums = risk_premiums.rename(columns=lambda col: col.replace('beta_', '') if col in risk_factors_beta_keys else col)
    
    return risk_premiums
    
###################### 3. Aggregate risk premium timeseries and calculate t-statistics for each risk factor #######################
def fama_macbeth_crosssection_premium_stat(risk_premiums):
    
    # calculate price of risk, normal t-statisics
    price_of_risk = (risk_premiums
                        .melt(id_vars="date_yyyymm", var_name="factor", value_name="estimate")
                        .groupby("factor")["estimate"]
                        .apply(lambda x: pd.Series({
                            "risk_premium": 100*x.mean(),
                            "t_statistic": x.mean()/x.std()*np.sqrt(len(x))
                          })
                        )
                        .reset_index()
                        .pivot(index="factor", columns="level_1", values="estimate")
                        .reset_index()
                      )

    # calculate price of risk - Newey West t-statistics
    price_of_risk_newey_west = (risk_premiums
    .melt(id_vars="date_yyyymm", var_name="factor", value_name="estimate")
    .groupby("factor")
    .apply(lambda x: (
    x["estimate"].mean()/
      smf.ols("estimate ~ 1", x)
      .fit(cov_type="HAC", cov_kwds={"maxlags": 6}).bse
    )
    )
    .reset_index()
    .rename(columns={"Intercept": "t_statistic_newey_west"})
    )

    # merge normal t-statistics and Newey-West t-statistics
    price_of_risk=(price_of_risk
    .merge(price_of_risk_newey_west, on="factor")
    .round(3)
    )

    # readjust the index order
    factor_order = ["Intercept"] + [factor for factor in price_of_risk["factor"] if factor != "Intercept"]
    price_of_risk = price_of_risk.set_index('factor')
    price_of_risk = price_of_risk.reindex(factor_order).reset_index()
    
    return price_of_risk
    
###################### 4. Aggregate risk premium timeseries and calculate t-statistics for each risk factor #######################
def fama_macbeth_regression(test_assets,risk_factors):

    """
    Steps:
    0. Standarize test asset monthly returns to only include test ids with enough sample periods
    1. Fama-MacBeth time-series analysis of returns to estimate beta: for each test_id, run the timeseries regression on a rolling basis to estimate beta
    2. Fama-MacBeth cross-sectional analysis of returns: at each date, calculate risk premium timeseries
    3. Aggregate risk premium timeseries and calculate t-statistics for each risk factor

    Args:
       risk_factors: a timeseries data formated as dataframe. It should include the following columns:
       - date
       - risk factors, e.g. market risk for CAPM model; market, size, and value factor for Fama-French 3 factor model;

       test_assets: a panel data formated as dataframe. It should include the following columns:
       - date
       - test asset returns (minus risk free rate): named as'ret_excess"

    Returns:
        A dataframe include columns: (factor, risk_premium, t_values, t_values_newey_west)
            - factor: the name of risk factor
            - risk_premium: Time-series means of risk premiums (magnitudes of risk premiums).
            - t_values: Time-series t-statistics of risk premiums (significance of risk premiums).
            - t_values_newey_west (t statistics adjusted with Newey West): Time-series t-statistics of risk premiums adjusted with Newey West standard error.
    """

    # format risk factors
    risk_factors['date'] = pd.to_datetime(risk_factors['date'], errors='coerce')
    risk_factors.loc[:,'date_yyyymm'] = risk_factors['date'].dt.strftime('%Y%m')

    risk_factors_keys = list(risk_factors.keys())

    # remove 'date' and 'date_yyyymm' from the list using a loop to only keep risk factor names to faciliate OLS regression, naming convention, etc.
    for key in ['date', 'date_yyyymm']:
        if key in risk_factors_keys:
            risk_factors_keys.remove(key)

    ############### 0. Standarize test asset monthly returns ###########
    # The following section standarize monthly returns to only include stock returns that have enough sample periods.

    # test_ids that have data records > window_size are included as valid_test_ids
    valid_test_ids = (test_assets
      .groupby("test_id")["test_id"]
      .count()
      .reset_index(name="counts")
      .query(f"counts > {window_size}+1")
    )

    # define the test period (first_date, last_date) for each valid test id
    test_id_information = (test_assets
      .merge(valid_test_ids, how="inner", on="test_id")
      .groupby(["test_id"])
      .aggregate(first_date=("date", "min"),
                 last_date=("date", "max"))
      .reset_index()
    )

    unique_test_id = test_assets["test_id"].unique()
    unique_month = risk_factors["date"].unique()

    all_combinations = pd.DataFrame(
      product(unique_test_id, unique_month),
      columns=["test_id", "date"]
    )

    # format dates to be datetime format
    test_id_information.loc[:, 'first_date'] = pd.to_datetime(test_id_information['first_date'], errors='coerce')
    test_id_information.loc[:, 'last_date'] = pd.to_datetime(test_id_information['last_date'], errors='coerce')
    all_combinations.loc[:, 'date'] = pd.to_datetime(all_combinations['date'], errors='coerce')
    test_assets.loc[:, 'date'] = pd.to_datetime(test_assets['date'], errors='coerce')

    # add a date variable in 'yyyymm' format, this facilitates the merge process because the date sometimes are displayed differently: e.g., the start of a month, or the end of a month
    test_id_information.loc[:, 'first_date_yyyymm'] = test_id_information['first_date'].dt.strftime('%Y%m')
    test_id_information.loc[:, 'last_date_yyyymm'] = test_id_information['last_date'].dt.strftime('%Y%m')
    all_combinations.loc[:, 'date_yyyymm'] = all_combinations['date'].dt.strftime('%Y%m')
    all_combinations['date'] = all_combinations['date'].astype('datetime64[ns]')
    test_assets.loc[:,'date_yyyymm'] = test_assets['date'].dt.strftime('%Y%m')

    # filter test assets monthly returns to only include test_ids that have enough data records
    returns_monthly = (all_combinations
      .merge(test_assets.get(["test_id", "date_yyyymm", "ret_excess"]),
             how="left", on=["test_id", "date_yyyymm"])
      .merge(test_id_information, how="left", on="test_id")
      .query("(date_yyyymm >= first_date_yyyymm) & (date_yyyymm <= last_date_yyyymm)")
      .drop(columns=["first_date_yyyymm", "last_date_yyyymm"])
    )

    # merge monthly returns with risk factors
    returns_monthly = (returns_monthly.drop(columns =['date'])
                                      .merge(risk_factors, how = "left", on = "date_yyyymm"))

    # group returns_monthly by test id, this prepares for the Fama-MacBeth regression by test_id
    permno_groups = (returns_monthly
          .groupby("test_id", group_keys=False)
        )
    ########################################################################

    ####################### 1. Fama-MacBeth time-series analysis of returns to estimate beta ###############
    # this function estimates beta on a rolling basis
    def roll_beta_estimation_for_joblib(test_id, group):

        """Calculate rolling beta estimation using joblib.
           test_id: test asset id (e.g., permno for sinle stock; industry name for Fama-French industry portfolio)
           group: the timeseries returns for each test_id
        """

        # Drop rows with NaN values in the 'returns' column and risk factors columns
        group = group.dropna(subset=['ret_excess'])
        group = group.dropna(subset=risk_factors_keys)

        # If the group is empty after dropping NaNs or if the group does not have enough data records, skip it;
        if group.empty or len(group) <= window_size:
            return None

        group = group.sort_values(by="date")

        beta_values = (RollingOLS.from_formula(
            formula = f"ret_excess ~ {' + '.join(risk_factors_keys)}",
            data=group,
            window=window_size,
            min_nobs=min_obs,
            missing="drop"
          )
          .fit()
          .params
        )

        result = pd.DataFrame(beta_values)
        result.columns = ['Intercept'] + risk_factors_keys
        result["date"] = group["date"].values
        result["test_id"] = test_id

        return result

    # estimate betas based on rolling stock returns.
    n_cores = cpu_count()-1

    beta_monthly = (
      pd.concat(Parallel(n_jobs=n_cores)
        (delayed(roll_beta_estimation_for_joblib)(name, group)
        for name, group in permno_groups)
      )
      .dropna()
    )

    # tag "beta_" in the beta estimation if the column name is in risk_factors_keys
    beta_monthly = beta_monthly.rename(columns=lambda col: 'beta_' + col if col in risk_factors_keys else col)
    # format the date to prepare for the merge with test asset returns
    beta_monthly.loc[:,'date_yyyymm'] = beta_monthly['date'].dt.strftime('%Y%m')
    ########################################################################


    ###################### 2. Fama-MacBeth cross-sectional analysis of returns ###########################

    ### 2.1 Merge test assets with beta estimation ###
    data_fama_macbeth = pd.DataFrame()

    # Calculate time-varying risk premium
    data_fama_macbeth = (test_assets.drop(columns=['date'])
          .merge(beta_monthly,
                 how="left",
                 on=["date_yyyymm","test_id"])
          .sort_values(["date_yyyymm", "test_id"])
        )

    data_fama_macbeth_lagged = (data_fama_macbeth
      .assign(date=lambda x: x["date"].dt.to_period("M").dt.to_timestamp()-pd.DateOffset(months=1))
      .get(["test_id", "date", "ret_excess"])
      .rename(columns={"ret_excess": "ret_excess_lead"})
    )

    data_fama_macbeth_lagged.loc[:,"date"] = pd.to_datetime(data_fama_macbeth_lagged["date"], errors='coerce')
    data_fama_macbeth_lagged.loc[:,"date_yyyymm"] = data_fama_macbeth_lagged["date"].dt.strftime('%Y%m')

    data_fama_macbeth = (data_fama_macbeth.drop(columns=['date'])
      .merge(data_fama_macbeth_lagged, how="left", on=["test_id", "date_yyyymm"])
    )

    ### 2.2 Estimate risk premium by date ###
    # calculate risk premium
    risk_factors_beta_keys = list(beta_monthly.keys())

    # Remove 'date' and 'date_yyyymm' from the list using a loop
    for key in ['date','date_yyyymm','test_id','Intercept']:
        if key in risk_factors_beta_keys:
            risk_factors_beta_keys.remove(key)

    risk_factors_beta_keys.append('ret_excess_lead')
    data_fama_macbeth.dropna(subset=risk_factors_beta_keys, inplace=True)
    risk_factors_beta_keys.remove('ret_excess_lead')

    # initiate a dataframe for risk premiums
    risk_premiums = pd.DataFrame()

    risk_premiums = (data_fama_macbeth
      .groupby("date_yyyymm")
      .apply(lambda x: smf.ols(
          formula=f"ret_excess_lead ~ {' + '.join(risk_factors_beta_keys)}",
          data=x
        ).fit().params if len(x) > 0 else pd.Series())
      .reset_index()
    )

    # rename risk premium factor key names (remove tag "beta_")
    risk_premiums = risk_premiums.rename(columns=lambda col: col.replace('beta_', '') if col in risk_factors_beta_keys else col)
    ########################################################################

    ###################### 3. Aggregate risk premium timeseries and calculate t-statistics for each risk factor #######################
    # calculate price of risk, normal t-statisics
    price_of_risk = (risk_premiums
                          .melt(id_vars="date_yyyymm", var_name="factor", value_name="estimate")
                          .groupby("factor")["estimate"]
                          .apply(lambda x: pd.Series({
                              "risk_premium": 100*x.mean(),
                              "t_statistic": x.mean()/x.std()*np.sqrt(len(x))
                            })
                          )
                          .reset_index()
                          .pivot(index="factor", columns="level_1", values="estimate")
                          .reset_index()
                        )

    # calculate price of risk - Newey West t-statistics
    price_of_risk_newey_west = (risk_premiums
    .melt(id_vars="date_yyyymm", var_name="factor", value_name="estimate")
    .groupby("factor")
    .apply(lambda x: (
      x["estimate"].mean()/
        smf.ols("estimate ~ 1", x)
        .fit(cov_type="HAC", cov_kwds={"maxlags": 6}).bse
    )
    )
    .reset_index()
    .rename(columns={"Intercept": "t_statistic_newey_west"})
    )

    # merge normal t-statistics and Newey-West t-statistics
    price_of_risk=(price_of_risk
      .merge(price_of_risk_newey_west, on="factor")
      .round(3)
    )

    # readjust the index order
    factor_order = ["Intercept"] + [factor for factor in price_of_risk["factor"] if factor != "Intercept"]
    price_of_risk = price_of_risk.set_index('factor')
    price_of_risk = price_of_risk.reindex(factor_order).reset_index()

    return price_of_risk