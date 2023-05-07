import numpy as np
import pandas as pd
from prophet import Prophet
from datetime import datetime as dt
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta
import warnings

def set_holidays(dt_transform, dt_holidays, intervalType):
    opts = ["day", "week", "month"]
    
    if intervalType not in opts:
        raise ValueError("Pass a valid 'intervalType'. Any of: ", f"{opts}")
    elif intervalType == "day":
        holidays = dt_holidays
    elif intervalType == "week":
        weekStartInput = dt.weekday(dt_transform["ds"][1])+1
        if weekStartInput not in range(1, 7):
            raise ValueError("Week start has to be Monday or Sunday")
        holidays['ds'] = holidays['ds'].apply(lambda x: floor_date(x, 'week', week_start=1))
        holidays = holidays[['ds','holiday','country','year']]\
            .groupby(['ds','country','year'])\
                .agg({'holiday': lambda x: ', '.join(x),'year':'count'})\
                    .rename(columns={'year':'n'}).reset_index()
    elif intervalType == "month":
        if all(dt_transform["ds"].dt.day()) != 1:
            raise ValueError("Monthly data should have first day of month as datestampe, e.g.'2020-01-01'")
        holidays = dt_holidays\
            .assign(ds=pd.cut(dt_holidays['ds'], bins=intervalType))\
                .loc[:, ['ds', 'holiday', 'country', 'year']] \
                    .groupby(['ds', 'country', 'year']) \
                        .agg(holiday = ('holiday', lambda x: ', '.join(x)), n=('ds', 'count'))\
                            .reset_index()
    return holidays

def prophet_decomp(
    self
    ,dt_transform
    ):
    holidays = set_holidays(
        dt_transform
        ,self.dt_holidays
        ,self.intervalType)
    recurrence = dt_transform[["ds","dep_var"]]
    recurrence.rename(columns={"dep_var":"y"},inplace=True)

    reg_cols = self.context_vars + self.paid_media_spends
    dt_regressors = pd.concat([
        recurrence
        ,dt_transform[reg_cols]
    ],axis=1)

    prophet_params = np.nan
    use_trend = False
    use_holiday = False
    use_season = False
    use_weekday = False
    use_monthly = False

    if len(self.holiday)==0:
        warnings.warn("No holiday vars", category=UserWarning)
    elif "trend" in self.holiday:
        use_trend = True
    elif "holiday" in self.holiday:
        prophet_params = holidays[holidays["country"] == self.country]
        use_holiday = True
    elif "season" in self.holiday:
        use_season = True
    elif "weekday" in self.holiday:
        use_weekday = True
    elif "monthly" in self.holiday:
        use_monthly = True
    
    modelRecurrence = Prophet(
        holidays=prophet_params,
        yearly_seasonality=use_season,
        weekly_seasonality=use_weekday,
        daily_seasonality=False
    )
    
    if use_monthly:
        modelRecurrence = modelRecurrence.add_seasonality(
            name = "monthly",
            period = 30.5, 
            fourier_order = 5
        )
    
    if self.factor_vars is not None & len(self.factor_vars)>0:
        dt_ohe_factors = pd.get_dummies(self.all_df[self.factor_vars])
        ohe_names = dt_ohe.columns
        for addreg in ohe_names:
            modelRecurrence = modelRecurrence.add_regressor(addreg)
        dt_ohe = pd.concat([
            self.all_df.drop(self.factor_vars,axis=1)
            ,dt_ohe_factors]
            ,axis=1)
        modelRecurrence.fit(dt_ohe)
        dt_forecastRegressor = modelRecurrence.predict(dt_ohe)
        remove_cols = [s for s in dt_forecastRegressor.columns if "_lower"|"upper" in s]
        forecastRecurrence = dt_forecastRegressor.drop(remove_cols,axis=1)
        for aggreg in self.factor_vars:
            oheRegNames = [col for col in forecastRecurrence.columns if re.match(f"^{aggreg}.*", col)]
            get_reg = forecastRecurrence[oheRegNames].sum(axis=1)
            scaler = StandardScaler(with_mean=False)
            dt_transform[aggreg] = scaler.fit_transform(get_reg)
    else:
        if self.dayInterval == 1:
            warnings.warn("Currently, there's a known issue with prophet that may crash this use case.",
            "\n Read more here: https://github.com/facebookexperimental/Robyn/issues/472")
            modelRecurrence.fit(dt_regressors)
            forecastRecurrence = modelRecurrence.predict(dt_regressors)
    
    seq_along = lambda x: range(1, len(x)+1)
    these = seq_along([item for sublist in recurrence[:,0] for item in sublist])
    if use_trend:
        dt_transform["trend"] = forecastRecurrence["trend"][these]
    if use_season:
        dt_transform["season"] = forecastRecurrence["yearly"][these]
    if use_monthly:
        dt_transform["monthly"] = forecastRecurrence["monthly"][these]
    if use_weekday:
        dt_transform["weekday"] = forecastRecurrence["weekday"][these]
    if use_holiday:
        dt_transform["holiday"] = forecastRecurrence["holiday"][these]
    
    self.dt_transform = dt_transform
    
def floor_date(
    x 
    ,unit
    ,week_start=0
    ):
    if unit == "week":
        # datetime型に変換
        x = dt.strptime(x, "%Y-%m-%d")
        # week_startが0の場合、月曜始まりに設定
        if week_start == 0:
            week_start = 1
        # xが含まれる週の月曜日の日付を計算
        start = x - dt.timedelta(days=x.weekday() - week_start)
        # xが含まれる週の日曜日の日付を計算
        end = start + dt.timedelta(days=6)
        # 日付を文字列に変換して返す
        return start.strftime("%Y-%m-%d")
    else:
        raise NotImplementedError('Only week unit is supported')