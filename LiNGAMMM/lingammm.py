import pandas as pd
import numpy as np
import warnings
import re
import checks as ck
import prophet_decomp as pd
from datetime import datetime as dt
import multiprocessing
import refresh as ref
import hyper_params as hp
import nevergrad as ng
import math
from tqdm import tqdm
import time
from scipy.stats import uniform
import lingam_model as lm
from joblib import Parallel, delayed
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import glmnet_python
from glmnet import ElasticNet

class LiNGAMMM:
    def __init__(self):
        pass

    def mmm_inputs(
        self
        ,dt_input = None
        ,dt_holidays = None
        ,date_var = None
        ,dep_var = None
        ,dep_var_type = None
        ,prophet_vars = None
        ,prophet_country = None
        ,prophet_signs = None
        ,context_vars = None
        ,context_signs = None
        ,paid_media_spends = None
        ,paid_media_vars = None
        ,paid_media_signs = None
        ,organic_vars = None
        ,organic_signs = None
        ,factor_vars = None
        ,window_start = None
        ,window_end = None
        ,train_size = [0.5, 0.8]
        ):
        self.dt_input = dt_input
        self.dt_holidays = dt_holidays
        self.dep_var = dep_var
        self.dep_var_type = dep_var_type
        self.prophet_vars = prophet_vars
        self.context_vars = context_vars
        self.paid_media_spends = paid_media_spends
        if self.paid_media_vars is None:
            self.paid_media_vars = self.paid_media_spends
        else:
            self.paid_media_vars = paid_media_vars
        self.organic_vars = organic_vars
        self.factor_vars = factor_vars
        self.country = prophet_country
        self.train_size = train_size

        ck.check_datevar(date_var)

        if self.dt_holidays is None or self.prophet_vars is None:
            self.dt_holidays = self.prophet_vars = self.prophet_country = prophet_signs = np.nan
        
        ck.check_prophet(prophet_country,prophet_signs)
        
        ck.check_context(context_signs)
        
        ck.check_paidmedia(paid_media_signs,self.paid_media_spends)
        
        self.exposure_vars = [paid_media_var for paid_media_var in self.paid_media_vars if paid_media_var != self.paid_media_spends]

        ck.check_organicvars(organic_signs)
        
        self.all_media = self.paid_media_spends+self.organic_vars
        self.all_ind_vars = self.prophet_vars.lower()+self.context_vars+self.all_media

        ck.check_windows(window_start, window_end)

        self.unused_vars = [s for s in self.dt_input.columns not in list(self.dep_var, date_var, self.context_vars, self.paid_media_vars, self.paid_media_spends, self.organic_vars)]
    
    def init_adstocks(
        self
        ,hyp_params_dict
        ,adstock = ["geometric","weibull_cdf","weibull_pdf"]
        ):

        if len(adstock)>1:
            raise ValueError("Specify only one adstock funtion among geometric/weibull_cdf/weibull_pdf")
        elif len(hyp_params_dict.keys())!=len(self.paid_media_spends):
            raise IndexError("The number of variables in hyp_params_dict are different from paid_media_spends")
        else:
            self.adstock = adstock
            self.hyper_params = hyp_params_dict
    
    def robyn_engineering(
        self
        ,quiet = False
        ):
        if not quiet:
            print(">> Running feature engineering...")
        dt_input = self.dt_input.drop(self.unused_vars,axis=1)
        dt_inputRollWind = dt_input.loc[self.rollingWindowStartWhich:self.rollingWindowEndWhich]
        
        dt_transform = dt_input
        dt_transform.rename(columns={self.date_var:"ds",self.dep_var:"dep_var"},inplace=True)
        dt_transform = dt_transform.sort_values('ds')

        # dt_transformRollWind = dt_transform.loc[self.rollingWindowStartWhich:self.rollingWindowEndWhich]

        #### Clean & aggregate data
        ## Transform all factor variables
        if len(self.factor_vars)>0:
            dt_transform[self.factor_vars] = dt_transform[self.factor_vars].astype('category')

        ################################################################
        ## Obtain prophet trend, seasonality and change-points
        if self.prophet_vars is not None & len(self.prophet_vars)>0:
            pd.prophet_decomp(dt_transform)
        
        ################################################################
        #### Finalize enriched input
        dt_transform = dt_transform[['ds', 'dep_var'] + self.all_ind_vars]
        self.dt_mod = dt_transform
        self.dt_modRollWind = dt_transform.loc[self.rollingWindowStartWhich:self.rollingWindowEndWhich,:]
        self.dt_inputRollWind = dt_inputRollWind
    
    def mysd(y):
        return np.sqrt(np.sum((y - np.mean(y)) ** 2) / len(y))
    
    def lambda_seq(
        self
        ,x
        ,y
        ,seq_len = 100
        ,lambda_min_ratio = 0.0001):
        
        sx = x / np.apply_along_axis(self.mysd, 0, x)
        check_nan = np.apply_along_axis(lambda sxj: np.all(np.isnan(sxj)), axis=0, arr=sx)
        sx = np.apply_along_axis(lambda sxj, v: np.repeat(0, len(sxj)) if v else sxj, axis=0, arr=sx, v=check_nan)
        sy = y

        # 0.001 is the default smalles alpha value of glmnet for ridge (alpha = 0)
        lambda_max = max(abs(np.sum(sx * sy, axis=0))) / (0.001 * x.shape[0])
        lambda_max_log = math.log(lambda_max)
        log_step = (np.log(lambda_max) - np.log(lambda_max * lambda_min_ratio)) / (seq_len - 1)
        log_seq = np.linspace(np.log(lambda_max), np.log(lambda_max * lambda_min_ratio), num=seq_len)
        lambdas = np.exp(log_seq)
        return lambdas
    
    # Must remain within this function for it to work
    def robyn_iterations(
        self,i
        ):
        t1 = time.time()
        #### Get hyperparameter sample
        hypParamSam = self.hypParamSamNG[i]

        #### Transform media for model fitting
        self.run_transformations()
        # dt_modSaturated = self.dt_modSaturated
        # dt_saturatedImmediate = self.dt_saturatedImmediate
        # dt_saturatedCarryover = self.dt_saturatedCarryover

        #####################################
        #### Split train & test and prepare data for modelling

        dt_window = self.dt_modSaturated

        ## Contrast matrix because glmnet does not treat categorical variables (one hot encoding)
        y_window = dt_window.dep_var
        x_window = pd.get_dummies(dt_window.drop("dep_var",axis=1))
        y_train = y_val = y_test = y_window
        x_train = x_val = x_test = x_window

        ## Split train, test, and validation sets
        train_size = hypParamSam["train_size"][1]
        val_size = test_size = (1 - train_size) / 2
        if train_size < 1:
            train_size_index = int(np.floor(np.quantile(np.arange(dt_window.shape[0]), train_size)))
            val_size_index = train_size_index + round(val_size * dt_window.shape[0])
            y_train = y_window.iloc[1:train_size_index]
            y_val = y_window.iloc[(train_size_index + 1):val_size_index]
            y_test = y_window.iloc[(val_size_index + 1):y_window.shape[0]]
            x_train = x_window.iloc[1:train_size_index, ]
            x_val = x_window.iloc[(train_size_index + 1):val_size_index, ]
            x_test = x_window.iloc[(val_size_index + 1):y_window.shape[0], ]
        else:
            y_val = y_test = x_val = x_test = None
            
        ## Define and set sign control
        dt_sign = dt_window.drop("dep_var",axis=1)
        x_sign = self.prophet_signs+self.context_signs+self.paid_media_signs+self.organic_signs
        x_sign.columns = self.prophet_vars + self.context_vars + self.paid_media_spends + self.organic_vars

        check_factor = dt_sign.dtypes.apply(lambda x: True if pd.api.types.is_categorical_dtype(x) else False)

        lower_limits = upper_limits = None
        for s in range(len(check_factor)):
            if check_factor[s] == True:
                level_n = len(dt_sign.iloc[:, s].astype(str).drop_duplicates())
                if level_n <= 1:
                  raise ValueError("All factor variables must have more than 1 level")
                if x_sign[s] == "positive":
                    lower_vec = [0] * (level_n - 1)
                else:
                  lower_vec = [-float('inf')] * (level_n - 1)
                if x_sign[s] == "negative":
                    upper_vec = [0] * (level_n - 1)
                else:
                    upper_vec = [float("inf")] * (level.n - 1)
                lower_limits = lower_limits+lower_vec
                upper_limits = upper_limits+upper_vec
            else:
                lower_limits = np.where(x_sign[s] == "positive", 0, -np.inf)
                upper_limits = np.append(upper_limits, [0 if x_sign[s] == "negative" else np.inf])

        #####################################
        #### Fit ridge regression with nevergrad's lambda
        # lambdas = lambda_seq(x_train, y_train, seq_len = 100, lambda_min_ratio = 0.0001)
        # lambda_max = max(lambdas)
        lambda_hp = np.ravel(self.hypParamSamNG['lambda'][i])

        if self.hyper_fixed == False:
            lambda_scaled = self.lambda_min + (self.lambda_max - self.lambda_min) * lambda_hp
        else:
            lambda_scaled = lambda_hp

        if self.add_penalty_factor:
            penalty_factor = self.hypParamSamNG.loc[i, np.grep("_penalty", self.hypParamSamNG.columns)].tolist()
        
        else:
            penalty_factor = np.ones(x_train.shape[1])

        #####################################
        ## NRMSE: Model's fit error

        if self.causal_mod==True:
            lm.causal_prediction(
                x_train, y_train,
                x_val, y_val,
                x_test, y_test,
                lambda_scaled = lambda_scaled,
                prior_knowledge = self.prior_knowledge
              )
        else:
            raise NotImplementedError("The process is under consideration...")
            ## If no lift calibration, refit using best lambda
            # self.mod_out = self.model_refit(
            #     x_train, y_train,
            #     x_val, y_val,
            #     x_test, y_test,
            #     lambda_ = lambda_scaled,
            #     lower_limits = lower_limits,
            #     upper_limits = upper_limits,
            #     intercept_sign = self.intercept_sign,
            #     penalty_factor = penalty_factor,
            #   )
            
        self.model_decomp()

        nrmse = np.where(self.ts_validation, self.nrmse_val, self.nrmse_train)
        mape = 0
        df_int = self.df_int

        #####################################
        #### MAPE: Calibration error
        if self.calibration_input is not None:
            liftCollect = None
            warnings.warn("Caliburation modeling is currently not available...")
            # liftCollect = robyn_calibrate(
            # calibration_input = calibration_input,
            # df_raw = dt_mod,
            # hypParamSam = hypParamSam,
            # wind_start = rollingWindowStartWhich,
            # wind_end = rollingWindowEndWhich,
            # dayInterval = InputCollect$dayInterval,
            # dt_modAdstocked = InputCollect$dt_mod,
            # adstock = adstock,
            # xDecompVec = decompCollect$xDecompVec,
            # coefs = decompCollect$coefsOutCat
            # )
            # mape = mean(liftCollect$mape_lift, na.rm = True)

        #####################################
        #### DECOMP.RSSD: Business error
        # Sum of squared distance between decomp share and spend share to be minimized
        dt_decompSpendDist = self.xDecompAgg
        dt_decompSpendDist = dt_decompSpendDist[dt_decompSpendDist['rn'].isin(self.paid_media_spends)]
        dt_decompSpendDist = dt_decompSpendDist[[
            'rn', 'xDecompAgg', 'xDecompPerc', 'xDecompMeanNon0Perc', 
            'xDecompMeanNon0', 'xDecompPercRF', 'xDecompMeanNon0PercRF',
            'xDecompMeanNon0RF']]
        dt_decompSpendDist = dt_decompSpendDist.merge(self.dt_spendShare[['rn', 'spend_share', 'spend_share_refresh', 
                                                             'mean_spend', 'total_spend']], on='rn')
        dt_decompSpendDist['effect_share'] = dt_decompSpendDist['xDecompPerc'] / dt_decompSpendDist['xDecompPerc'].sum()
        dt_decompSpendDist['effect_share_refresh'] = dt_decompSpendDist['xDecompPercRF'] / dt_decompSpendDist['xDecompPercRF'].sum()
        dt_decompSpendDist = pd.merge(
            self.xDecompAgg.query("rn in @paid_media_spends"),
            dt_decompSpendDist.filter(regex='_spend|_share', axis=1),
            on='rn',
            how='left')

        if not self.refresh:
            decomp_rssd = np.sqrt(np.sum((dt_decompSpendDist.effect_share - dt_decompSpendDist.spend_share)**2))
            # Penalty for models with more 0-coefficients
            if self.rssd_zero_penalty:
                is_0eff = np.isclose(np.round(dt_decompSpendDist['effect_share'], 4), 0)
                share_0eff = sum(dt_decompSpendDist.effect_share == 0) / len(dt_decompSpendDist.effect_share)
                decomp_rssd = decomp_rssd * (1 + share_0eff)
            else:
                dt_decompRF = (
                    self.xDecompAgg
                    .loc[:, ["rn", "xDecompPerc"]]
                    .merge(
                        self.xDecompAggPrev[["rn", "xDecompPerc"]],
                        on="rn",
                        how="left",
                        suffixes=("", "_prev")))
                decomp_rssd_media = dt_decompRF[dt_decompRF['rn'].isin(self.paid_media_spends), :]. \
                    assign(diff_decomp_perc=lambda x: x['decomp_perc'] - x['decomp_perc_prev']) \
                        .pipe(lambda x: np.sqrt(np.mean(x['diff_decomp_perc'] ** 2))) \
                            .item()

                decomp_rssd_nonmedia = dt_decompRF[~dt_decompRF['rn'].isin(self.paid_media_spends),].agg(rssd_nonmedia=('decomp_perc', lambda x: np.sqrt(np.mean((x - x.shift(1).fillna(0)) ** 2))))['rssd_nonmedia'][0]

                decomp_rssd = decomp_rssd_media + decomp_rssd_nonmedia / (1 - self.refresh_steps / self.rollingWindowLength)
        
            # When all media in this iteration have 0 coefficients
            if decomp_rssd is None:
                decomp_rssd = float('inf')
                dt_decompSpendDist.effect_share = 0

            #####################################
            #### Collect Multi-Objective Errors and Iteration Results
            # Auxiliary dynamic vector
            common = pd.DataFrame(
                rsq_train = self.mod_out.rsq_train,
                rsq_val = self.mod_out.rsq_val,
                rsq_test = self.mod_out.rsq_test,
                nrmse_train = self.mod_out.nrmse_train,
                nrmse_val = self.mod_out.nrmse_val,
                nrmse_test = self.mod_out.nrmse_test,
                nrmse = nrmse,
                decomp_rssd = decomp_rssd,
                mape = mape,
                _lambda = lambda_scaled,
                lambda_hp = lambda_hp,
                lambda_max = self.lambda_max,
                lambda_min_ratio = self.lambda_min_ratio,
                solID = f"{self.trial}_{self.lng}_{i}",
                trial = self.trial,
                iterNG = self.lng,
                iterPar = i
                )

            total_common = common.shape[1]
            split_common = np.where(common.columns == 'lambda_min_ratio')[0][0]

            hypParamSam = hypParamSam.drop(columns=['lambda']).reset_index(drop=True)
            common_cols = common.columns[0:split_common]
            common_part1 = common[common_cols]
            common_part2 = common.iloc[:, split_common:total_common]
            pos = np.prod(self.xDecompAgg['pos'])
            Elapsed = (dt.datetime.now() - t1).total_seconds()
            ElapsedAccum = (dt.datetime.now() - self.t0).total_seconds()
            common_part1 = pd.concat([common_part1]*len(hypParamSam)).reset_index(drop=True)
            common_part2 = pd.concat([common_part2]*len(hypParamSam)).reset_index(drop=True)
            hypParamSam = pd.concat([hypParamSam, common_part1, common_part2], axis=1)
            hypParamSam = hypParamSam.apply(pd.Series.explode).reset_index(drop=True)
            
            self.resultHypParam = hypParamSam

            self.xDecompAgg = self.xDecompAgg.assign(train_size=train_size).join(common)

            if self.calibration_input is not None:
                self.liftCalibration = liftCollect.merge(common, left_index=True, right_index=True)
            
            self.decompSpendDist = dt_decompSpendDist.merge(common)

            self.common = common
    
    def robyn_mmm(
        self
        ,hyper_collect
        ,iterations
        ,cores
        ,nevergrad_algo
        ,intercept_sign
        ,ts_validation = True
        ,add_penalty_factor = False
        ,dt_hyper_fixed = None
        ,causal_mod = False
        ,rssd_zero_penalty = True
        ,refresh = False
        ,trial = 1
        ,seed = 123
        ,quiet = False):
        ################################################
        #### Collect hyperparameters

        if True:
            hypParamSamName = hyper_collect.hyper_list_all
            # Optimization hyper-parameters
            hyper_bound_list_updated = hyper_collect.hyper_bound_list_updated
            hyper_bound_list_updated_name = hyper_bound_list_updated
            hyper_count = len(hyper_bound_list_updated_name)
            # Fixed hyper-parameters
            hyper_bound_list_fixed = hyper_collect.hyper_bound_list_fixed
            hyper_bound_list_fixed_name = hyper_bound_list_fixed
            hyper_count_fixed = len(hyper_bound_list_fixed_name)
            dt_hyper_fixed_mod = hyper_collect.dt_hyper_fixed_mod
            hyper_fixed = hyper_collect.all_fixed
        
        ## Get environment for parallel backend
        if True:
            self.nevergrad_algo = nevergrad_algo
            self.ts_validation = ts_validation
            self.add_penalty_factor = add_penalty_factor
            self.intercept_sign = intercept_sign
            self.i = None
            self.rssd_zero_penalty = rssd_zero_penalty
            self.causal_mod = causal_mod

        ################################################
        #### Setup environment
        if self.dt_mod is None:
            raise LookupError("Run InputCollect$dt_mod = robyn_engineering() first to get the dt_mod")
        
        ################################################
        #### Get spend share
        dt_inputTrain = self.dt_input.loc[self.rollingWindowStartWhich:self.rollingWindowEndWhich]
        temp = dt_inputTrain[self.paid_media_spends]
        dt_spendShare = pd.DataFrame({
            'rn': self.paid_media_spends,
            'total_spend': temp.apply(lambda x: x.sum(), axis=1),
            'mean_spend': temp.apply(lambda x: x[x > 0].mean() if x[x > 0].size > 0 else 0, axis=1)
            })
        dt_spendShare['spend_share'] = dt_spendShare['total_spend'] / dt_spendShare['total_spend'].sum()

        refreshAddedStartWhich = np.where(self.dt_modRollWind['ds'] == self.refreshAddedStart)[0]

        temp = dt_inputTrain[self.paid_media_spends].iloc[refreshAddedStartWhich:self.rollingWindowLength]
        dt_spendShareRF = pd.DataFrame({
            'rn': self.paid_media_spends,
            'total_spend': temp.sum(),
            'mean_spend': temp.mean()
            })
        dt_spendShareRF['spend_share'] = dt_spendShareRF['total_spend'] / dt_spendShareRF['total_spend'].sum()
        dt_spendShare = dt_spendShare.merge(dt_spendShareRF, on='rn', suffixes=('', '_refresh'), how='left')

        ################################################
        #### Get lambda
        lambda_min_ratio = 0.0001 # default  value from glmnet
        lambdas = self.lambda_seq(
            x = self.dt_mod.drop(["ds","dep_var"],axis=1),
            y = self.dt_mod.dep_var,
            seq_len = 100, 
            lambda_min_ratio = lambda_min_ratio)
        lambda_max = max(lambdas) * 0.1
        lambda_min = lambda_max * lambda_min_ratio

        ################################################
        #### Start Nevergrad loop
        start_time = dt.now()

        ## Set iterations
        if hyper_fixed == False:
            iterTotal = iterations
            iterPar = cores
            iterNG = math.ceil(iterations / cores) # Sometimes the progress bar may not get to 100%
        else:
            iterTotal = iterPar = iterNG = 1
        ## Start Nevergrad optimizer
        if not hyper_fixed:
            my_tuple = tuple(hyper_count)
            instrumentation = ng.p.Array(shape = my_tuple, lower = 0, upper = 1)
            optimizer = ng.optimizers.registry[nevergrad_algo](instrumentation, budget = iterTotal, num_workers = cores)
            # Set multi-objective dimensions for objective functions (errors)
            if self.calibration_input is None:
                optimizer.tell(ng.p.MultiobjectiveReference(), tuple(1, 1))
            else:
                optimizer.tell(ng.p.MultiobjectiveReference(), tuple(1, 1, 1))
        
        ## Prepare loop
        resultCollectNG = dict()
        cnt = 0
        if not hyper_fixed and not quiet:
            pb = tqdm(total=iterTotal, desc='Progress', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
        start_time = time.time()
        for lng in range(1,iterNG):
            nevergrad_hp = dict()
            nevergrad_hp_val = dict()
            hypParamSamList = dict()
            hypParamSamNG = dict()
            
            if hyper_fixed == False:
                # Setting initial seeds (co = cores)
                for co in range(1,iterPar):
                    ## Get hyperparameter sample with ask (random)
                    nevergrad_hp[co] = optimizer.ask()
                    nevergrad_hp_val[co] = nevergrad_hp[co].value
                    ## Scale sample to given bounds using uniform distribution
                    for hypNameLoop in hyper_bound_list_updated_name:
                        index = np.where(hypNameLoop == hyper_bound_list_updated_name)
                        channelBound = list(hyper_bound_list_updated[hypNameLoop])[0]
                        hyppar_value = round(nevergrad_hp_val[co][index], 6)
                        if len(channelBound) > 1:
                            hypParamSamNG[hypNameLoop] = uniform.ppf(hyppar_value, min(channelBound), max(channelBound))
                        else:
                            hypParamSamNG[hypNameLoop] = hyppar_value
                    hypParamSamList[co] = pd.DataFrame(hypParamSamNG.T)
                hypParamSamNG = pd.concat(hypParamSamList)
                hypParamSamNG.columns = hyper_bound_list_updated_name
                ## Add fixed hyperparameters
                if hyper_count_fixed != 0:
                    hypParamSamNG = pd.concat(hypParamSamList)
                    hypParamSamNG.columns = hyper_bound_list_updated_name
                    hypParamSamNG = pd.concat([hypParamSamNG, dt_hyper_fixed_mod], axis=1)
                    hypParamSamNG = hypParamSamNG.loc[:, hypParamSamName]
                else:
                    hypParamSamNG = dt_hyper_fixed_mod[hypParamSamName]
            ########### Parallel start
            nrmse_collect = None
            decomp_rssd_collect = None
            best_mape = float("inf")

            if self.core == 1:
                doparCollect = Parallel(n_jobs=2)(delayed(self.robyn_iterations)(i) for i in range(1, iterPar+1))
            else:
                if check_parallel() and not hyper_fixed:
                    # Set up a pool of worker processes
                    pool = multiprocessing.Pool(self.core)
                    # Use the pool to parallelize some function across the workers
                    doparCollect = pool.map(self.robyn_iterations)
                    # Close the pool when done
                    pool.close()
                else:
                    doparCollect = []
                    with ThreadPoolExecutor() as executor:
                        for result in executor.map(self.robyn_iterations, range(iterPar)):
                            doparCollect.append(result)
            
            nrmse_collect = [x['nrmse'] for x in doparCollect]
            decomp_rssd_collect = [x['decomp_rssd'] for x in doparCollect]
            mape_lift_collect = [x['mape'] for x in doparCollect]

            #####################################
            #### Nevergrad tells objectives

            if not hyper_fixed:
                if self.calibration_input is None:
                    for co in range(iterPar):
                        optimizer.tell(nevergrad_hp[[co]], tuple(nrmse_collect[co], decomp_rssd_collect[co]))
                    else:
                        for co in range(iterPar):
                            optimizer.tell(nevergrad_hp[[co]], tuple(nrmse_collect[co], decomp_rssd_collect[co], mape_lift_collect[co]))
            
            resultCollectNG[lng] = doparCollect
        
        #####################################
        #### Final result collect
        self.resultHypParam = pd.concat([pd.concat([pd.DataFrame(y['resultHypParam']) for y in x]) for x in resultCollectNG]).reset_index(drop=True)
        self.xDecompAgg = pd.concat([pd.concat([y['xDecompAgg'] for y in x]) for x in resultCollectNG]).reset_index(drop=True)

        if self.calibration_input is not None:
            lift_cal_collect = []
            for result_collect in resultCollectNG:
                lift_cal = []
                for result in result_collect:
                    lift_cal.append(result["liftCalibration"])
                lift_cal_collect.append(pd.concat(lift_cal))
            result_df = pd.concat(lift_cal_collect).sort_values(["mape", "liftMedia", "liftStart"]).reset_index(drop=True)
            result_df = result_df.reset_index(drop=True)
            result_df.columns = result_df.columns.str.replace(".", "_")
            result_df = result_df.astype({"pos": np.int32})
            result_df = result_df.reset_index(drop=True)
            self.liftCalibration = result_df
        
        self.decompSpendDist = pd.concat([pd.concat([pd.DataFrame(y) for y in x], ignore_index=True) for x in resultCollectNG]).reset_index(drop=True)

        self.iter = len(self.mape)
        # Adjust accumulated time
        self.resultHypParam = self.resultHypParam.assign(
            ElapsedAccum=lambda x: x.ElapsedAccum - min(x.ElapsedAccum) + 
            x.Elapsed.iloc[x.ElapsedAccum.idxmin()])
        self.hyperBoundNG = hyper_bound_list_updated
        self.hyperBoundFixed = hyper_bound_list_fixed

    def robyn_train(
        self
        ,hyper_collect
        ,cores
        ,iterations
        ,trials
        ,intercept_sign
        ,nevergrad_algo
        ,dt_hyper_fixed = None
        ,ts_validation = True
        ,add_penalty_factor = False
        ,rssd_zero_penalty = True
        ,refresh = False
        ,seed = 123
        ,causal_mod = True
        ,quiet = False
        ):
        hyper_fixed = hyper_collect.all_fixed

        if hyper_fixed:
            self.OutputModels = self.robyn_mmm(
                self,
                hyper_collect = hyper_collect,
                iterations = iterations,
                cores = cores,
                nevergrad_algo = nevergrad_algo,
                intercept_sign = intercept_sign,
                dt_hyper_fixed = dt_hyper_fixed,
                ts_validation = ts_validation,
                add_penalty_factor = add_penalty_factor,
                rssd_zero_penalty = rssd_zero_penalty,
                seed = seed,
                quiet = quiet,
                causal_mod = causal_mod
                )
            self.trial = 1

            if "solID" in dt_hyper_fixed:
                these = list("resultHypParam", "xDecompVec", "xDecompAgg", "decompSpendDist")
                for tab in these:
                    self.OutputModels[tab]["solID"] = dt_hyper_fixed.solID
        else:
            # ck.check_init_msg(cores)
            if not quiet:
                print(f">>> Starting {trials} trials with {iterations} iterations each using with calibration using {nevergrad_algo} nevergrad algorithm...")
        
            OutputModels = dict()
            
            for ngt in range(1,trials+1):
                if not quiet:
                    print(f"  Running trial {ngt} of {trials}")
                    model_output = self.robyn_mmm(
                        hyper_collect = hyper_collect,
                        iterations = iterations,
                        cores = cores,
                        nevergrad_algo = nevergrad_algo,
                        intercept_sign = intercept_sign,
                        ts_validation = ts_validation,
                        add_penalty_factor = add_penalty_factor,
                        rssd_zero_penalty = rssd_zero_penalty,
                        refresh = refresh,
                        trial = ngt,
                        seed = seed + ngt,
                        quiet = quiet,
                        causal_mod = causal_mod
                    )
                    check_coef0 = any(np.isinf(self.decompSpendDist["decomp_rssd"]))
                    if check_coef0:
                        num_coef0_mod = self.decompSpendDist.loc[self.decompSpendDist["decomp_rssd"].isin([np.inf, -np.inf])].drop_duplicates(["iterNG", "iterPar"]).shape[0]
                        num_coef0_mod = iterations if num_coef0_mod > iterations else num_coef0_mod
                self.trial = ngt
                self.OutputModels[f"trial{ngt}"] = model_output

    def robyn_run(
        self
        ,dt_hyper_fixed = None
        ,ts_validation = False
        ,add_penalty_factor = False
        ,refresh = False
        ,seed = 123
        ,outputs = False
        ,quiet = False
        ,cores = None
        ,trials = 5
        ,iterations = 2000
        ,rssd_zero_penalty = True
        ,nevergrad_algo = "TwoPointsDE"
        ,intercept_sign = "non_negative"
        ,lambda_control = None
        ,causal_mod = False
        ):

        start_time = dt.now()
        # Use previously exported model (Consider to add in the future)

        if self.hyper_params is None:
            raise ValueError("Must provide 'hyperparameters' in robyn_inputs()'s output first")
        
        max_cores = max(1, multiprocessing.cpu_count())
        if cores is None:
            cores = max_cores
        elif cores > max_cores:
            warnings.warn(f"Max possible cores in your machine is {max_cores} (your input was {cores})")
            cores = max_cores
        if cores == 0:
            cores = 1
        
        hyps_fixed = dt_hyper_fixed is not None
        if hyps_fixed:
            trials = iterations = 1
        
        ck.check_run_inputs(cores,iterations,trials,intercept_sign,nevergrad_algo)

        # currently unable to calibrate
        self.calibration_input = None
        ck.check_iteration(iterations,trials,hyps_fixed,refresh)
        ref.init_msgs_run(refresh,lambda_control, quiet)

        #####################################
        #### Prepare hyper-parameters
        hyper_collect = hp.hyper_collector(
            hyper_in = self.hyperparameters
            ,ts_validation = ts_validation
            ,add_penalty_factor = add_penalty_factor
            ,dt_hyper_fixed = dt_hyper_fixed
            ,cores = cores
            )

        self.hyper_updated = hyper_collect.hyper_list_all

        self.robyn_train(
            hyper_collect
            ,cores = cores
            ,iterations = iterations
            ,trials = trials
            ,intercept_sign = intercept_sign
            ,nevergrad_algo = nevergrad_algo
            ,dt_hyper_fixed = dt_hyper_fixed
            ,ts_validation = ts_validation
            ,add_penalty_factor = add_penalty_factor
            ,rssd_zero_penalty = rssd_zero_penalty
            ,refresh = refresh
            ,seed = seed
            ,quiet = quiet
            ,causal_mod = causal_mod
        )

        self.OutputModels.attr["hyper_fixed"] = hyper_collect.all_fixed
        # self.OutputModels.attr["bootstrap"] = bootstrap
        self.OutputModels.attr["refresh"] = refresh

        if True:
            self.OutputModels.cores = cores
            self.OutputModels.iterations = iterations
            self.OutputModels.trials = trials
            self.OutputModels.intercept_sign = intercept_sign
            self.OutputModels.nevergrad_algo = nevergrad_algo
            self.OutputModels.ts_validation = ts_validation
            self.OutputModels.add_penalty_factor = add_penalty_factor
            self.OutputModels.hyper_updated = hyper_collect.hyper_list_all
        
        # Not direct output & not all fixed hyperparameters
        if not outputs and dt_hyper_fixed is None:
            output = self.OutputModels
        elif not hyper_collect.all_fixed:
            raise NotImplementedError("The process is under implementing...")
        else:
            raise NotImplementedError("The process is under implementing...")
        
        # Check convergence when more than 1 iteration
        if not hyper_collect.all_fixed:
            raise NotImplementedError("The process is under implementing...")
        else:
            if "solID" in dt_hyper_fixed:
                output.solID = dt_hyper_fixed.solID
            else:
                output.selectID = self.OutputModels.trial1.resultHypParam.solID
            if not quiet:
                print(f"Successfully recreated model ID: {output.selectID}")
        
        # Save hyper-parameters list
        output.hyper_updated = hyper_collect.hyper_list_all
        output.seed = seed

        # # Report total timing
        # attr(output, "runTime") <- round(difftime(Sys.time(), t0, units = "mins"), 2)
        # if not quiet and iterations > 1:
        #     message(paste("Total run time:", attr(output, "runTime"), "mins"))

        output.__class__ = type("robyn_models", (type(output),), {})

        self.output = output
    
    def model_decomp(self):
        ## Input for decomp
        y = self.dt_modSaturated.dep_var
        x = self.dt_modSaturated.drop("dep_var",axis=1)
        intercept = self.coefs[1]
        x_name = [s for s in x.columns]
        x_factor = x_name[np.apply_along_axis(lambda col: np.issubdtype(col.dtype, np.object), 0, x)]

        ## Decomp x
        xDecomp = pd.DataFrame(np.multiply(x, self.coefs[1:, np.newaxis]), columns=x.columns)
        xDecomp = pd.concat([pd.DataFrame({'intercept': [intercept] * xDecomp.shape[0]}), xDecomp], axis=1)
        xDecompOut = pd.concat([self.dt_modRollWind['ds'], y, self.y_pred, xDecomp], axis=1)

        ## Decomp immediate & carryover response
        sel_coef = [col in list(self.dt_saturatedImmediate.columns) for col in self.coefs.index.values+[s for s in self.coefs.columns]]
        coefs_media = self.coefs[sel_coef]
        coefs_media.columns = self.coefs[sel_coef].index
        mediaDecompImmediate = pd.DataFrame(map(lambda regressor, coeff: regressor * coeff, self.dt_saturatedImmediate.values(), coefs_media.values()), index=self.dt_saturatedImmediate.index, columns=coefs_media.index)
        mediaDecompCarryover = pd.DataFrame(self.dt_saturatedCarryover * coefs_media.values, columns=coefs_media.index, index=self.dt_saturatedCarryover.index)

        ## Output decomp
        y_hat = np.sum(xDecomp, axis=1, where=~np.isnan(xDecomp))
        y_hat_scaled = np.nansum(np.abs(xDecomp), axis=1)
        xDecompOutPerc_scaled = abs(xDecomp) / y_hat_scaled
        xDecompOut_scaled = y_hat * xDecompOutPerc_scaled

        temp = xDecompOut[["intercept"] + x_name]
        xDecompOutAgg = temp.apply(sum, axis=0).tolist()
        xDecompOutAggPerc = xDecompOutAgg / sum(y_hat)
        xDecompOutAggMeanNon0 = [np.mean(x[x != 0]) if np.mean(x[x > 0]) else 0 for _, x in temp.iteritems()]
        xDecompOutAggMeanNon0 = [0 if np.isnan(x) else x for x in xDecompOutAggMeanNon0]
        xDecompOutAggMeanNon0Perc = xDecompOutAggMeanNon0 / sum(xDecompOutAggMeanNon0)

        refreshAddedStartWhich = np.where(xDecompOut['ds'] == self.refreshAddedStart)[0]
        refreshAddedEnd = max(xDecompOut.ds)
        refreshAddedEndWhich = np.where(xDecompOut['ds'] == refreshAddedEnd)[0]

        temp = xDecompOut.loc[xDecompOut['ds'].between(self.refreshAddedStart, refreshAddedEnd),["intercept"] + x_name]
        xDecompOutAggRF = [sum(x) for _, x in temp.iteritems()]
        y_hatRF = y_hat.loc[refreshAddedStartWhich:refreshAddedEndWhich]
        xDecompOutAggPercRF = xDecompOutAggRF / sum(y_hatRF)
        xDecompOutAggMeanNon0RF = [np.mean(x[x != 0]) if np.mean(x[x > 0]) == np.mean(x[x > 0]) else 0 for x in temp.values.T]
        xDecompOutAggMeanNon0RF = np.where(np.isnan(xDecompOutAggMeanNon0RF), 0, xDecompOutAggMeanNon0RF)
        xDecompOutAggMeanNon0PercRF = xDecompOutAggMeanNon0RF / sum(xDecompOutAggMeanNon0RF)

        coefsOutCat = coefsOut = pd.DataFrame({'rn': np.concatenate([list(self.coefs.index), self.coefs.columns]), 'coefs': self.coefs.values.flatten()})
        if len(x_factor) > 0:
            for factor in x_factor:
                coefsOut['rn'] = coefsOut['rn'].str.replace(f'{factor}.*', factor)
        
        rn_order = list(xDecompOutAgg.keys())
        rn_order = [s.replace('intercept', '(Intercept)') for s in rn_order]

        coefsOut = coefsOut.groupby(coefsOut['rn']).mean()
        coefsOut = coefsOut.rename(columns={coefsOut.columns[1]: 'coef'})
        coefsOut = coefsOut.reset_index()
        coefsOut = coefsOut.sort_values('rn', key=lambda x: x.map(dict(zip(rn_order, range(len(rn_order)))))).reset_index(drop=True)

        decompOutAgg = pd.concat([
            coefsOut, 
            pd.DataFrame({
                "xDecompAgg": xDecompOutAgg,
                "xDecompPerc": xDecompOutAggPerc,
                "xDecompMeanNon0": xDecompOutAggMeanNon0,
                "xDecompMeanNon0Perc": xDecompOutAggMeanNon0Perc,
                "xDecompAggRF": xDecompOutAggRF,
                "xDecompPercRF": xDecompOutAggPercRF,
                "xDecompMeanNon0RF": xDecompOutAggMeanNon0RF,
                "xDecompMeanNon0PercRF": xDecompOutAggMeanNon0PercRF,
                "pos": xDecompOutAgg >= 0
                })]
                , axis=1)
        
        self.xDecompVec = xDecompOut
        self.xDecompVec_scaled = xDecompOut_scaled
        self.xDecompAgg = decompOutAgg
        self.coefsOutCat = coefsOutCat
        self.mediaDecompImmediate = mediaDecompImmediate = mediaDecompImmediate.assign(ds=xDecompOut.ds, y=xDecompOut.y)
        self.mediaDecompCarryover = mediaDecompCarryover.assign(ds = xDecompOut.ds, y = xDecompOut.y)
    
    def model_refit(
        self
        ,x_train
        ,y_train
        ,x_val
        ,y_val
        ,x_test
        ,y_test
        ,lambda_
        ,lower_limits
        ,upper_limits
        ,intercept_sign = "non_negative"
        ,penalty_factor = None
        ):
        if penalty_factor is None:
            penalty_factor = np.ones(y_train.shape[1])
        
        mod = ElasticNet(alpha=0, lambdau=lambda_, lower_limits=lower_limits, upper_limits=upper_limits, scoring = "mean_squared_error", penalty_factor=penalty_factor)
        mod.fit(x_train, y_train)

        df_int = 1

        if intercept_sign == "non_negative" and mod.coef_[1]<0:
            mod = ElasticNet(alpha = 0, lambdau = lambda_, lower_limits = lower_limits, upper_limits = upper_limits, penalty_factor = penalty_factor, fit_intercept = False)
            mod.fit(x_train, y_train)

            df_int = 1 

        # Calculate all Adjusted R2
        y_train_pred = mod.predict(x_train, lamb = lambda_)
        rsq_train = lm.get_rsq(true = y_train, predicted = y_train_pred, p = x_train.shape[1], df_int = df_int)
        if x_val is not None:
            y_val_pred = mod.predict(x_val, lamb = lambda_) 
            rsq_val = lm.get_rsq(true = y_val, predicted = y_val_pred, p = x_val.shape[1], df_int = df_int, n_train = len(y_train))
            y_test_pred = mod.predict(x_test, lamb = lambda_)
            rsq_test = lm.get_rsq(true = y_test, predicted = y_test_pred, p = x_test.shape[1], df_int = df_int, n_train = len(y_train))
            y_pred = np.concatenate((y_train_pred, y_val_pred, y_test_pred))
        else:
            rsq_val = rsq_test = None
            y_pred = y_train_pred
        
        # Calculate all NRMSE
        nrmse_train = np.sqrt(np.mean((y_train - y_train_pred)^2)) / (max(y_train) - min(y_train))
        if x_val is not None:
            nrmse_val = np.sqrt(np.mean(sum((y_val - y_val_pred)^2))) / (max(y_val) - min(y_val))
            nrmse_test = np.sqrt(np.mean(sum((y_test - y_test_pred)^2))) / (max(y_test) - min(y_test))
        else:
            nrmse_val = nrmse_test = y_val_pred = y_test_pred = None
        
        self.rsq_train = rsq_train
        self.rsq_val = rsq_val
        self.rsq_test = rsq_test
        self.nrmse_train = nrmse_train
        self.nrmse_val = nrmse_val
        self.nrmse_test = nrmse_test
        self.coefs = mod.coef_
        self.y_train_pred = y_train_pred
        self.y_val_pred = y_val_pred
        self.y_test_pred = y_test_pred
        self.y_pred = y_pred
        self.mod = mod
        self.df_int = df_int

