import pandas as pd
import numpy as np
import re
from scipy.stats import weibull_min

def run_transformations(self):
    all_media = self.all_media
    rollingWindowStartWhich = self.rollingWindowStartWhich
    rollingWindowEndWhich = self.rollingWindowEndWhich
    dt_modAdstocked = self.dt_mod.drop(["ds"],axis=1)

    mediaAdstocked = dict()
    mediaImmediate = list()
    mediaCarryover = list()
    mediaVecCum = list()
    mediaSaturated = list()
    mediaSaturatedImmediate = list()
    mediaSaturatedCarryover = list()

    for v in range(len(all_media)):
        ################################################
        ## 1. Adstocking (whole data)
        # Decayed/adstocked response = Immediate response + Carryover response
        m = dt_modAdstocked[all_media[v]]
        if self.adstock == "geometric":
            theta = self.hypParamSam[f"{all_media[v]}_thetas"][0][0]
        if re.search("weibull", self.adstock) is not None:
            shape = self.hypParamSam[f"{all_media[v]}_shapes"][0][0]
            scale = self.hypParamSam[f"{all_media[v]}_scales"][0][0]
        x_list = self.transform_adstock(m, self.adstock, theta = theta, shape = shape, scale = scale)
        m_adstocked = x_list.x_decayed
        mediaAdstocked[v] = m_adstocked
        m_carryover = m_adstocked - m
        m[m_carryover < 0] = m_adstocked[m_carryover < 0] # adapt for weibull_pdf with lags
        m_carryover[m_carryover < 0] = 0 # adapt for weibull_pdf with lags
        mediaImmediate[v] = m
        mediaCarryover[v] = m_carryover
        mediaVecCum[[v]] = x_list.thetaVecCum

        ################################################
        ## 2. Saturation (only window data)
        # Saturated response = Immediate response + carryover response
        m_adstockedRollWind = m_adstocked.loc[rollingWindowStartWhich:rollingWindowEndWhich]
        m_carryoverRollWind = m_carryover.loc[rollingWindowStartWhich:rollingWindowEndWhich]

        alpha = self.hypParamSam[f"{all_media[v]}_alphas"][0][0]
        gamma = self.hypParamSam[f"{all_media[v]}_gammas"][0][0]
        mediaSaturated[v] = self.saturation_hill(
            m_adstockedRollWind
            ,alpha = alpha
            , gamma = gamma
            )
        mediaSaturatedCarryover[v] = self.saturation_hill(
            m_adstockedRollWind
            ,alpha = alpha
            ,gamma = gamma
            ,x_marginal = m_carryoverRollWind
            )
        mediaSaturatedImmediate[v] = mediaSaturated[v] - mediaSaturatedCarryover[v]

        mediaAdstocked.name = mediaImmediate.name = mediaCarryover.name = mediaVecCum.name = \
        mediaSaturated.name = mediaSaturatedImmediate.name = mediaSaturatedCarryover.name = all_media
        
        dt_modAdstocked.drop(columns=all_media, inplace=True)
        dt_modAdstocked = pd.concat([dt_modAdstocked, mediaAdstocked], axis=1)
        dt_mediaImmediate = pd.concat(mediaImmediate, axis=1)
        dt_mediaCarryover = pd.concat(mediaCarryover, axis=1)
        mediaVecCum = pd.concat(mediaVecCum, axis=1)
        dt_modSaturated = dt_modAdstocked.iloc[rollingWindowStartWhich:rollingWindowEndWhich] \
                  .drop(all_media, axis=1) \
                  .join(mediaSaturated)

        dt_saturatedImmediate = pd.concat(mediaSaturatedImmediate, axis=1)
        dt_saturatedImmediate[dt_saturatedImmediate is None] = 0
        dt_saturatedCarryover = pd.concat(mediaSaturatedCarryover, axis=1)
        dt_saturatedCarryover[dt_saturatedCarryover is None] = 0
        
        self.dt_modSaturated = dt_modSaturated
        self.dt_saturatedImmediate = dt_saturatedImmediate
        self.dt_saturatedCarryover = dt_saturatedCarryover

def transform_adstock(
    self
    ,x 
    ,theta = None
    ,shape = None
    ,scale = None
    ,windlen = None
    ):
    if windlen is None:
        windlen = len(x)
    # check_adstock(adstock)
    if self.adstock == "geometric":
        x_list_sim = self.adstock_geometric(x = x, theta = theta)
    elif self.adstock == "weibull_cdf":
        x_list_sim = self.adstock_weibull(x = x, shape = shape, scale = scale, windlen = windlen, type = "cdf")
    elif self.adstock == "weibull_pdf":
        x_list_sim = self.adstock_weibull(x = x, shape = shape, scale = scale, windlen = windlen, type = "pdf")
    return x_list_sim

def saturation_hill(
    self
    ,x
    ,alpha
    ,gamma
    ,x_marginal = None):
    # stopifnot(length(alpha) == 1)
    assert len(alpha) == 1, "alpha must be a single value"
    # stopifnot(length(gamma) == 1)
    assert len(gamma) == 1, "gamma must be a single value"
    inflexion = np.dot(np.array(range(x)), np.array([1-gamma, gamma]))
    # inflexion = c(range(x) %*% c(1 - gamma, gamma)) # linear interpolation by dot product
    if x_marginal is None:
        x_scurve = x**alpha / (x**alpha + inflexion**alpha) # plot(x_scurve) summary(x_scurve)
    else:
        x_scurve = x_marginal**alpha / (x_marginal**alpha + inflexion**alpha)
    return x_scurve

def adstock_geometric(
    self
    ,x
    ,theta
    ):
    # stopifnot(length(theta) == 1)
    assert len(theta) == 1, "theta must be a single value"
    if len(x) > 1:
        x_decayed = [x[0]] + [0]*(len(x)-1)
        
        for xi in range(2,len(x_decayed)):
            x_decayed[xi] = x[xi] + theta * x_decayed[xi - 1]
        
        thetaVecCum = theta
        for t in range(2,len(x)):
            thetaVecCum[t] = thetaVecCum[t - 1] * theta
    else:
        x_decayed = x
        thetaVecCum = theta
    inflation_total = sum(x_decayed) / sum(x)
    return {
        "x" : x
        ,"x_decayed" : x_decayed
        ,"thetaVecCum" : thetaVecCum
        ,"inflation_total" : inflation_total
        }

def adstock_weibull(
    self
    ,x
    ,shape
    ,scale
    ,windlen = None
    ,type = "cdf"
    ):
    if windlen is None:
        windlen = len(x)
    # stopifnot(length(shape) == 1)
    assert len(shape) == 1, "shape must be a single value"
    # stopifnot(length(scale) == 1)
    assert len(scale) == 1, "scale must be a single value"
    if len(x) > 1:
        # check_opts(tolower(type), c("cdf", "pdf"))
        x_bin = list(range(1, windlen+1))
        scaleTrans = np.round(np.quantile(np.arange(1, windlen+1), scale))
        if shape == 0 or scale == 0:
            x_decayed = x
            thetaVecCum = thetaVec = [0] * windlen
        else:
            if "cdf" in type.lower():
                thetaVec = np.concatenate(([1], 1 - weibull_min.cdf(x_bin[:-1], shape, scale=scaleTrans))) # plot(thetaVec)
                thetaVecCum = np.cumprod(thetaVec) # plot(thetaVecCum)
            elif "pdf" in type.lower():
                thetaVecCum = self.normalize(weibull_min.pdf(x_bin, shape, scale=scaleTrans)) # plot(thetaVecCum)

            x_decayed = np.zeros(windlen)
            for i in range(len(x)):
                x_vec = np.concatenate((np.zeros(i), np.full(windlen - i, x[i])))
                thetaVecCumLag = np.concatenate((np.zeros(i), thetaVecCum[:-i]))
                x_prod = x_vec * thetaVecCumLag
                x_decayed += x_prod

            x_decayed = np.sum(x_decayed, axis=0)[np.arange(len(x))]
    else:
        x_decayed = x
        thetaVecCum = 1
    
    inflation_total = sum(x_decayed) / sum(x)

    return {
        "x" : x
        ,"x_decayed" : x_decayed
        ,"thetaVecCum" : thetaVecCum
        ,"inflation_total" : inflation_total
        }


def normalize(self,x):
    if (max(x) - min(x) == 0):
        return [1] + [0] * (len(x) - 1)
    else:
        return (x - min(x)) / (max(x) - min(x))