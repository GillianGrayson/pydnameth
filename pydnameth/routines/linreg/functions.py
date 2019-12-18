import statsmodels.api as sm
import numpy as np
from scipy.stats import shapiro, kstest, normaltest
from statsmodels.stats.stattools import jarque_bera, omni_normtest, durbin_watson


def process_linreg(x, y, metrics_dict, suffix):
    x = sm.add_constant(x)

    results = sm.OLS(y, x).fit()

    residuals = results.resid

    jb, jbpv, skew, kurtosis = jarque_bera(results.wresid)
    omni, omnipv = omni_normtest(results.wresid)

    res_mean = np.mean(residuals)
    res_std = np.std(residuals)

    _, normality_p_value_shapiro = shapiro(residuals)
    _, normality_p_value_ks_wo_params = kstest(residuals, 'norm')
    _, normality_p_value_ks_with_params = kstest(residuals, 'norm', (res_mean, res_std))
    _, normality_p_value_dagostino = normaltest(residuals)

    metrics_dict['mean' + suffix].append(np.mean(y))
    metrics_dict['R2' + suffix].append(results.rsquared)
    metrics_dict['R2_adj' + suffix].append(results.rsquared_adj)
    metrics_dict['f_stat' + suffix].append(results.fvalue)
    metrics_dict['prob(f_stat)' + suffix].append(results.f_pvalue)
    metrics_dict['log_likelihood' + suffix].append(results.llf)
    metrics_dict['AIC' + suffix].append(results.aic)
    metrics_dict['BIC' + suffix].append(results.bic)
    metrics_dict['omnibus' + suffix].append(omni)
    metrics_dict['prob(omnibus)' + suffix].append(omnipv)
    metrics_dict['skew' + suffix].append(skew)
    metrics_dict['kurtosis' + suffix].append(kurtosis)
    metrics_dict['durbin_watson' + suffix].append(durbin_watson(results.wresid))
    metrics_dict['jarque_bera' + suffix].append(jb)
    metrics_dict['prob(jarque_bera)' + suffix].append(jbpv)
    metrics_dict['cond_no' + suffix].append(results.condition_number)
    metrics_dict['normality_p_value_shapiro' + suffix].append(normality_p_value_shapiro)
    metrics_dict['normality_p_value_ks_wo_params' + suffix].append(normality_p_value_ks_wo_params)
    metrics_dict['normality_p_value_ks_with_params' + suffix].append(normality_p_value_ks_with_params)
    metrics_dict['normality_p_value_dagostino' + suffix].append(normality_p_value_dagostino)
    metrics_dict['intercept' + suffix].append(results.params[0])
    metrics_dict['slope' + suffix].append(results.params[1])
    metrics_dict['intercept_std' + suffix].append(results.bse[0])
    metrics_dict['slope_std' + suffix].append(results.bse[1])
    metrics_dict['intercept_p_value' + suffix].append(results.pvalues[0])
    metrics_dict['slope_p_value' + suffix].append(results.pvalues[1])
