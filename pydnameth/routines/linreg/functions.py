import statsmodels.api as sm
import numpy as np
from scipy.stats import shapiro, kstest, normaltest
from statsmodels.stats.stattools import jarque_bera, omni_normtest, durbin_watson


def process_linreg(x, y, metrics_dict):
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

    metrics_dict['mean'].append(np.mean(y))
    metrics_dict['R2'].append(results.rsquared)
    metrics_dict['R2_adj'].append(results.rsquared_adj)
    metrics_dict['f_stat'].append(results.fvalue)
    metrics_dict['prob(f_stat)'].append(results.f_pvalue)
    metrics_dict['log_likelihood'].append(results.llf)
    metrics_dict['AIC'].append(results.aic)
    metrics_dict['BIC'].append(results.bic)
    metrics_dict['omnibus'].append(omni)
    metrics_dict['prob(omnibus)'].append(omnipv)
    metrics_dict['skew'].append(skew)
    metrics_dict['kurtosis'].append(kurtosis)
    metrics_dict['durbin_watson'].append(durbin_watson(results.wresid))
    metrics_dict['jarque_bera'].append(jb)
    metrics_dict['prob(jarque_bera)'].append(jbpv)
    metrics_dict['cond_no'].append(results.condition_number)
    metrics_dict['normality_p_value_shapiro'].append(normality_p_value_shapiro)
    metrics_dict['normality_p_value_ks_wo_params'].append(normality_p_value_ks_wo_params)
    metrics_dict['normality_p_value_ks_with_params'].append(normality_p_value_ks_with_params)
    metrics_dict['normality_p_value_dagostino'].append(normality_p_value_dagostino)
    metrics_dict['intercept'].append(results.params[0])
    metrics_dict['slope'].append(results.params[1])
    metrics_dict['intercept_std'].append(results.bse[0])
    metrics_dict['slope_std'].append(results.bse[1])
    metrics_dict['intercept_p_value'].append(results.pvalues[0])
    metrics_dict['slope_p_value'].append(results.pvalues[1])
