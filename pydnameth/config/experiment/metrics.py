from pydnameth import DataType, Task, Method


def get_method_metrics_keys(config):
    metrics = []

    if config.experiment.data in [DataType.betas,
                                  DataType.betas_adj,
                                  DataType.epimutations,
                                  DataType.entropy,
                                  DataType.residuals,
                                  DataType.cells,
                                  DataType.genes]:

        if config.experiment.task == Task.table:

            if config.experiment.method == Method.heteroskedasticity:

                metrics = [
                    'item',
                    'aux',
                    'type',
                    'bp_lm',
                    'bp_lm_pvalue',
                    'bp_lm_pvalue_fdr_bh',
                    'bp_lm_pvalue_bonferroni',
                    'bp_fvalue',
                    'bp_f_pvalue',
                    'bp_f_pvalue_fdr_bh',
                    'bp_f_pvalue_bonferroni',
                    'w_lm',
                    'w_lm_pvalue',
                    'w_lm_pvalue_fdr_bh',
                    'w_lm_pvalue_bonferroni',
                    'w_fvalue',
                    'w_f_pvalue',
                    'w_f_pvalue_fdr_bh',
                    'w_f_pvalue_bonferroni',
                    'gq_fvalue',
                    'gq_f_pvalue',
                    'gq_f_pvalue_fdr_bh',
                    'gq_f_pvalue_bonferroni',
                    'gq_type'
                ]

            if config.experiment.method == Method.linreg:

                metrics = [
                    'item',
                    'aux',
                    'mean',
                    'R2',
                    'R2_adj',
                    'f_stat',
                    'prob(f_stat)',
                    'log_likelihood',
                    'AIC',
                    'BIC',
                    'omnibus',
                    'prob(omnibus)',
                    'skew',
                    'kurtosis',
                    'durbin_watson',
                    'jarque_bera',
                    'prob(jarque_bera)',
                    'cond_no',
                    'intercept',
                    'slope',
                    'intercept_std',
                    'slope_std',
                    'intercept_p_value',
                    'slope_p_value',
                    'normality_p_value_shapiro',
                    'normality_p_value_ks_wo_params',
                    'normality_p_value_ks_with_params',
                    'normality_p_value_dagostino'
                ]

            elif config.experiment.method == Method.ancova:

                metrics = [
                    'item',
                    'aux',
                    'p_value',
                    'p_value_fdr_bh',
                    'p_value_bonferroni'
                ]

            elif config.experiment.method == Method.oma:

                metrics = [
                    'item',
                    'aux',
                    'corr_coeff',
                    'p_value',
                    'p_value_fdr_bh',
                    'p_value_bonferroni'
                ]

            elif config.experiment.method == Method.variance:

                metrics = [
                    'item',
                    'aux',

                    'best_R2',

                    'increasing_div',
                    'increasing_sub',

                    'increasing_type',

                    'box_b_best_type',
                    'box_b_best_R2',
                    'box_b_lin_lin_R2',
                    'box_b_lin_lin_intercept',
                    'box_b_lin_lin_slope',
                    'box_b_lin_lin_intercept_std',
                    'box_b_lin_lin_slope_std',
                    'box_b_lin_lin_intercept_p_value',
                    'box_b_lin_lin_slope_p_value',
                    'box_b_lin_log_R2',
                    'box_b_lin_log_intercept',
                    'box_b_lin_log_slope',
                    'box_b_lin_log_intercept_std',
                    'box_b_lin_log_slope_std',
                    'box_b_lin_log_intercept_p_value',
                    'box_b_lin_log_slope_p_value',
                    'box_b_log_log_R2',
                    'box_b_log_log_intercept',
                    'box_b_log_log_slope',
                    'box_b_log_log_intercept_std',
                    'box_b_log_log_slope_std',
                    'box_b_log_log_intercept_p_value',
                    'box_b_log_log_slope_p_value',

                    'box_m_best_type',
                    'box_m_best_R2',
                    'box_m_lin_lin_R2',
                    'box_m_lin_lin_intercept',
                    'box_m_lin_lin_slope',
                    'box_m_lin_lin_intercept_std',
                    'box_m_lin_lin_slope_std',
                    'box_m_lin_lin_intercept_p_value',
                    'box_m_lin_lin_slope_p_value',
                    'box_m_lin_log_R2',
                    'box_m_lin_log_intercept',
                    'box_m_lin_log_slope',
                    'box_m_lin_log_intercept_std',
                    'box_m_lin_log_slope_std',
                    'box_m_lin_log_intercept_p_value',
                    'box_m_lin_log_slope_p_value',
                    'box_m_log_log_R2',
                    'box_m_log_log_intercept',
                    'box_m_log_log_slope',
                    'box_m_log_log_intercept_std',
                    'box_m_log_log_slope_std',
                    'box_m_log_log_intercept_p_value',
                    'box_m_log_log_slope_p_value',

                    'box_t_best_type',
                    'box_t_best_R2',
                    'box_t_lin_lin_R2',
                    'box_t_lin_lin_intercept',
                    'box_t_lin_lin_slope',
                    'box_t_lin_lin_intercept_std',
                    'box_t_lin_lin_slope_std',
                    'box_t_lin_lin_intercept_p_value',
                    'box_t_lin_lin_slope_p_value',
                    'box_t_lin_log_R2',
                    'box_t_lin_log_intercept',
                    'box_t_lin_log_slope',
                    'box_t_lin_log_intercept_std',
                    'box_t_lin_log_slope_std',
                    'box_t_lin_log_intercept_p_value',
                    'box_t_lin_log_slope_p_value',
                    'box_t_log_log_R2',
                    'box_t_log_log_intercept',
                    'box_t_log_log_slope',
                    'box_t_log_log_intercept_std',
                    'box_t_log_log_slope_std',
                    'box_t_log_log_intercept_p_value',
                    'box_t_log_log_slope_p_value',
                ]

            elif config.experiment.method == Method.cluster:

                metrics = [
                    'item',
                    'aux',
                    'number_of_clusters',
                    'number_of_noise_points',
                    'percent_of_noise_points',
                ]

            elif config.experiment.method == Method.polygon:

                if config.experiment.method_params['method'] == Method.linreg:

                    metrics = [
                        'item',
                        'aux',
                        'area_intersection',
                        'slope_intersection',
                        'max_abs_slope'
                    ]

                elif config.experiment.method_params['method'] == Method.variance:

                    metrics = [
                        'item',
                        'aux',
                        'area_intersection',
                        'increasing',
                        'increasing_id',
                        'begin_rel',
                        'end_rel'
                    ]

            elif config.experiment.method == Method.special:

                metrics = [
                    'item'
                ]

            elif config.experiment.method == Method.z_test_linreg:

                metrics = [
                    'item',
                    'aux',
                    'z_value',
                    'p_value',
                    'abs_z_value'
                ]

            elif config.experiment.method == Method.aggregator:

                metrics = [
                    'item',
                    'aux'
                ]

        elif config.experiment.task == Task.clock:

            if config.experiment.method == Method.linreg:
                metrics = [
                    'item',
                    'aux',
                    'R2',
                    'r',
                    'evs',
                    'mae',
                    'rmse',
                ]

    metrics = [f'{m}_{config.hash[0:8]}' if m not in ['item', 'aux'] else m for m in metrics]

    return metrics
