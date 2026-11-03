#%%
import os
import itertools
import pandas as pd
import numpy as np
import random
import string
import scipy.stats as stats
import pingouin as pg
import scikit_posthocs as sp
from statsmodels.stats import multicomp
from typing import Any, Callable, Dict, List, Optional, Tuple


def generate_random_dataframe(
    df_size: int,
    var_str_num: List[int],
    var_num_num: int = 1,
    var_pair: bool = False,
) -> pd.DataFrame:
    """
    Generate a custom random dataframe based on input parameters.

    Parameters:
    df_size (int): The number of rows in the dataframe.
    var_str_num (list): A list containing the number of unique values for each string column.
    var_num_num (int): The number of numeric columns in the dataframe.
    var_pair (bool): Whether to include a pair column for the first string column.

    Returns:
    pd.DataFrame: The generated dataframe.
    """
    if len(var_str_num) < 1 or any(v < 1 for v in var_str_num) or var_num_num < 1:
        raise ValueError("Input parameters are not valid. Ensure var_str_num length >= 1, values >= 1, and var_num_num >= 1.")

    # Initialize an empty dataframe
    df = pd.DataFrame()

    # Truncate dataframe size if pair column is required
    if var_pair:
        trunc_size = (df_size // var_str_num[0]) * var_str_num[0]
        df_size = trunc_size

        # Generate column A with equal sample sizes for each unique value
        unique_values = [string.ascii_lowercase[j] for j in range(var_str_num[0])]
        df['A'] = [val for val in unique_values for _ in range(df_size // var_str_num[0])]

        # Shuffle the values in column A
        random.shuffle(df['A'])

        # Assign pair values
        df['pair'] = df.groupby('A', observed=True).cumcount() + 1
    else:
        # Generate string columns with random sample sizes
        for i, num_values in enumerate(var_str_num):
            col_name = string.ascii_uppercase[i]
            unique_values = [string.ascii_lowercase[j] for j in range(num_values)]
            df[col_name] = random.choices(unique_values, k=df_size)

    # Generate other string columns if var_pair is True
    if var_pair:
        for i in range(1, len(var_str_num)):
            col_name = string.ascii_uppercase[i]
            unique_values = [string.ascii_lowercase[j] for j in range(var_str_num[i])]
            df[col_name] = random.choices(unique_values, k=df_size)

    # Generate numeric columns
    for i in range(var_num_num):
        col_name = f"value_{i + 1}"
        df[col_name] = np.random.random(size=df_size)

    return df


def normality_test(
    df: pd.DataFrame,
    var: str,
    value: str,
    p: float = 0.05,
) -> Tuple[bool, List[Tuple[Any, bool]]]:
    """
    Perform Shapiro-Wilk normality test for each group in the dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data to be tested.
    var (str): The variable name to group by.
    value (str): The column name on which to perform the normality test.
    p (float): The significance level for the normality test.

    Returns:
    tuple:
        - all_groups_normal (bool): True if all groups are normally distributed, False otherwise.
        - normality_results (list of tuples): A list containing tuples of group names and whether they passed the normality test.
    """
    # Group the dataframe by the specified variable
    grouped = df.groupby(var,observed=True)

    normality_results = []
    all_groups_normal = True

    # Perform the Shapiro-Wilk normality test for each group
    for group_name, group_data in grouped:
        # Extract the specified column for the group to avoid conversion issues
        numeric_data = group_data[value].dropna()
        stat, p_value = stats.shapiro(numeric_data)
        # If p-value is less than p, we reject the null hypothesis that data is normal
        is_normal = p_value > p
        normality_results.append((group_name, is_normal))
        if not is_normal:
            all_groups_normal = False

    # Return whether all groups are normal and the list of results
    return all_groups_normal, normality_results


def equal_variances_test(
    df: pd.DataFrame,
    var: str,
    value: str,
    p: float = 0.1,
) -> Tuple[bool, Dict[str, float]]:
    """
    Perform test for equal variances for data grouped by a variable.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data to be tested.
    var (str): The variable name to group by.
    value (str): The column name on which to perform the equal variances test.
    p (float): The significance level for the equal variances test.

    Returns:
    dict: A dictionary containing the group names and their corresponding p-value from Bartlett's test.
    """
    groups = [group[value].dropna().values for name, group in df.groupby(var,observed=True)]
    stat, p_value = stats.bartlett(*groups)
    results = {'statistic': stat, 'p_value': p_value}
    significance = p_value > p
    return significance, results



def mc_pairwise_statistical_tests(
    df: pd.DataFrame,
    var: str,
    value: str,
    method: Callable[[pd.Series, pd.Series], Tuple[float, float]],
    var_pair: Optional[str] = None,
    method_mc: str = "bonf",
    p: float = 0.05,
) -> pd.DataFrame:
    """
    Perform pairwise statistical tests and correct p-values for multiple comparisons.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data to be tested.
    var (list): The list of variables to group by.
    value (str): The column name on which to perform the statistical test.
    method (function): The statistical test function to use (e.g., stats.ttest_ind).
    var_pair (str): Optional, a column to indicate pair matching.
    method_mc (str): The multiple comparisons method (e.g., 'Tukey').
        Method used for testing and adjustment of p-values. Can be either the full name or initial letters. Available methods are:
        'bonf': one-step Bonferroni correction
        'sidak': one-step Sidak correction
        'holm': step-down method using Bonferroni adjustments
        'fdr_bh': Benjamini/Hochberg FDR correction
        'fdr_by': Benjamini/Yekutieli FDR correction
        'none': pass-through option (no correction applied)
    p (float): The significance level for the tests.

    Returns:
    pd.DataFrame: A dataframe containing the pairwise comparison results.
    reject (array, boolean): True for hypothesis that can be rejected for given alpha.
    pvals_corrected (array): P-values corrected for multiple testing.
    """
    # Get all combinations of unique values for each variable in var
    combinations = list(itertools.combinations(df.groupby(var,observed=True).groups.keys(), 2))

    mc_results = {
        'A': [],
        'B': [],
        'mean(A)': [],
        'mean(B)': [],
        'diff': [],
        'pval': []
    }

    # Perform pairwise comparisons for all combinations
    for comb in combinations:
        group_a = df[df[var] == comb[0]][value]
        group_b = df[df[var] == comb[1]][value]

        # Perform the statistical test
        if var_pair:
            df0 = df.set_index(var_pair)
            group_a = df0.loc[(df0[var] == comb[0]), value].dropna()
            group_b = df0.loc[(df0[var] == comb[1]), value].dropna()
            group_a = group_a.sort_index(ascending=True)
            group_b = group_b.sort_index(ascending=True)
            stat, pval = method(group_a, group_b)
        else:
            stat, pval = method(group_a, group_b)

        # Store the results
        mc_results['A'].append(comb[0])
        mc_results['B'].append(comb[1])
        mc_results['mean(A)'].append(group_a.mean())
        mc_results['mean(B)'].append(group_b.mean())
        mc_results['diff'].append(group_a.mean() - group_b.mean())
        mc_results['pval'].append(pval)

    # Correct p-values for multiple comparisons
    pvals = mc_results['pval']
    reject, pvals_corrected = pg.multicomp(pvals, method=method_mc, alpha=p)

    # Update corrected p-values in results
    mc_results['pval_corrected'] = pvals_corrected
    mc_results['reject'] = reject

    # Create a dataframe from the results
    results_df = pd.DataFrame(mc_results)

    return results_df



def single_var_stat_test(
    df: pd.DataFrame,
    var: str,
    value: str,
    p: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform statistical tests on a single variable grouped by a categorical variable.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data to be tested.
    var (str): The variable name to group by.
    value (str): The column name on which to perform the statistical tests.
    p (float): The significance level for the normality test.

    Returns:
    dict: A dictionary containing the results of the statistical tests performed.
    """
    # Perform normality and equal variances tests
    all_groups_normal, normality_results = normality_test(df, var, value, p)
    equal_var_significance, equal_var_result = equal_variances_test(df, var, value, p)
    equal_var_p_value = equal_var_result['p_value']

    # Determine the number of unique groups
    unique_groups = df[var].nunique()
    if unique_groups < 2:
        raise ValueError("The variable must have at least 2 unique groups for statistical testing.")

    results = {}

    if unique_groups == 2:
        if all_groups_normal:
            if equal_var_significance:
                # Perform t-test
                stat, p_value = stats.ttest_ind(*[group[value].dropna() for name, group in df.groupby(var,observed=True)])
                results['test'] = 't-test'
                results['statistic'] = stat
                results['p_value'] = p_value
            else:
                # Perform Welch's t-test
                stat, p_value = stats.ttest_ind(*[group[value].dropna() for name, group in df.groupby(var,observed=True)], equal_var=False)
                results['test'] = "'Welch's t-test'"
                results['statistic'] = stat
                results['p_value'] = p_value
        else:
            # Perform Wilcoxon rank-sum test
            # stat, p_value = stats.ranksums(*[group[value].dropna() for name, group in df.groupby(var)])

            # Perform Wilcoxon rank-sum test (=Mann-Whitney U Test) using Pingouin
            stat = pg.mwu(*[group[value].dropna() for name, group in df.groupby(var,observed=True)])
            results['test'] = 'Mann-Whitney U Test'
            results['statistic'] = stat['U-val']['MWU']
            results['p_value'] = stat['p-val']['MWU']

    elif unique_groups >= 3:
        if all_groups_normal:
            if equal_var_significance:
                # Perform one-way ANOVA
                # stat, p_value = stats.f_oneway(*[group[value].dropna() for name, group in df.groupby(var)])
                anova = pg.anova(data=df, dv=value, between=var)
                results['test'] = 'One-way ANOVA'
                results['statistic'] = anova['F'][0]
                results['p_value'] = anova['p-unc'][0]
                results['multiple_comparisons'] = 'Tukey HSD'
                # Perform multiple comparisons
                # mc = pg.pairwise_tests(data=df, dv=value, between=var, padjust='bonferroni')
                mc = pg.pairwise_tukey(data=df, dv=value, between=var)
                results['multiple_comparisons_results'] = mc
            else:
                # Perform Welch ANOVA using Pingouin
                welch_anova = pg.welch_anova(data=df, dv=value, between=var)
                stat = welch_anova['F'][0]
                p_value = welch_anova['p-unc'][0]
                results['test'] = 'Welch ANOVA'
                results['statistic'] = stat
                results['p_value'] = p_value
                results['multiple_comparisons'] = 'Games-Howell'
                # Perform multiple comparisons
                mc = pg.pairwise_gameshowell(data=df, dv=value, between=var)
                results['multiple_comparisons_results'] = mc
        else:
            # Perform Kruskal-Wallis test
            # stat, p_value = stats.kruskal(*[group[value].dropna() for name, group in df.groupby(var)])
            stat = pg.kruskal(data=df, dv=value, between=var)
            results['test'] = 'Kruskal-Wallis(inconsistent with Prism?)'
            print(f'warning: multiple comparisons method Kruskal-Wallis is inconsistent with Prism')
            results['statistic'] = stat['H']['Kruskal']
            results['p_value'] = stat['p-unc']['Kruskal']
            results['multiple_comparisons'] = "Dunn's(inconsistent with Prism?)"
            print(f"warning: multiple comparisons method Dunn's is inconsistent with Prism")
            # Perform multiple comparisons
            data = list(df.groupby(var,observed=True)[value])
            data_var = [group[0] for group in data]
            data = [group[1].dropna().values for group in data]
            mc = sp.posthoc_dunn(data, p_adjust='bonferroni')
            mc_results = {'A': [], 'B': [], 'mean(A)': [], 'mean(B)': [], 'diff': [], 'pval': []}
            for i in range(len(data_var)):
                for j in range(i + 1, len(data_var)):
                    mc_results['A'].append(data_var[i])
                    mc_results['B'].append(data_var[j])
                    mc_results['mean(A)'].append(np.mean(data[i]))
                    mc_results['mean(B)'].append(np.mean(data[j]))
                    mc_results['diff'].append(np.mean(data[i]) - np.mean(data[j]))
                    mc_results['pval'].append(mc.iloc[i, j])
            mc = pd.DataFrame(mc_results)
            results['multiple_comparisons_results'] = mc

    return results

def single_var_stat_pair_test(
    df: pd.DataFrame,
    var: str,
    var_pair: str,
    value: str,
    p: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform statistical tests on a single variable grouped by a categorical variable with pair matching.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data to be tested.
    var (str): The variable name to group by.
    var_pair (str): The column name that indicates pair matching.
    value (str): The column name on which to perform the statistical tests.
    p (float): The significance level for the normality test.

    Returns:
    dict: A dictionary containing the results of the statistical tests performed.
    """
    # Perform normality and equal variances tests
    all_groups_normal, normality_results = normality_test(df, var, value, p)
    equal_var_significance, equal_var_result = equal_variances_test(df, var, value, p)
    equal_var_p_value = equal_var_result['p_value']

    # Determine the number of unique groups
    unique_groups = df[var].nunique()
    if unique_groups < 2:
        raise ValueError("The variable must have at least 2 unique groups for statistical testing.")

    results = {}

    if unique_groups == 2:
        if all_groups_normal:
            # Perform paired t-test
            group_data = [group.set_index(var_pair)[value].dropna().sort_index(ascending=True) 
                          for name, group in df.groupby(var,observed=True)]
            stat, p_value = stats.ttest_rel(group_data[0], group_data[1])
            results['test'] = 'paired t-test'
            results['statistic'] = stat
            results['p_value'] = p_value
        else:
            # Perform Wilcoxon signed-rank test
            group_data = [group.set_index(var_pair)[value].dropna().sort_index(ascending=True) 
                          for name, group in df.groupby(var,observed=True)]
            stat, p_value = stats.wilcoxon(group_data[0], group_data[1])
            results['test'] = 'Wilcoxon signed-rank test'
            results['statistic'] = stat
            results['p_value'] = p_value
    elif unique_groups >= 3:
        if all_groups_normal:
            if equal_var_significance:
                # Perform RM one-way ANOVA
                rm_anova = pg.rm_anova(data=df, dv=value, within=var, subject=var_pair, correction='auto', detailed=True)
                stat = rm_anova['F'][0]
                p_value = rm_anova['p-GG-corr'][0] if 'p-GG-corr' in rm_anova else rm_anova['p-unc'][0]
                results['test'] = 'RM one-way ANOVA'
                results['statistic'] = stat
                results['p_value'] = p_value
                results['multiple_comparisons'] = 'Bonferroni(inconsistent with Prism?)'
                print(f'warning: multiple comparisons method Bonferroni is inconsistent with Prism')
                # Perform multiple comparisons with Bonferroni method
                mc = pg.pairwise_tests(data=df, dv=value, within=var, subject=var_pair, parametric=True, padjust='bonf')
                results['multiple_comparisons_results'] = mc
            else:
                # Perform RM one-way ANOVA with Greenhouse-Geisser correction
                rm_anova = pg.rm_anova(data=df, dv=value, within=var, subject=var_pair, correction='auto', detailed=True)
                stat = rm_anova['F'][0]
                p_value = rm_anova['p-GG-corr'][0] if 'p-GG-corr' in rm_anova else rm_anova['p-unc'][0]
                results['test'] = 'RM one-way ANOVA with Greenhouse-Geisser correction'
                results['statistic'] = stat
                results['p_value'] = p_value
                results['multiple_comparisons'] = 'Bonferroni'
                # Perform multiple comparisons with Bonferroni method
                mc = pg.pairwise_tests(data=df, dv=value, within=var, subject=var_pair, parametric=True, padjust='bonf')
                results['multiple_comparisons_results'] = mc
        else:
            # Perform Friedman test
            friedman_result = pg.friedman(data=df, dv=value, within=var, subject=var_pair)
            stat = friedman_result['Q']['Friedman']
            p_value = friedman_result['p-unc']['Friedman']
            results['test'] = 'Friedman test(inconsistent with Prism?)'
            print(f'warning: method Friedman is inconsistent with Prism')
            results['statistic'] = stat
            results['p_value'] = p_value
            results['multiple_comparisons'] = "Dunn's(inconsistent with Prism?)"
            print(f"warning: multiple comparisons method Dunn's is inconsistent with Prism")
            # Perform multiple comparisons with Dunn's method
            data = list(df.groupby(var,observed=True)[value])
            data_var = [group[0] for group in data]
            data = [group[1].dropna().values for group in data]
            mc = sp.posthoc_dunn(data, p_adjust='bonferroni')
            mc_results = {'A': [], 'B': [], 'mean(A)': [], 'mean(B)': [], 'diff': [], 'pval': []}
            for i in range(len(data_var)):
                for j in range(i + 1, len(data_var)):
                    mc_results['A'].append(data_var[i])
                    mc_results['B'].append(data_var[j])
                    mc_results['mean(A)'].append(np.mean(data[i]))
                    mc_results['mean(B)'].append(np.mean(data[j]))
                    mc_results['diff'].append(np.mean(data[i]) - np.mean(data[j]))
                    mc_results['pval'].append(mc.iloc[i, j])
            mc_results = pd.DataFrame(mc_results)
            results['multiple_comparisons_results'] = mc_results

    return results
        
def two_var_stat_test(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    value: str,
    p: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform statistical tests on a variable grouped by two categorical variables.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data to be tested.
    var1 (str): The first variable name to group by.
    var2 (str): The second variable name to group by.
    value (str): The column name on which to perform the statistical tests.
    p (float): The significance level for the statistical test.

    Returns:
    dict: A dictionary containing the results of the statistical tests performed.
    """
    # Perform two-way ANOVA
    aov = pg.anova(data=df, dv=value, between=[var1, var2], detailed=True)
    results = {
        'test': 'Two-way ANOVA',
        'statistic': aov[['F']].values.tolist(),
        'p_value': aov[['p-unc']].values.tolist(),
    }

    # Perform multiple comparisons with Tukey method
    comp = multicomp.MultiComparison(df[value], df[var1] + '_' + df[var2])
    mc = comp.tukeyhsd(alpha=p).summary()
    mc = pd.DataFrame(mc.data[1:], columns=mc.data[0])
    results['multiple_comparisons'] = 'Tukey'
    results['multiple_comparisons_results'] = mc
    return results
