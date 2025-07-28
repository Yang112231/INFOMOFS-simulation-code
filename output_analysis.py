import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import truncnorm
import openpyxl
from simulation import Simulation

strategy_list = [
    'base',
    'price',
    'fcfs',
    'elfs',
    'preempt',
]

scenario_list = [
    'no_solar',
    'p6p7_w',
    'p6p7_s',
    'p1p2p6p7_w',
    'p1p2p6p7_s',
]


# -------- Paired t-test --------
def paired(strategy_A, strategy_B):
    strategy_A = np.array(strategy_A)
    strategy_B = np.array(strategy_B)
    paired_diff = strategy_A - strategy_B
    mean_diff = np.mean(paired_diff)
    se = np.std(paired_diff, ddof=1) / np.sqrt(len(paired_diff))
    margin = stats.t.ppf(0.975, df=len(paired_diff) - 1) * se
    ci_paired = (mean_diff - margin, mean_diff + margin)
    return ci_paired, margin


# -------- Welch's t-test --------
def welch(strategy_A, strategy_B):
    strategy_A = np.array(strategy_A)
    strategy_B = np.array(strategy_B)
    mean_A, mean_B = np.mean(strategy_A), np.mean(strategy_B)
    diff = mean_A - mean_B
    s1, s2 = np.var(strategy_A, ddof=1), np.var(strategy_B, ddof=1)
    se_welch = np.sqrt(s1 / len(strategy_A) + s2 / len(strategy_B))
    df_welch = (s1 / 10 + s2 / 10) ** 2 / ((s1 / 10) ** 2 / 9 + (s2 / 10) ** 2 / 9)
    margin_welch = stats.t.ppf(0.975, df=df_welch) * se_welch
    ci_welch = (diff - margin_welch, diff + margin_welch)
    return ci_welch, margin_welch


def return_dataframe(data):
    columns = {}
    for item in data:
        for key, value in item.items():
            columns[key] = value
    df = pd.DataFrame(columns)
    return df


def run_case(strategy, comparison_parameters, scenario, runs: int, horizon: int, seed: int):
    cable_collect = []
    veh_collect = []
    for run in range(runs):
        sim = Simulation(strategy, scenario, horizon, rng_seed=seed + run)
        sim.advance()
        df_c, df_v = sim.results()
        cable_collect.append(df_c)
        veh_collect.append(df_v)
    # cables = pd.concat(cable_collect)
    vehicles = pd.concat(veh_collect)
    vehicles_mean = pd.concat(veh_collect).agg(['mean'])
    vehicles_std = pd.concat(veh_collect).agg(['std'])
    return vehicles[comparison_parameters], vehicles_mean[comparison_parameters], vehicles_std[comparison_parameters]


# same seed for each run with scenarios
def table_list_mean_std(strategy, comparison_parameters, runs: int, horizon: int, seed: int):
    data = []
    data_mean = []
    data_std = []
    for scenario in scenario_list:
        parameters, parameter_mean, parameter_std = run_case(strategy, comparison_parameters, scenario, runs, horizon,
                                                             seed)
        parameters = parameters.to_string(index=False)
        parameters = [float(x.strip()) for x in parameters.split('\n')]
        parameter_mean = [float(x) for x in parameter_mean]
        data.append({
            scenario: parameters,
        })
        data_mean.append({
            scenario: parameter_mean[0],
        })
        data_std.append({
            scenario: parameter_std.iloc[0],
        })
    df = return_dataframe(data)
    for i, scenario in enumerate(scenario_list):
        df.loc['mean', scenario] = data_mean[i][scenario]
        df.loc['std', scenario] = data_std[i][scenario]
    print(df)
    return data, data_mean


# random seed for each run with scenarios
def w_table_list_mean_std(strategy, comparison_parameters, runs: int, horizon: int, seed: int):
    data = []
    data_mean = []
    data_std = []
    for scenario in scenario_list:
        parameters, parameter_mean, parameter_std = run_case(strategy, comparison_parameters, scenario, runs, horizon,
                                                             seed)
        parameters = parameters.to_string(index=False)
        parameters = [float(x.strip()) for x in parameters.split('\n')]
        parameter_mean = [float(x) for x in parameter_mean]
        data.append({
            scenario: parameters,
        })
        data_mean.append({
            scenario: parameter_mean[0],
        })
        data_std.append({
            scenario: parameter_std.iloc[0],
        })
        seed += 10
    df = return_dataframe(data)
    for i, scenario in enumerate(scenario_list):
        df.loc['mean', scenario] = data_mean[i][scenario]
        df.loc['std', scenario] = data_std[i][scenario]
    print(df)
    return data, data_mean


def p_w_test(data, data_mean, w_data, w_data_mean):
    ray = []
    w_ray = []
    data_analysis = []
    p_interval = []
    p_half_length = []
    w_interval = []
    w_half_length = []
    index = [i for i in range(2, len(scenario_list) + 1)]
    data_analysis.append({
        'i': index,
    })
    for i, scenario in enumerate(scenario_list):
        if i == 0:
            value = data_mean[i][scenario]
            w_value = w_data_mean[i][scenario]
            pw_value = data[i][scenario]
            w_pw_value = w_data[i][scenario]
        else:
            p_interval_value, p_half_value = paired(data[i][scenario], pw_value)
            w_interval_value, w_half_value = welch(w_data[i][scenario], w_pw_value)
            ray.append(np.array(data_mean[i][scenario]) - np.array(value))
            w_ray.append(np.array(w_data_mean[i][scenario]) - np.array(w_value))
            p_interval.append(p_interval_value)
            p_half_length.append(p_half_value)
            w_interval.append(w_interval_value)
            w_half_length.append(w_half_value)
    data_analysis.append({
        'X_i-X_1': [float(num) for num in ray],
        'p_half_length': [float(num) for num in p_half_length],
        'p_interval': [(float(x), float(y)) for x, y in p_interval],
        'X"_i-X"_1': [float(num) for num in w_ray],
        'w_half_length': [float(num) for num in w_half_length],
        'w_interval': [(float(x), float(y)) for x, y in w_interval],
    })
    df = return_dataframe(data_analysis)

    # set pandas display options to show all columns
    pd.set_option('display.max_columns', None)  # show all columns
    pd.set_option('display.expand_frame_repr', False)  # display without line breaks
    print(df)


def run_experiment(strategy, comparison_parameters, runs: int, horizon: int, seed: int):
    # compare different scenarios # standard: strategy + no_solar
    print(
        "\nSimulation results (mean and variance) for {} under {} strategy with same seed:".format(
            comparison_parameters,
            strategy))
    data, data_mean = table_list_mean_std(strategy, comparison_parameters, runs, horizon, seed)  # same seed
    print(
        "\nSimulation results (mean and variance) for {} under {} strategy with random seed:".format(
            comparison_parameters,
            strategy))
    w_data, w_data_mean = w_table_list_mean_std(strategy, comparison_parameters, runs, horizon,
                                                seed + 10)  # random seed
    # paired-t and welch test
    print("\nIndividual 97.5 percent confidence intervals for all comparisons with the standard system ({}+{}):".format(
        strategy, strategy_list[0]))
    p_w_test(data, data_mean, w_data, w_data_mean)


def main():
    # choose 1 strategy(+ no_solar scenario) as a standard system
    strategy = input("strategy:")  # strategy:   base | price | fcfs | elfs | preempt
    # choose 1 parameter to compare different scenarios
    comparison_parameters = input("comparison parameter:")  # non_served(%) | delayed(%)
    runs = 10
    horizon = 30
    seed = 42
    run_experiment(strategy, comparison_parameters, runs, horizon, seed)


if __name__ == '__main__':
    main()
