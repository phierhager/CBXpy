import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from result import ExperimentResult
import sys
import seaborn.objects as so
import csv
import typing
# # Visualize the DataFrame
# def plot_bar_plot_comparing_dynamics_to_best_value(df: pd.DataFrame):
#     dfm = pd.melt(df, id_vars="class", var_name="sex", value_name="survival rate")
#     sns.set_theme(style="whitegrid")
#     plt.figure(figsize=(10, 6))
#     sns.barplot(
#         data=df, x="Dynamic Name", y="Relative Function Value", hue="Experiment Name"
#     )
#     plt.title("Comparison of Experiment Results")
#     plt.xlabel("Dynamic Name")
#     plt.ylabel("Best Function Value")
#     plt.show()


def plot_bar_plot_comparing_dynamics_to_rel_value(df: pd.DataFrame):
    dfm = df.melt(
        id_vars=["Function Name", "Dynamic Name"],
        value_vars="Relative Function Value",
        var_name="variable",
        value_name="value",
    )
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=dfm, x="Function Name", y="value", hue="Dynamic Name")
    plt.title(
        f"Dynamics on different Functions (M={df.iloc[0].M}, N={df.iloc[0].N}, d={df.iloc[0].d})"
    )
    plt.xlabel("Function Name")
    plt.ylabel("Relative Function Value (1 is best)")
    plt.show()


def plot_N(df: pd.DataFrame):
    df["Success Probability"] = df["Success"].apply(np.mean)
    dfm = df.melt(
        id_vars=["Dynamic Name", "Function Name", "N"],
        value_vars=["Success Probability"],
        var_name="variable",
        value_name="success_prob",
    )
    sns.set_theme(style="white")
    sns.lineplot(data=dfm, x="N", y="success_prob", hue="Dynamic Name", marker="o")
    d = df.iloc[0].loc["d"]
    max_it = df.iloc[0].loc["max_it"]
    alpha = df.iloc[0].loc["alpha"]
    dt = df.iloc[0].loc["dt"]
    f = df.iloc[0].loc["Function Name"]
    sigma = df.iloc[0].loc["sigma"]
    plt.title("Success Probability for d={}, max_it={}, alpha={}, dt={}, sigma={}, and {}".format(d, max_it, alpha, dt, sigma, f))
    plt.xlabel("N")
    plt.ylabel("Success Probability")
    plt.savefig(f"results/images/{df.iloc[0].loc["Experiment Name"]}__plot_n.png")
    
def plot_sigma(df: pd.DataFrame):
    df["Success Probability"] = df["Success"].apply(np.mean)
    dfm = df.melt(
        id_vars=["Dynamic Name", "Function Name", "sigma"],
        value_vars=["Success Probability"],
        var_name="variable",
        value_name="success_prob",
    )
    sns.set_theme(style="white")
    sns.lineplot(data=dfm, x="sigma", y="success_prob", hue="Dynamic Name", marker="o")
    # get the parameters from the configuration
    d = df.iloc[0].loc["d"]
    max_it = df.iloc[0].loc["max_it"]
    alpha = df.iloc[0].loc["alpha"]
    dt = df.iloc[0].loc["dt"]
    f = df.iloc[0].loc["Function Name"]
    N = df.iloc[0].loc["N"]
    plt.title("Success Probability for d={}, max_it={}, alpha={}, dt={}, N={}, and {}".format(d, max_it, alpha, dt, N, f))
    plt.xlabel("sigma")
    plt.ylabel("Success Probability")
    plt.tight_layout()
    plt.savefig(f"results/images/{df.iloc[0].loc["Experiment Name"]}__plot_sigma.png")


def plot_2d_sigma_lamda(df: pd.DataFrame):
    df["Success Probability"] = df["Success"].apply(np.mean)
    dfm = df.melt(
        id_vars=["Dynamic Name", "sigma", "lamda", "Function Name"],
        value_vars=["Success Probability"],
        var_name="variable",
        value_name="success_prob",
    )
    sns.set_theme(style="white")

    # # Example data (replace this with your DataFrame)
    # data = {
    #     "sigma": [0, 1, 2, 3, 4],
    #     "lamda": [0, 1, 2, 3, 4],
    #     "success_prob": [0.5, 0.6, 0.7, 0.8, 0.9],
    # }
    # df = pd.DataFrame(data)

    # # Create a grid for the data
    sigma_values = sorted(dfm["sigma"].unique())
    lamda_values = sorted(dfm["lamda"].unique())
    dynamic_names = sorted(dfm["Dynamic Name"].unique())
    function_names = sorted(dfm["Function Name"].unique())
    success_prob_matrix = np.zeros(
        (len(dynamic_names), len(function_names), len(lamda_values), len(sigma_values))
    )

    # Fill the success probability matrix
    for index, row in dfm.iterrows():
        sigma_index = sigma_values.index(row["sigma"])
        lamda_index = lamda_values.index(row["lamda"])
        dynamic_index = dynamic_names.index(row["Dynamic Name"])
        function_index = function_names.index(row["Function Name"])
        success_prob_matrix[dynamic_index, function_index, lamda_index, sigma_index] = (
            row["success_prob"]
        )

    # Determine the number of dynamic names
    num_dynamic_names = len(dynamic_names)
    num_function_names = len(function_names)

    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Success Probabilities for different Dynamics")

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=num_dynamic_names, ncols=1)

    # fill subfigs
    for row_index, subfig in enumerate(subfigs):
        dynamic_name = dynamic_names[row_index]
        dynamic_index = dynamic_names.index(dynamic_name)
        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=num_function_names, sharey=True)
        for col_index, ax in enumerate(axs):
            ax.plot()
            ax.set_title("Function ...")

            function_name = function_names[col_index]
            function_index = function_names.index(function_name)

            # Get the success probability matrix for the current dynamic name
            success_prob_matrix_dynamic = success_prob_matrix[
                dynamic_index, function_index
            ]

            # Plot the color map in the appropriate subplot
            im = ax.imshow(
                success_prob_matrix_dynamic,
                cmap="viridis",
                origin="lower",
                extent=[0, len(sigma_values), 0, len(lamda_values)],
            )
        subfig.colorbar(im, shrink=0.6, ax=axs)
        subfig.suptitle(f"{dynamic_name}")

    # Adjust layout to make space for the colorbar on the left
    # plt.subplots_adjust(right=2, left=1)

    # # Add colorbar
    # cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.4])  # adjust position as needed
    # fig.colorbar(im, cax=cbar_ax)
    # plt.tight_layout()

    # fig.colorbar(im, ax=[subfigs[:]], location="left")
    plt.show()

    # cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
    # fig.colorbar(im, cax=cbar_ax)
    # plt.show()

    # # Set up subplots
    # fig, axes = plt.subplots(
    #     num_function_names,
    #     num_dynamic_names,
    #     figsize=(5 * num_dynamic_names, 5 * num_function_names),
    #     sharex=True,
    #     sharey=True,
    # )

    # # Loop through each dynamic name
    # for i, function_name in enumerate(function_names):
    #     function_index = function_names.index(function_name)
    #     for j, dynamic_name in enumerate(dynamic_names):
    #         # Find the index of the dynamic name
    #         dynamic_index = dynamic_names.index(dynamic_name)

    #         # Get the success probability matrix for the current dynamic name
    #         success_prob_matrix_dynamic = success_prob_matrix[
    #             dynamic_index, function_index
    #         ]

    #         # Plot the color map in the appropriate subplot
    #         ax = axes[i, j]
    #         im = ax.imshow(
    #             success_prob_matrix_dynamic,
    #             cmap="viridis",
    #             origin="lower",
    #             extent=[0, len(sigma_values), 0, len(lamda_values)],
    #         )

    #         # col.colorbar(label="Success Probability")
    #         # col.xlabel("Sigma")
    #         # col.ylabel("lamda")
    #         # col.title(f"Success Probability for {dynamic_name}")
    #         # col.xticks(np.arange(len(sigma_values)), sigma_values)
    #         # col.yticks(np.arange(len(lamda_values)), lamda_values)
    #         # col.grid(False)

    #         # Set TICKS; labels and title
    #         ax.set_xticks(np.arange(len(sigma_values)) + 0.5)
    #         ax.set_xticklabels(sigma_values)
    #         ax.set_yticks(np.arange(len(lamda_values)) + 0.5)
    #         ax.set_yticklabels(lamda_values)
    #         # ax.set_xlabel("Sigma")
    #         # ax.set_ylabel("Lambda")
    #         ax.set_title(f"{dynamic_name}")

    # # Adjust layout
    # plt.tight_layout()

    # plt.title(f"Success Probabilities for different dynamics")

    # # Show plot
    # plt.show()


def plot_bar_plot_comparing_dynamics_to_time(df: pd.DataFrame):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df, x="Dynamic Name", y="Best Function Value", hue="Experiment Name"
    )
    plt.title("Comparison of Experiment Results")
    plt.xlabel("Dynamic Name")
    plt.ylabel("Best Function Value")
    plt.show()


def fit_gaussian_exp(x_data, y_data):
    from scipy.optimize import curve_fit

    # Define the fitting function
    def fit_func(x, A, mu, sigma1, B, sigma2):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma1**2)) + B * np.exp(-x / sigma2)

    # Initial guesses for the parameters
    initial_guesses = [1, 5, 1, 0.5, 10]

    # Perform the curve fitting
    popt, pcov = curve_fit(fit_func, x_data, y_data, p0=initial_guesses)

    # Get the fitted values
    fitted_y = fit_func(x_data, *popt)

    return fitted_y


def plot_sigma_N_success(df: pd.DataFrame):
    """Plot the success probability against sigma for different N values.

    Args:
        df (pd.DataFrame): The DataFrame containing the experiment results.
    """
    df["Success Probability"] = df["Success"].apply(np.mean)
    dfm = df.melt(
        id_vars=["Dynamic Name", "sigma", "Function Name", "N"],
        value_vars=["Success Probability"],
        var_name="variable",
        value_name="success_prob",
    )

    # Plot the original data
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7))
    so.Plot(dfm, x="sigma", y="success_prob", color="Dynamic Name").add(
        so.Dots(), pointsize="N", marker="Function Name"
    ).show()

    # dynamic_name = "cbo"
    # function_name = "ackley"
    # subset = dfm[
    #     (dfm["Dynamic Name"] == dynamic_name) & (dfm["Function Name"] == function_name)
    # ]

    # x_data_qcbo = subset["sigma"].values
    # y_data_qcbo = subset["success_prob"].values

    # Generate a smooth curve for plotting the fitted function
    # fitted_y = fit_gaussian_exp(x_data_qcbo, y_data_qcbo)

    # sns.set_theme(style="whitegrid")
    # plt.figure(figsize=(14, 7))
    # sns.scatterplot(
    #     data=dfm,
    #     x="sigma",
    #     y="success_prob",
    #     hue="Dynamic Name",
    #     size="N",
    #     style="Function Name",
    #     alpha=0.5,
    # )
    # plt.plot(
    #     x_data_qcbo,
    #     fitted_y,
    #     color="red",
    #     label=f'Fitted Function ({"qcbo"}, {"ackley"})',
    # )
    # plt.xlabel("sigma")
    # plt.ylabel("success_prob")
    # plt.legend()
    # plt.show()


def plot_sigma_N_success_heat(df: pd.DataFrame, dynamic_name="cbo"):
    """Plot the success probability against sigma for different N values."""
    df["Success Probability"] = df["Success"].apply(np.mean)
    df = df[df["Dynamic Name"] == dynamic_name]
    dfm = df.melt(
        id_vars=["Dynamic Name", "sigma", "Function Name", "N"],
        value_vars=["Success Probability"],
        var_name="variable",
        value_name="success_prob",
    ).pivot(index="N", columns="sigma", values="success_prob")

    # Plot the original data
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7))
    sns.heatmap(dfm)  # , annot=True, cmap="viridis", fmt=".2f")
    plt.title(
        f"Success Probability for {dynamic_name} and Ackley at d=2, max_it=100, alpha=100, dt=0.1"
    )
    plt.show()


def plot_sigma_lamda_success_heat_dynamics_with_overlay(df: pd.DataFrame):
    """Plot the success probability against sigma for cbo and qcbo dynamic names."""
    # Calculate success probability
    df["Success Probability"] = df["Success"].apply(np.mean)

    # Prepare the data for cbo
    df_cbo = df[df["Dynamic Name"] == "cbo"]
    dfm_cbo = df_cbo.melt(
        id_vars=["Dynamic Name", "sigma", "Function Name", "lamda"],
        value_vars=["Success Probability"],
        var_name="variable",
        value_name="success_prob",
    ).pivot(index="lamda", columns="sigma", values="success_prob")

    # Prepare the data for qcbo
    df_qcbo = df[df["Dynamic Name"] == "qcbo"]
    dfm_qcbo = df_qcbo.melt(
        id_vars=["Dynamic Name", "sigma", "Function Name", "lamda"],
        value_vars=["Success Probability"],
        var_name="variable",
        value_name="success_prob",
    ).pivot(index="lamda", columns="sigma", values="success_prob")

    # Define the quadratic function
    def quadratic_function(x, a=0.8):
        return a * x**2

    def exp_function(x, a=0.2):
        return a * np.exp(x)

    def log_gaussian(x, a=20):
        return a * np.log(0.3 * x) * np.exp(-0.1 * x)

    # Plot the data side by side
    x_size = 20
    y_size = 7
    sns.set_theme(style="whitegrid")
    _, axes = plt.subplots(1, 2, figsize=(x_size, y_size), sharey=True)

    # set parameter for functions
    a_qu = 0.8
    a_exp = 0.2

    # Plot heatmap and quadratic function for cbo
    sns.heatmap(dfm_cbo, ax=axes[0], cbar=True)
    axes[0].set_title(
        "Success Probability for cbo and Ackley at d=2, max_it=500, alpha=100, dt=0.1, N=100"
    )
    x_cbo = np.linspace(0, x_size, 100)
    y_cbo_qu = quadratic_function(x_cbo, a_qu)
    y_cbo_exp = exp_function(x_cbo, a_exp)
    axes[0].plot(
        x_cbo,
        y_cbo_qu,
        color="red",
        linewidth=2,
        label=f"Quadratic Function, factor {a_qu}",
    )
    axes[0].plot(
        x_cbo,
        y_cbo_exp,
        color="blue",
        linewidth=2,
        label=f"Exponential Function, factor {a_exp}",
    )
    axes[0].legend()

    # Plot heatmap and quadratic function for qcbo
    sns.heatmap(dfm_qcbo, ax=axes[1], cbar=True)
    axes[1].set_title(
        "Success Probability for qcbo and Ackley at d=2, max_it=500, alpha=100, dt=0.1, N=100"
    )
    x_qcbo = np.linspace(0, x_size, 100)
    y_qcbo_qu = quadratic_function(x_qcbo, a_qu)
    y_qcbo_exp = exp_function(x_qcbo, a_exp)
    axes[1].plot(
        x_cbo,
        y_qcbo_qu,
        color="red",
        linewidth=2,
        label=f"Quadratic Function, factor {a_qu}",
    )
    axes[1].plot(
        x_cbo,
        y_qcbo_exp,
        color="blue",
        linewidth=2,
        label=f"Exponential Function, factor {a_exp}",
    )
    axes[1].legend()

    plt.show()


def plot_sigma_lamda_success_heat_dynamics(df: pd.DataFrame):
    """Plot the success probability against sigma for cbo and qcbo dynamic names."""
    # Calculate success probability
    df["Success Probability"] = df["Success"].apply(np.mean)

    # Prepare the data for cbo
    df_cbo = df[df["Dynamic Name"] == "cbo"]
    dfm_cbo = df_cbo.melt(
        id_vars=["Dynamic Name", "sigma", "Function Name", "lamda"],
        value_vars=["Success Probability"],
        var_name="variable",
        value_name="success_prob",
    ).pivot(index="lamda", columns="sigma", values="success_prob")

    # Prepare the data for qcbo
    df_qcbo = df[df["Dynamic Name"] == "qcbo"]
    dfm_qcbo = df_qcbo.melt(
        id_vars=["Dynamic Name", "sigma", "Function Name", "lamda"],
        value_vars=["Success Probability"],
        var_name="variable",
        value_name="success_prob",
    ).pivot(index="lamda", columns="sigma", values="success_prob")

    # get the parameters from the configuration
    d = df.iloc[0].loc["d"]
    max_it = df.iloc[0].loc["max_it"]
    alpha = df.iloc[0].loc["alpha"]
    dt = df.iloc[0].loc["dt"]
    f = df.iloc[0].loc["Function Name"]

    # Plot the data side by side
    sns.set_theme(style="whitegrid")
    _, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)

    sns.heatmap(dfm_cbo, ax=axes[0], cbar=True)
    axes[0].set_title(
        f"Success Probability for cbo and {f} at d={d}, max_it={max_it}, alpha={alpha}, dt={dt}"
    )
    axes[0].invert_yaxis()
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=45)

    sns.heatmap(dfm_qcbo, ax=axes[1], cbar=True)
    axes[1].set_title(
        f"Success Probability for qcbo and {f} at d={d}, max_it={max_it}, alpha={alpha}, dt={dt}"
    )
    axes[1].invert_yaxis()
    plt.tight_layout()
    
    plt.savefig(f"results/images/{df.iloc[0].loc["Experiment Name"]}__sigma_lamda_success_heat_dynamics.png")


def plot_sigma_N_success_heat_dynamics(df: pd.DataFrame):
    """Plot the success probability against sigma for cbo and qcbo dynamic names."""
    # Calculate success probability
    df["Success Probability"] = df["Success"].apply(np.mean)

    # Prepare the data for cbo
    df_cbo = df[df["Dynamic Name"] == "cbo"]
    dfm_cbo = df_cbo.melt(
        id_vars=["Dynamic Name", "sigma", "Function Name", "N"],
        value_vars=["Success Probability"],
        var_name="variable",
        value_name="success_prob",
    ).pivot(index="N", columns="sigma", values="success_prob")

    # Prepare the data for qcbo
    df_qcbo = df[df["Dynamic Name"] == "qcbo"]
    dfm_qcbo = df_qcbo.melt(
        id_vars=["Dynamic Name", "sigma", "Function Name", "N"],
        value_vars=["Success Probability"],
        var_name="variable",
        value_name="success_prob",
    ).pivot(index="N", columns="sigma", values="success_prob")

    # Plot the data side by side
    sns.set_theme(style="whitegrid")
    _, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)

    # get the parameters from the configuration
    d = df.iloc[0].loc["d"]
    max_it = df.iloc[0].loc["max_it"]
    alpha = df.iloc[0].loc["alpha"]
    dt = df.iloc[0].loc["dt"]
    f = df.iloc[0].loc["Function Name"]

    sns.heatmap(dfm_cbo, ax=axes[0], cbar=True)
    axes[0].set_title(
        f"Success Probability for cbo and {f} at d={d}, max_it={max_it}, alpha={alpha}, dt={dt}"
    )
    axes[0].invert_yaxis()
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=45)

    sns.heatmap(dfm_qcbo, ax=axes[1], cbar=True)
    axes[1].set_title(
        f"Success Probability for qcbo and {f} at d={d}, max_it={max_it}, alpha={alpha}, dt={dt}"
    )
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f"results/images/{df.iloc[0].loc["Experiment Name"]}__sigma_N_success_heat_dynamics.png")


MODE = typing.Literal["sigma_lamda", "sigma_N"]
ALPHA_D = typing.Literal["alpha", "d"]
def plot_sigma_success_heat_subplots(df: pd.DataFrame, mode: MODE = "sigma_lamda", alpha_d: ALPHA_D = "alpha", alpha_values = [1, 10, 100, 1000, 10000], d_values = [2, 5, 10]):
    """Plot the success probability against sigma for cbo and qcbo dynamic names."""
    # Calculate success probability
    df["Success Probability"] = df["Success"].apply(np.mean)
    if mode == "sigma_lamda":
        melt_vars = ["Dynamic Name", "sigma", "Function Name", "lamda"]
        pivot_vars = ["lamda", "sigma"]
    elif mode == "sigma_N":
        melt_vars = ["Dynamic Name", "sigma", "Function Name", "N"]
        pivot_vars = ["N", "sigma"]

    dfs = {}
    for dynamic in ["cbo", "qcbo"]:
        if alpha_d == "alpha":
            alpha_df = {}
            for alpha in alpha_values:
                df_alpha = df[(df["Dynamic Name"] == dynamic) & (df["alpha"] == alpha)]
                dfm_alpha = df_alpha.melt(
                    id_vars=melt_vars,
                    value_vars=["Success Probability"],
                    var_name="variable",
                    value_name="success_prob",
                ).pivot(index=pivot_vars[0], columns=pivot_vars[1], values="success_prob")
                alpha_df[f"{alpha}"] = dfm_alpha
            dfs[dynamic] = alpha_df
        elif alpha_d == "d":
            d_df = {}
            for d in d_values:
                df_d = df[(df["Dynamic Name"] == dynamic) & (df["d"] == d)]
                dfm_d = df_d.melt(
                    id_vars=melt_vars,
                    value_vars=["Success Probability"],
                    var_name="variable",
                    value_name="success_prob",
                ).pivot(index=pivot_vars[0], columns=pivot_vars[1], values="success_prob")
                d_df[f"{d}"] = dfm_d
            dfs[dynamic] = d_df

    # Plot the data side by side
    sns.set_theme(style="whitegrid")
    num_rows = len(alpha_values) if alpha_d == "alpha" else len(d_values)
    fig, axes = plt.subplots(num_rows, 2, figsize=(20, 30), sharey=True, sharex=True)
    
    d = df.iloc[0].loc["d"]
    max_it = df.iloc[0].loc["max_it"]
    dt = df.iloc[0].loc["dt"]
    f = df.iloc[0].loc["Function Name"]
    title_str = f"{f}, max_it={max_it}, dt={dt}"
    if mode == "sigma_N":
        lamda = df.iloc[0].loc["lamda"]
        title_str += f", lamda={lamda}"
    else:
        n = df.iloc[0].loc["N"]
        title_str += f", N={n}"
    

    for i, dynamic in enumerate(["cbo", "qcbo"]):
        if alpha_d == "alpha":
            for j, alpha in enumerate(alpha_values):
                sns.heatmap(dfs[dynamic][f"{alpha}"], ax=axes[j][i], cbar=True)
                d = df.iloc[0].loc["d"]
                axes[j][i].set_title(f"Success Probability for {dynamic}, alpha={alpha}, and d={d}, "+ title_str)
                axes[j][i].invert_yaxis()
        elif alpha_d == "d":
            for j, d in enumerate(d_values):
                sns.heatmap(dfs[dynamic][f"{d}"], ax=axes[j][i], cbar=True)
                alpha = df.iloc[0].loc["alpha"]
                axes[j][i].set_title(f"Success Probability for {dynamic}, alpha={alpha}, and d={d}, "+ title_str)
                axes[j][i].invert_yaxis()
            

    plt.subplots_adjust(hspace=0.5)
    for ax in axes.flat:
        ax.tick_params(axis="y", labelrotation=45)
        ax.tick_params(axis="x", labelsize=8, labelrotation=45)
    
    plt.tight_layout()
    plt.savefig(f"results/images/{df.iloc[0].loc["Experiment Name"]}__plot_sigma_lamda_success_heat_subplots.png")


if __name__ == "__main__":
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)

    experiment_result = ExperimentResult.from_csv(sys.argv[1])
    df = experiment_result.to_dataframe()

    print(df.columns)

    # concatenate the dataframes
    if len(sys.argv) >= 3:
        for i in range(2, len(sys.argv) - 1):
            experiment_result = ExperimentResult.from_csv(sys.argv[i])
            df_extra = experiment_result.to_dataframe()
            df = pd.concat(df, df_extra, ignore_index=True)
    
    # plot_sigma_success_heat_subplots(df, mode="sigma_N", alpha_d="d", d_values=[10,15,20])	
    plot_N(df)
    # plot_sigma(df)

    # plot_sigma_N_success_heat_dynamics(df)

    # plot_sigma_lamda_success_heat_dynamics(df)
    # plot_sigma_N_success_heat_dynamics(df)

    # plot_sigma_lamda_success_heat_dynamics_with_overlay(df)

    # df["Success Probability"] = df["Success"].apply(np.mean)
    # dfm = df.melt(
    #     id_vars=["Dynamic Name", "sigma", "Function Name", "N"],
    #     value_vars=["Success Probability"],
    #     var_name="variable",
    #     value_name="success_prob",
    # )

    # print(dfm.head())

    # print(df.columns)
    # print(df)
    # plot_2d_sigma_lamda(df)

    # dfm = df.melt(
    #     id_vars="Dynamic Name",
    #     value_vars=["Time", "Best Function Value"],
    #     var_name="class",
    #     value_name="value",
    # )

    # sns.catplot(
    #     x="class",
    #     y="value",
    #     hue="Dynamic Name",
    #     data=dfm,
    #     kind="bar",
    #     height=5,
    #     aspect=1,
    # )
    # plt.show()

    # plot_bar_plot_comparing_dynamics_to_best_value(df)

    # # Time Series Analysis: Time vs. Experiment Name
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(data=df, x="Time", y="Experiment Name", hue="Dynamic Name", marker="o")
    # plt.title("Time vs. Experiment Name")
    # plt.xlabel("Time")
    # plt.ylabel("Experiment Name")
    # plt.show()

    # # Distribution Analysis: Best Function Value
    # plt.figure(figsize=(10, 6))
    # sns.histplot(data=df, x="Best Function Value", bins=20, kde=True)
    # plt.title("Distribution of Best Function Values")
    # plt.xlabel("Best Function Value")
    # plt.ylabel("Frequency")
    # plt.show()

    # # Correlation Analysis: Heatmap of Correlation Matrix
    # numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    # corr_matrix = df[numeric_cols].corr()
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    # plt.title("Correlation Matrix")
    # plt.show()
