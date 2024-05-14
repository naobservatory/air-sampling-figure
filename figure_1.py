import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

SPEAKING_DATA_FILE = "data/speaking_2023_bagheri.csv"
BREATHING_DATA_FILE = "data/breathing_2023_bagheri.csv"


def get_cunningham_corr_factor(diameter):
    λ = 6.4e-8  # mean free path
    d = diameter  # particle diameter
    A1, A2, A3 = 1.257, 0.400, 0.55  # experimentally derived coefficients
    corr_factor = 1 + ((2 * λ) / d) * (A1 + A2 * np.exp((-A3 * d) / λ))
    return corr_factor


def get_terminal_velocity(diameter):
    g = 9.81  # m/s^2, gravitation
    p = 997  # kg/m^3, density of water (aerosol)
    e = 1.825e-5  # dynamic viscosity of air at 20°C
    Cc = get_cunningham_corr_factor(diameter)
    terminal_velocity = (diameter**2 * g * p * Cc) / (18 * e)
    return terminal_velocity


def get_settling_time():
    diameters = np.logspace(-7.3, -3.4, num=30)
    residence_times = []

    for diameter in diameters:

        terminal_velocity = get_terminal_velocity(diameter)
        residence_time_s = 1.5 / terminal_velocity


        residence_times.append(residence_time_s)

    diameters_um = diameters * 1e6
    settling_df = pd.DataFrame(
        {
            "Aerosol diameter (um)": diameters_um,
            "Residence time (seconds)": residence_times,
        }
    )

    return settling_df


def return_bagheri_dfs():
    # Data extracted from Figure 6 of Bagheri et. al. 2023 (Breathing + Speaking (normal))
    # https://doi.org/10.1016/j.jaerosci.2022.106102
    df_speaking = pd.read_csv(SPEAKING_DATA_FILE)
    df_breathing = pd.read_csv(BREATHING_DATA_FILE)

    def process_df(df):
        df = df.pivot(
            index="point_group",
            columns="point_type",
            values=["concentration", "diameter_um"],
        )
        df.columns = ["_".join(col).strip() for col in df.columns.values]
        df["diameter"] = df["diameter_um_median"]


        df["concentration_upper_ci_err"] = (
            df["concentration_upper_ci"] - df["concentration_median"]
        )
        df["concentration_lower_ci_err"] = (
            df["concentration_median"] - df["concentration_lower_ci"]
        )

        df = df.drop(
            columns=[
                "diameter_um_median",
                "diameter_um_lower_ci",
                "diameter_um_upper_ci",

            ],
            inplace=False
        )

        return df

    df_pivot_breathing = process_df(df_breathing)
    df_pivot_speaking = process_df(df_speaking)

    return df_pivot_breathing, df_pivot_speaking


def return_figure_1():
    df_pivot_breathing, df_pivot_speaking = return_bagheri_dfs()
    fig, axs = plt.subplots(
        2, 1, figsize=(8, 5.8), height_ratios=[2, 2], dpi=600, sharex=True
    )

    sns_colors = sns.color_palette()

    axs[0].errorbar(
        df_pivot_breathing["diameter"],
        df_pivot_breathing["concentration_median"],
        yerr=[
            df_pivot_breathing["concentration_lower_ci_err"],
            df_pivot_breathing["concentration_upper_ci_err"],
        ],
        fmt="o",
        color=sns_colors[0],
        label="Median with CI",
        markersize=4,
    )
    axs[0].text(
        0.48,
        0.3,
        "Breathing",
        transform=axs[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=sns_colors[0],
    )



    axs[0].errorbar(
        df_pivot_speaking["diameter"],
        df_pivot_speaking["concentration_median"],
        yerr=[
            df_pivot_speaking["concentration_lower_ci_err"],
            df_pivot_speaking["concentration_upper_ci_err"],
        ],
        fmt="o",
        color=sns_colors[1],
        label="Median with CI",
        markersize=4,
    )
    axs[0].text(
        0.69,
        0.75,
        "Speaking",
        transform=axs[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=sns_colors[1],
    )

    axs[0].set_title("a", loc="left", fontsize=12, fontweight="bold", x=-0.095, y=1.05)

    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    axs[0].set_xlabel("Aerosol diameter (μm)", fontsize=9)
    axs[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.2f}".format(x)))
    axs[0].tick_params(axis="x", which="major", labelbottom=True, labelsize=8)
    axs[0].tick_params(axis="y", which="major", labelsize=8)
    axs[0].set_ylabel(
        r"Particle concentration (log) / $\mathregular{cm^{-3}}$", fontsize=9
    )
    axs[0].set_title("")

    axs[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.4),
        title=None,
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    settling_df = get_settling_time()

    settling_df_sorted = settling_df.sort_values(by="Aerosol diameter (um)")
    sns.lineplot(
        data=settling_df_sorted,
        x="Aerosol diameter (um)",
        y="Residence time (seconds)",
        ax=axs[1],
        color=sns_colors[4],
    )

    axs[1].set_title("b", loc="left", fontsize=12, fontweight="bold", x=-0.095, y=0.95)

    axs[1].set_yscale("log")
    axs[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.2f}".format(x)))

    axs[1].set_xlabel("Aerosol diameter (μm)", fontsize=9)
    axs[1].set_ylabel("Residence time", fontsize=9)

    tick_values = [
        1,
        60,
        3600,
        86400,
    ]  # Values in seconds for 1 second, 1 minute, 1 hour, and 1 day
    tick_labels = ["1s", "1m", "1h", "24h"]

    axs[1].set_yticks(tick_values)
    axs[1].set_yticklabels(tick_labels)

    axs[1].set_title("")
    axs[1].tick_params(axis="x", which="major", bottom=False, labelsize=8)
    axs[1].tick_params(axis="y", which="major", labelsize=8)

    for ax in axs:
        ax.tick_params(
            axis="both", which="minor", left=False, right=False, top=False, bottom=False
        )
        ax.grid(True, which="major", color="gray", linewidth=0.2)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    plt.savefig("fig/aerosol_size_distribution.png", dpi=600)
    plt.savefig("fig/aerosol_size_distribution.pdf")


if __name__ == "__main__":
    return_figure_1()
