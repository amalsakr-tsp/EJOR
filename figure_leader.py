import json
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
})


TYPE_NAMES = {
    1: "RN",
    2: "MRA",
    3: "HRA",
    4: "ERA",
}


def plot_metric_by_cv(all_case_data, metric_key, ylabel, filename, scale=1.0,title=None):
    follower_types = sorted(set(int(case["follower_type"]) for case in all_case_data))
    cv_values = sorted(set(float(case["cv_percent"]) for case in all_case_data))
    cv_values = [cv for cv in cv_values if cv >= 25]

    x = np.arange(len(follower_types))

    type_labels = [
        TYPE_NAMES.get(ft, f"Type {ft}")
        for ft in follower_types
    ]

    curve_colors = [
        "#7A7A7A",
        "#8FB9D9",
        "#3F88C5",
        "#08306B",
    ]

    fig, ax = plt.subplots(figsize=(6.5, 5.1))

    for i, cv_percent in enumerate(cv_values):
        y_values = []

        for ft in follower_types:
            matching_cases = [
                case for case in all_case_data
                if float(case["cv_percent"]) == cv_percent
                and int(case["follower_type"]) == ft
            ]

            if matching_cases:
                y_values.append(float(matching_cases[0][metric_key]) / scale)
            else:
                y_values.append(np.nan)

        ax.plot(
            x,
            y_values,
            marker="o",
            linewidth = 4,
            markersize = 9,
            color=curve_colors[i % len(curve_colors)],
            label=f"CV = {cv_percent:.0f}%",
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("SP risk class")
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False)

    plt.tight_layout()

    # Save only as PDF
    plt.savefig(filename, format="pdf", bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print("Saved PDF figure:", filename)


def plot_leader_figure(json_file="all_type_cv_results_CV_0_25_50_75_100.json"):
    with open(json_file, "r") as f:
        all_case_data = json.load(f)

    if len(all_case_data) == 0:
        raise ValueError(f"No data found in {json_file}")

    plot_metric_by_cv(
        all_case_data,
        metric_key="leader_profit",
        ylabel="InP profit\n(million dollars)",
        filename="leader_profit_all_CV.pdf",
        scale=1e6,
    )

    plot_metric_by_cv(
        all_case_data,
        metric_key="C_star",
        ylabel=r"$C^*$ (vCores)",
        filename="capacity_all_CV.pdf",
        scale=1.0,
    )

    plot_metric_by_cv(
        all_case_data,
        metric_key="theta_star",
        ylabel=r"$\theta^*$ (\$/vCore-hour)",
        filename="price_all_CV.pdf",
        scale=1.0,
    )


if __name__ == "__main__":
    plot_leader_figure()