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


def clean_type_name(ft):
    return TYPE_NAMES.get(ft, f"Type {ft}").replace("\n", " ")


def format_cell_value(value):
    if np.isnan(value):
        return "NA"
    return f"{value:.2f}"


def plot_one_empirical_probability_figure(cases, cv_percent):
    cases = sorted(cases, key=lambda c: int(c["follower_type"]))

    empirical = np.column_stack([
        np.array(case["empirical_probability"], dtype=float)
        for case in cases
    ])

    n_followers = empirical.shape[0]
    n_cases = empirical.shape[1]

    followers = [f"SP {i + 1}" for i in range(n_followers)]

    types = [
        TYPE_NAMES.get(int(case["follower_type"]), f"Type {case['follower_type']}")
        for case in cases
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.1))

    im = ax.imshow(
        empirical,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="RdBu",
    )

    #ax.set_title(f"Profitability probability, CV = {cv_percent:.0f}%", pad=10)
    ax.set_xticks(np.arange(n_cases))
    ax.set_xlabel("SP risk class")
    ax.set_xticklabels(types, rotation=0, ha="center")
    ax.tick_params(axis="x", pad=6)

    ax.set_yticks(np.arange(n_followers))
    ax.set_yticklabels(followers)

    ax.set_xlabel("SP risk class")
    ax.set_ylabel("SPs")

    for i in range(n_followers):
        for j in range(n_cases):
            val = empirical[i, j]
            ax.text(
                j,
                i,
                format_cell_value(val),
                ha="center",
                va="center",
                color="white" if not np.isnan(val) and val > 0.65 else "black",
                fontsize=16,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability")

    plt.tight_layout()

    filename = f"empirical_probability_CV_{cv_percent:.0f}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("Saved figure:", filename)


def plot_difference_curves_by_follower(all_case_data):
    follower_types = sorted(set(int(case["follower_type"]) for case in all_case_data))
    cv_values = sorted(set(float(case["cv_percent"]) for case in all_case_data))
    cv_values = [cv for cv in cv_values if cv >= 25]

    curve_colors = [
        "#7A7A7A",
        "#8FB9D9",
        "#3F88C5",
        "#08306B",
    ]

    n_followers = len(all_case_data[0]["nu_hat"])

    x = np.arange(len(follower_types))

    type_labels = [
        TYPE_NAMES.get(ft, f"Type {ft}")
        for ft in follower_types
    ]

    for follower_idx in range(n_followers):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))

        for i, cv_percent in enumerate(cv_values):
            gaps = []

            for ft in follower_types:
                matching_cases = [
                    case for case in all_case_data
                    if float(case["cv_percent"]) == cv_percent
                    and int(case["follower_type"]) == ft
                ]

                if matching_cases:
                    case = matching_cases[0]

                    lower_bound = np.array(case["nu_hat"], dtype=float)[follower_idx]
                    empirical = np.array(case["empirical_probability"], dtype=float)[follower_idx]

                    gaps.append(empirical - lower_bound)
                else:
                    gaps.append(np.nan)

            ax.plot(
                x,
                gaps,
                marker="o",
                linewidth=4,
                color=curve_colors[i % len(curve_colors)],
                label=f"CV = {cv_percent:.0f}%",
            )

        ax.axhline(0, linestyle="--", linewidth=1, color="black", alpha=0.5)

        #ax.set_title(f"Guarantee gap, Follower {follower_idx + 1}")
        ax.set_ylabel("Gap")
        ax.set_xticks(x)
        ax.set_xlabel("SP risk class")
        ax.set_xticklabels(type_labels, rotation=0, ha="center")
        ax.tick_params(axis="x", pad=6)

        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(frameon=False)

        plt.tight_layout()

        filename = f"guarantee_gap_follower_{follower_idx + 1}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.show()
        plt.close(fig)

        print("Saved figure:", filename)


def plot_max_gap_across_followers(all_case_data):
    follower_types = sorted(set(int(case["follower_type"]) for case in all_case_data))
    cv_values = sorted(set(float(case["cv_percent"]) for case in all_case_data))
    cv_values = [cv for cv in cv_values if cv >= 25]

    curve_colors = [
        "#7A7A7A",
        "#8FB9D9",
        "#3F88C5",
        "#08306B",
    ]

    x = np.arange(len(follower_types))

    type_labels = [
        TYPE_NAMES.get(ft, f"Type {ft}")
        for ft in follower_types
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    print("\nMAX GUARANTEE GAP ACROSS FOLLOWERS")
    print("=" * 80)
    print("CV | Risk type | Max gap | Follower")
    print("-" * 80)

    for i, cv_percent in enumerate(cv_values):
        max_gaps = []

        for ft in follower_types:
            matching_cases = [
                case for case in all_case_data
                if float(case["cv_percent"]) == cv_percent
                and int(case["follower_type"]) == ft
            ]

            if matching_cases:
                case = matching_cases[0]

                lower_bound = np.array(case["nu_hat"], dtype=float)
                empirical = np.array(case["empirical_probability"], dtype=float)

                gaps = empirical - lower_bound

                if np.all(np.isnan(gaps)):
                    max_gap = np.nan
                    max_follower = "NA"
                else:
                    max_idx = int(np.nanargmax(gaps))
                    max_gap = float(gaps[max_idx])
                    max_follower = f"SP {max_idx + 1}"

                max_gaps.append(max_gap)

                max_gap_text = "NA" if np.isnan(max_gap) else f"{max_gap:.4f}"

                print(
                    f"{cv_percent:.0f}% | "
                    f"{clean_type_name(ft)} | "
                    f"{max_gap_text} | "
                    f"{max_follower}"
                )
            else:
                max_gaps.append(np.nan)

        ax.plot(
            x,
            max_gaps,
            marker="o",
            linewidth=4,
            markersize = 9,
            color=curve_colors[i % len(curve_colors)],
            label=f"CV = {cv_percent:.0f}%",
        )

    print("=" * 80)

    ax.axhline(0, linestyle="--", linewidth=1, color="black", alpha=0.5)

    #ax.set_title("Max guarantee gap")
    ax.set_ylabel("Maximum\nprobability gap")
    ax.set_xlabel("SP risk class")
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels, rotation=0, ha="center")
    ax.tick_params(axis="x", pad=6)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
        ncol=2,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.tight_layout()

    filename = "max_guarantee_gap_across_followers.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("Saved figure:", filename)

def plot_one_nu_hat_figure(cases, cv_percent):
    cases = sorted(cases, key=lambda c: int(c["follower_type"]))

    nu_hat = np.column_stack([
        np.array(case["nu_hat"], dtype=float)
        for case in cases
    ])

    n_followers = nu_hat.shape[0]
    n_cases = nu_hat.shape[1]

    followers = [f"SP {i + 1}" for i in range(n_followers)]

    types = [
        TYPE_NAMES.get(int(case["follower_type"]), f"Type {case['follower_type']}")
        for case in cases
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.1))

    im = ax.imshow(
        nu_hat,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="RdBu",
    )

    ax.set_xticks(np.arange(n_cases))
    ax.set_xlabel("SP risk class")
    ax.set_xticklabels(types, rotation=0, ha="center")
    ax.tick_params(axis="x", pad=6)

    ax.set_yticks(np.arange(n_followers))
    ax.set_yticklabels(followers)

    ax.set_xlabel("SP risk class")
    ax.set_ylabel("SPs")

    for i in range(n_followers):
        for j in range(n_cases):
            val = nu_hat[i, j]
            ax.text(
                j,
                i,
                format_cell_value(val),
                ha="center",
                va="center",
                color="white" if not np.isnan(val) and val > 0.65 else "black",
                fontsize=19,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Lower bound $\hat{\nu}$")

    plt.tight_layout()

    filename = f"nu_hat_CV_{cv_percent:.0f}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("Saved figure:", filename)

def plot_probability_figure(json_file="all_type_cv_results_CV_0_25_50_75_100.json"):
    with open(json_file, "r") as f:
        all_case_data = json.load(f)

    if len(all_case_data) == 0:
        raise ValueError(f"No data found in {json_file}")

    required_keys = ["nu_hat", "empirical_probability", "follower_type", "cv_percent"]
    for idx, case in enumerate(all_case_data):
        for key in required_keys:
            if key not in case:
                raise KeyError(f"Missing key '{key}' in case index {idx}")

    cv_values = sorted(set(float(case["cv_percent"]) for case in all_case_data))

    for cv_percent in cv_values:
        cases_for_cv = [
            case for case in all_case_data
            if float(case["cv_percent"]) == cv_percent
        ]

        #plot_one_empirical_probability_figure(cases_for_cv, cv_percent)
        plot_one_nu_hat_figure(cases_for_cv, cv_percent)

    #plot_difference_curves_by_follower(all_case_data)
    plot_max_gap_across_followers(all_case_data)
    #plot_one_nu_hat_figure(cases_for_cv, cv_percent)


if __name__ == "__main__":
    plot_probability_figure()