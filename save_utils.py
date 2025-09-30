from pathlib import Path

import pandas as pd


def save_stats_to_tex(results, filename: Path):
    # also save as csv
    results_df = {'statistic': results.statistic, 'pvalue': results.pvalue, 'df': results.df}
    results_df = pd.DataFrame(results_df, index=[0])
    results_df.to_csv(filename.with_suffix(".csv"), header=True, index=False)

    # check parent folder exists
    if isinstance(filename, str):
        filename = Path(filename)
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)
    with open(filename, "w") as f:
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lr}\n")
        f.write("\\toprule\n")
        f.write("Statistic & Value \\\\\n")
        f.write("\\midrule\n")
        f.write(f"$t$-statistic & {results.statistic:.4f} \\\\\n")
        f.write(f"$p$-value & {results.pvalue:.4g} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Results of the t-test between Group 1 and Group 2}\n")
        f.write("\\label{tab:ttest_results}\n")
        f.write("\\end{table}\n")
