from scipy import stats
import numpy as np


def find_max_difference(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lenght!!")

    max_diff = 0
    max_index = 0
    differences = []

    for i in range(len(list1)):
        diff = abs(list1[i] - list2[i])
        differences.append(diff)

        if diff > max_diff:
            max_diff = diff
            max_index = i

    return max_diff, max_index, differences


twct_before = [
    35942,
    26410,
    27959,
    22919,
    23118,
    34906,
    29662,
    28353,
    36517,
    37497,
    29706,
]

twct_after = [
    35783,
    26297,
    27785,
    22864,
    23009,
    34737,
    29568,
    28243,
    36507,
    37412,
    29625,
]

max_diff, max_index, all_differences = find_max_difference(twct_before, twct_after)

print(f"Größte Differenz: {max_diff} bei Index {max_index}")
print(f"Werte: vorher = {twct_before[max_index]}, nachher = {twct_after[max_index]}")
print(f"Durchschnittliche Differenz: {sum(all_differences) / len(all_differences):.2f}")
print(
    f"Anzahl Verbesserungen: {sum(1 for i in range(len(twct_before)) if twct_after[i] < twct_before[i])}"
)
print(
    f"Anzahl Verschlechterungen: {sum(1 for i in range(len(twct_before)) if twct_after[i] > twct_before[i])}"
)


differences = np.array(twct_after) - np.array(twct_before)

wilcoxon_result = stats.wilcoxon(
    twct_before,
    twct_after,
    alternative="greater",
    zero_method="zsplit",
    correction=False,
)

print("=== Wilcoxon Signed-Rank Test Results ===")
print(f"Test Statistic W: {wilcoxon_result.statistic}")
print(f"P-value: {wilcoxon_result.pvalue:.20f}")
print(f"Sample size: {len(twct_before)}")
print(f"Mean difference: {np.mean(differences):.2f}")
print(f"Median difference: {np.median(differences):.2f}")
print(f"Standard deviation: {np.std(differences):.2f}")
print(f"Improvements: {sum(d < 0 for d in differences)}")
print(f"Deteriorations: {sum(d > 0 for d in differences)}")
print(f"No changes: {sum(d == 0 for d in differences)}")

alpha = 0.05
if wilcoxon_result.pvalue < alpha:
    print(f"\nStatistically significant improvement (p < {alpha})")
else:
    print(f"\nNo statistically significant improvement (p ≥ {alpha})")
