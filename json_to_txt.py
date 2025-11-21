import json

'''transformation from json data to txt data'''

input_file = "sa_50_3.json"
output_file = "sa_50_3.txt"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "w", encoding="utf-8") as f:
    for idx, entry in enumerate(data, start=1):
        epsilon = entry.get("best_epsilon")
        twct = entry.get("best_twct")
        f.write(f"{idx}. epsilon = {epsilon}, twct = {twct}\n")

print(f"Ergebnis ist im {output_file}")
