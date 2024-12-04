import os
import subprocess
import json
import matplotlib.pyplot as plt

beam_sizes = [1, 5, 10, 15, 20, 25]
bleu_scores = []
brevity_penalties = []
execution_times = []

with open("beam_search_times.txt", "r") as file:
    for line in file:
        if "completed in" in line:
            time_str = line.split("completed in")[1].strip()
            execution_times.append(float(time_str.split(" ")[0]))  # Extracting the time in seconds
assert len(execution_times) == len(beam_sizes), "Mismatch between beam sizes and execution times."

for k in beam_sizes:
    # Command to calculate BLEU and BP using SacreBLEU
    command = f"cat translations_beam_baseline_{k}.txt | sacrebleu data/en-fr/raw/test.en"
    output = subprocess.check_output(command, shell=True)
    output = output.decode("utf-8")

    bleu_data = json.loads(output)
    bleu = bleu_data["score"]  # BLEU score
    verbose_score = bleu_data["verbose_score"]

    bp_str = verbose_score.split("(BP = ")[1].split(" ")[0]  # Extract BP value from verbose score
    bp = float(bp_str)

    bleu_scores.append(bleu)
    brevity_penalties.append(bp)

print(bleu_scores)
print(brevity_penalties)
print(execution_times)

fig, ax1 = plt.subplots()

ax1.set_xlabel('Beam Size')
ax1.set_ylabel('BLEU Score', color='tab:blue')
ax1.plot(beam_sizes, bleu_scores, color='tab:blue', marker='o', label="BLEU Score")
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Brevity Penalty', color='tab:red')
ax2.plot(beam_sizes, brevity_penalties, color='tab:red', marker='o', label="Brevity Penalty")
ax2.tick_params(axis='y', labelcolor='tab:red')

# Plot Execution Time on a third y-axis
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Move the third y-axis a bit to the right
ax3.set_ylabel('Execution Time (s)', color='tab:green')
ax3.plot(beam_sizes, execution_times, color='tab:green', marker='^', label="Execution Time")
ax3.tick_params(axis='y', labelcolor='tab:green')

plt.title("BLEU Score, Brevity Penalty, and Execution Time vs Beam Size")
fig.tight_layout()
plt.show()