import matplotlib.pyplot as plt
#
# max_len = [1, 2, 5, 10, 20]
#
# # BLEU scores for the previous implementation
# bleu_previous = [0.0, 0.0, 13.5, 20.3, 19.4]
# # Decoding times for the previous implementation (in seconds)
# time_previous = [6.24, 6.56, 7.93, 10.22, 13.01]
# # BLEU scores for the constant beam size stopping criterion
# bleu_constant = [0.0, 0.0, 13.3, 4.3, 1.7]
# # Decoding times for the constant beam size stopping criterion (in seconds)
# time_constant = [6.24, 6.59, 7.75, 12.69, 62.84]
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
#
# ax1.plot(max_len, bleu_previous, label='Previous Implementation', marker='o', color='b', linestyle='-', linewidth=2)
# ax1.plot(max_len, bleu_constant, label='Constant Beam Size', marker='o', color='r', linestyle='-', linewidth=2)
# ax1.set_xlabel('Max Length', fontsize=12)
# ax1.set_ylabel('BLEU Score', fontsize=12, color='black')
# ax1.set_title('BLEU Score Comparison', fontsize=14)
# ax1.set_xticks(max_len)
# ax1.legend(loc='upper left', fontsize=10)
#
# ax2.plot(max_len, time_previous, label='Previous Implementation', marker='s', color='b', linestyle='--', linewidth=2)
# ax2.plot(max_len, time_constant, label='Constant Beam Size', marker='s', color='r', linestyle='--', linewidth=2)
# ax2.set_xlabel('Max Length', fontsize=12)
# ax2.set_ylabel('Decoding Time (seconds)', fontsize=12, color='gray')
# ax2.set_title('Decoding Time Comparison', fontsize=14)
# ax2.set_xticks(max_len)
# ax2.legend(loc='upper left', fontsize=10)
#
# # Adjust layout to make room for the legends and labels
# plt.tight_layout()
#
# # Show the plot
# plt.show()


# Data
max_len = [4, 5, 6, 8, 10, 15, 20]
bleu_constant = [8.6, 12.2, 8.0, 6.0, 5.7, 5.6, 5.4]
time_constant = [7.90, 8.58, 9.48, 12.09, 13.88, 21.14, 33.03]

# Create figure and axis
fig, ax1 = plt.subplots()

# Plot BLEU score
color = 'tab:blue'
ax1.set_xlabel('Max Length')
ax1.set_ylabel('BLEU Score', color=color)
ax1.set_xticks(max_len)
ax1.plot(max_len, bleu_constant, color=color, marker='o', label='BLEU Score')
ax1.tick_params(axis='y', labelcolor=color)

# Create second y-axis for decoding time
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Decoding Time (seconds)', color=color)
ax2.plot(max_len, time_constant, color=color, marker='^', label='Decoding Time')
ax2.tick_params(axis='y', labelcolor=color)

# Add title and show plot
plt.title('BLEU Score and Decoding Time vs Max Sequence Length')

# Display the plot
plt.show()