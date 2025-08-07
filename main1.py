import matplotlib.pyplot as plt

# Data
lambda_values = [150, 200, 250, 300]
# effective_capacity = [18.92, 27.75, 38.25, 50.74]
ir_harq = [0.013, 0.016, 0.02, 0.025]
cc_harq_feedback = [0.0011, 0.00117, 0.00124, 0.00131]
# cc_harq_no_feedback = [5.35, 6.03, 6.57, 6.93]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, ir_harq, marker='o', label='IR-HARQ', linestyle='-', linewidth=2)
# plt.plot(lambda_values, effective_capacity, marker='*', label='effective capacity', linestyle='-.', linewidth=2)
plt.plot(lambda_values, cc_harq_feedback, marker='s', label='CC-HARQ with Feedback', linestyle='--', linewidth=2)
# plt.plot(lambda_values, cc_harq_no_feedback, marker='^', label='CC-HARQ without Feedback', linestyle=':', linewidth=2)

# Adding labels and title
plt.xlabel('Average Arrival Rate (bits/slot)', fontsize=12)
plt.ylabel('Average Transmit Power(W)', fontsize=12)
# plt.title('Average Transmit Power vs. Average Arrival Rate for HARQ Schemes', fontsize=14)
plt.grid(True)
plt.legend()

# Set x-axis ticks to only show 150, 200, 250, 300
plt.xticks(lambda_values)

# Display the plot
plt.show()