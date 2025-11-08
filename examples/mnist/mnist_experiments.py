import os
import time
import subprocess
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt

# Parameter grids
epochs_list = [1, 3, 5, 10, 15]
batch_sizes = [32, 64, 128, 256]
learning_rates = [0.001, 0.005, 0.01, 0.05]

# Output files
csv_file = "mnist_results.csv"
log_file = "mnist_experiments.log"

results = []

def run_experiment(epochs, batch, lr, label, log):
    """Run one MNIST experiment and record accuracy + runtime."""
    log.write("\n" + "="*80 + "\n")
    log.write(f"Running {label}: Epochs={epochs}, Batch={batch}, LR={lr}\n")
    log.write("="*80 + "\n")

    print(f"\n=== Starting experiment: {label} | Epochs={epochs}, Batch={batch}, LR={lr} ===\n")
    log.write(f"\n=== Starting experiment: {label} | Epochs={epochs}, Batch={batch}, LR={lr} ===\n")

    start = time.time()
    cmd = f"docker run -it --rm mnist python -u main.py --epochs {epochs} --batch-size {batch} --lr {lr}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    full_output = []
    for line in process.stdout:
        # Print minimal progress to terminal
        if "Epoch" in line or "Accuracy" in line:
            print(line.strip())
        # Write all lines to log
        log.write(line)
        full_output.append(line)

    process.wait()
    end = time.time()
    runtime = round(end - start, 2)

    # Extract accuracy (pattern like "Accuracy: 9876/10000 (98.76%)")
    joined_output = "".join(full_output)
    match = re.findall(r"Accuracy:\s*\d+/\d+\s*\(([\d\.]+)%\)", joined_output)
    accuracy = float(match[-1]) if match else 0.0

    results.append([label, epochs, batch, lr, accuracy, runtime])
    print(f"Finished {label}: accuracy={accuracy:.2f}%, time={runtime}s\n")
    log.write(f"✅ Completed: {label} | Accuracy={accuracy:.2f}% | Time={runtime}s\n")


# --- Run all experiments ---
with open(log_file, "w") as log:
    log.write("MNIST Docker Experiment Logs\n")
    log.write("="*80 + "\n\n")

    # 1. Epoch tests
    for epochs in epochs_list:
        run_experiment(epochs, 64, 0.01, "Epoch Sweep", log)

    # 2. Batch size tests
    for batch in batch_sizes:
        run_experiment(5, batch, 0.01, "Batch Sweep", log)

    # 3. Learning rate tests
    for lr in learning_rates:
        run_experiment(5, 64, lr, "LR Sweep", log)

# --- Save all results to CSV ---
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Category", "Epochs", "Batch Size", "Learning Rate", "Accuracy (%)", "Time (s)"])
    writer.writerows(results)

print("\nAll experiments completed.")
print(f"Results saved to {csv_file}")
print(f"Detailed logs saved to {log_file}")

# --- Read CSV for plotting ---
df = pd.read_csv(csv_file)

def plot_category(df, category, x_col, y_col, title, ylabel, filename, color):
    subset = df[df["Category"] == category]
    plt.figure(figsize=(6,4))
    plt.plot(subset[x_col], subset[y_col], marker="o", color=color)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- Create plots ---
plot_category(df, "Epoch Sweep", "Epochs", "Accuracy (%)", "Accuracy vs. Epochs", "Accuracy (%)", "accuracy_vs_epochs.png", "blue")
plot_category(df, "Epoch Sweep", "Epochs", "Time (s)", "Execution Time vs. Epochs", "Time (s)", "time_vs_epochs.png", "red")

plot_category(df, "Batch Sweep", "Batch Size", "Accuracy (%)", "Accuracy vs. Batch Size", "Accuracy (%)", "accuracy_vs_batch.png", "blue")
plot_category(df, "Batch Sweep", "Batch Size", "Time (s)", "Execution Time vs. Batch Size", "Time (s)", "time_vs_batch.png", "red")

plot_category(df, "LR Sweep", "Learning Rate", "Accuracy (%)", "Accuracy vs. Learning Rate", "Accuracy (%)", "accuracy_vs_lr.png", "blue")
plot_category(df, "LR Sweep", "Learning Rate", "Time (s)", "Execution Time vs. Learning Rate", "Time (s)", "time_vs_lr.png", "red")

print("Graphs saved:")
print("  accuracy_vs_epochs.png, time_vs_epochs.png")
print("  accuracy_vs_batch.png, time_vs_batch.png")
print("  accuracy_vs_lr.png, time_vs_lr.png")