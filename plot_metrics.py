import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(log_file, dataset_name):
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return

    df = pd.read_csv(log_file)
    epochs = df['epoch']
    train_loss = df['train_loss']
    test_loss = df['test_loss']
    ci = df['ci']

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{dataset_name} Dataset - Loss')
    plt.legend()
    plt.grid(True)

    # Plot CI
    plt.subplot(1, 2, 2)
    plt.plot(epochs, ci, label='Concordance Index (CI)', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('CI')
    plt.title(f'{dataset_name} Dataset - CI')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # Save the figure to the same directory as the log
    save_path = os.path.join(os.path.dirname(log_file), f'{dataset_name.lower()}_metrics.png')
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Log files paths
    davis_log = "out/train_log_davis.txt"
    kiba_log = "out/train_log_kiba.txt"

    print("Plotting metrics for Davis dataset...")
    plot_metrics(davis_log, "Davis")
    
    print("Plotting metrics for Kiba dataset...")
    plot_metrics(kiba_log, "Kiba")
