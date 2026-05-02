import pandas as pd
import matplotlib.pyplot as plt

# Load SimCLR training statistics
simclr_stats = pd.read_csv('results/128_0.5_20_256_200_statistics.csv')
linear_stats = pd.read_csv('results/linear_statistics.csv')

# Plot 1: SimCLR Loss Curve
plt.figure(figsize=(10, 4))
plt.plot(simclr_stats['epoch'], simclr_stats['train_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('SimCLR Training Loss')
plt.grid(True)
plt.savefig('results/simclr_loss.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: kNN Accuracy Curve (only every 5 epochs, filter out 0 values)
knn_data = simclr_stats[simclr_stats['test_acc@1'] > 0]
plt.figure(figsize=(10, 4))
plt.plot(knn_data['epoch'], knn_data['test_acc@1'])
plt.xlabel('Epoch')
plt.ylabel('kNN Accuracy (%)')
plt.title('kNN Monitor Accuracy during SimCLR Training')
plt.grid(True)
plt.savefig('results/simclr_knn.png', dpi=150, bbox_inches='tight')
plt.close()

# Print final results
print("=== Experiment 1 Results ===")
print(f"Final SimCLR Loss: {simclr_stats['train_loss'].iloc[-1]:.4f}")
print(f"Best kNN Accuracy: {simclr_stats['test_acc@1'].max():.2f}%")
print(f"Best Linear Probing Accuracy: {linear_stats['test_acc@1'].max():.2f}%")


# Plot 3: Linear Probing Accuracy Curve
plt.figure(figsize=(10, 4))
plt.plot(linear_stats['epoch'], linear_stats['test_acc@1'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Linear Probing Test Accuracy')
plt.grid(True)
plt.savefig('results/linear_probing.png', dpi=150, bbox_inches='tight')
plt.close()

df = pd.read_csv('results/linear_statistics.csv')
print('Best Top-1:', df['test_acc@1'].max())
print('Best Top-5:', df['test_acc@5'].max())
print('Best Epoch:', df['test_acc@1'].idxmax() + 1)