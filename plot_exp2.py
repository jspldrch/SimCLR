import matplotlib.pyplot as plt
import pandas as pd

linear_stats = pd.read_csv('results/linear_statistics.csv')
supervised_stats = pd.read_csv('results/supervised_statistics.csv')

# Bar chart comparison
models = ['SimCLR\n(Linear Probing)', 'Supervised\nLearning']
accuracies = [linear_stats['test_acc@1'].max(), supervised_stats['test_acc@1'].max()]

plt.figure(figsize=(6, 5))
plt.bar(models, accuracies, color=['steelblue', 'orange'], width=0.4)
plt.ylabel('Test Accuracy Top-1 (%)')
plt.title('SimCLR vs Supervised Learning')
plt.ylim(80, 100)
plt.grid(axis='y')
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.1, f'{v:.2f}%', ha='center')
plt.savefig('results/comparison.png', dpi=150, bbox_inches='tight')
plt.close()


plt.figure(figsize=(10, 4))
plt.plot(supervised_stats['epoch'], supervised_stats['test_acc@1'], label='Supervised Learning')
plt.plot(linear_stats['epoch'], linear_stats['test_acc@1'], label='SimCLR Linear Probing')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy Top-1 (%)')
plt.title('Training Progress: SimCLR vs Supervised Learning')
plt.legend()
plt.grid(True)
plt.savefig('results/comparison_curve.png', dpi=150, bbox_inches='tight')
plt.close()