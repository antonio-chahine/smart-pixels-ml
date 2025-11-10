
# Read the results
test_result = pd.read_csv("/work/submit/anton100/msci-project/smart-pixels-ml/outfile_3e778b82/evaluation_results_weights.01-t-353.70-v-537.08.csv")


# Calculate residuals
test_result['residual_x'] = test_result['xtrue'] - test_result['x']
test_result['residual_y'] = test_result['ytrue'] - test_result['y']
test_result['residual_cotA'] = test_result['cotAtrue'] - test_result['cotA']
test_result['residual_cotB'] = test_result['cotBtrue'] - test_result['cotB']

# Set up the figure
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Model Predictions vs Ground Truth', fontsize=16)

# Plot 1: x prediction vs true
axes[0, 0].scatter(test_result['xtrue'], test_result['x'], alpha=0.3, s=1)
axes[0, 0].plot([test_result['xtrue'].min(), test_result['xtrue'].max()], 
                [test_result['xtrue'].min(), test_result['xtrue'].max()], 'r--', lw=2)
axes[0, 0].set_xlabel('True x')
axes[0, 0].set_ylabel('Predicted x')
axes[0, 0].set_title('x-midplane')

# Plot 2: y prediction vs true
axes[0, 1].scatter(test_result['ytrue'], test_result['y'], alpha=0.3, s=1)
axes[0, 1].plot([test_result['ytrue'].min(), test_result['ytrue'].max()], 
                [test_result['ytrue'].min(), test_result['ytrue'].max()], 'r--', lw=2)
axes[0, 1].set_xlabel('True y')
axes[0, 1].set_ylabel('Predicted y')
axes[0, 1].set_title('y-midplane')

# Plot 3: cotA prediction vs true
axes[0, 2].scatter(test_result['cotAtrue'], test_result['cotA'], alpha=0.3, s=1)
axes[0, 2].plot([test_result['cotAtrue'].min(), test_result['cotAtrue'].max()], 
                [test_result['cotAtrue'].min(), test_result['cotAtrue'].max()], 'r--', lw=2)
axes[0, 2].set_xlabel('True cotAlpha')
axes[0, 2].set_ylabel('Predicted cotAlpha')
axes[0, 2].set_title('cotAlpha')

# Plot 4: cotB prediction vs true
axes[0, 3].scatter(test_result['cotBtrue'], test_result['cotB'], alpha=0.3, s=1)
axes[0, 3].plot([test_result['cotBtrue'].min(), test_result['cotBtrue'].max()], 
                [test_result['cotBtrue'].min(), test_result['cotBtrue'].max()], 'r--', lw=2)
axes[0, 3].set_xlabel('True cotBeta')
axes[0, 3].set_ylabel('Predicted cotBeta')
axes[0, 3].set_title('cotBeta')

# Plot 5-8: Residual distributions
axes[1, 0].hist(test_result['residual_x'], bins=100, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Residual')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title(f'x Residuals (μ={test_result["residual_x"].mean():.4f}, σ={test_result["residual_x"].std():.4f})')
axes[1, 0].axvline(0, color='r', linestyle='--', lw=2)

axes[1, 1].hist(test_result['residual_y'], bins=100, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Residual')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'y Residuals (μ={test_result["residual_y"].mean():.4f}, σ={test_result["residual_y"].std():.4f})')
axes[1, 1].axvline(0, color='r', linestyle='--', lw=2)

axes[1, 2].hist(test_result['residual_cotA'], bins=100, edgecolor='black', alpha=0.7)
axes[1, 2].set_xlabel('Residual')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title(f'cotA Residuals (μ={test_result["residual_cotA"].mean():.4f}, σ={test_result["residual_cotA"].std():.4f})')
axes[1, 2].axvline(0, color='r', linestyle='--', lw=2)

axes[1, 3].hist(test_result['residual_cotB'], bins=100, edgecolor='black', alpha=0.7)
axes[1, 3].set_xlabel('Residual')
axes[1, 3].set_ylabel('Frequency')
axes[1, 3].set_title(f'cotB Residuals (μ={test_result["residual_cotB"].mean():.4f}, σ={test_result["residual_cotB"].std():.4f})')
axes[1, 3].axvline(0, color='r', linestyle='--', lw=2)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===")
print(f"\nx-midplane:")
print(f"  Mean residual: {test_result['residual_x'].mean():.6f}")
print(f"  Std residual:  {test_result['residual_x'].std():.6f}")
print(f"  RMSE:          {np.sqrt((test_result['residual_x']**2).mean()):.6f}")

print(f"\ny-midplane:")
print(f"  Mean residual: {test_result['residual_y'].mean():.6f}")
print(f"  Std residual:  {test_result['residual_y'].std():.6f}")
print(f"  RMSE:          {np.sqrt((test_result['residual_y']**2).mean()):.6f}")

print(f"\ncotAlpha:")
print(f"  Mean residual: {test_result['residual_cotA'].mean():.6f}")
print(f"  Std residual:  {test_result['residual_cotA'].std():.6f}")
print(f"  RMSE:          {np.sqrt((test_result['residual_cotA']**2).mean()):.6f}")

print(f"\ncotBeta:")
print(f"  Mean residual: {test_result['residual_cotB'].mean():.6f}")
print(f"  Std residual:  {test_result['residual_cotB'].std():.6f}")
print(f"  RMSE:          {np.sqrt((test_result['residual_cotB']**2).mean()):.6f}")