import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats

# Đọc dữ liệu đã được xử lý
df = pd.read_csv('dataset_proceeded.csv')

# Chuẩn bị dữ liệu
X = df.drop(columns='salary_in_usd')
y = df['salary_in_usd']

# Split data (giống model.py)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions (trong log scale)
y_pred = model.predict(X_test)

# Convert về original scale để tính metrics
y_test_original = np.exp(y_test)
y_pred_original = np.exp(y_pred)

# Tính metrics trên original scale
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_original)
mae = mean_absolute_error(y_test_original, y_pred_original)

# Residuals trong original scale
residuals_original = y_test_original - y_pred_original

# Residuals trong log scale
residuals_log = y_test - y_pred

print("="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)
print(f"R² Score:                {r2:.4f}")
print(f"Mean Squared Error:      ${mse:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")
print(f"Mean Absolute Error:     ${mae:,.2f}")
print("="*60)

# ========== VISUALIZATION ==========
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(18, 12))

# 1. Actual vs Predicted (Original Scale)
ax1 = plt.subplot(2, 3, 1)
plt.scatter(y_test_original, y_pred_original, alpha=0.6, color='steelblue', 
            edgecolors='navy', s=60, linewidth=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], 
         [y_test_original.min(), y_test_original.max()], 
         'r--', lw=2.5, label='Perfect Prediction')
plt.xlabel('Actual Salary (USD)', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Salary (USD)', fontsize=12, fontweight='bold')
plt.title('Actual vs Predicted Salary\n(Original Scale)', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 2. Residuals Distribution (Original Scale)
ax2 = plt.subplot(2, 3, 2)
plt.hist(residuals_original, bins=50, color='lightcoral', edgecolor='darkred', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Zero Error')
plt.xlabel('Residuals (USD)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Distribution of Residuals\n(Original Scale)', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')

# 3. Q-Q Plot (Log Scale Residuals)
ax3 = plt.subplot(2, 3, 3)
stats.probplot(residuals_log, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals\n(Log Scale)', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)

# 4. Feature Importance (Coefficients)
ax4 = plt.subplot(2, 3, 4)
feature_names = [
    'work_On-site',
    'work_Remote',
    'comp_is_US',
    'company_size',
    'DE',  # Data Engineering
    'DS',  # Data Science
    'ML/AI',
    'Other',
    'exp_level'
]
coefficients = model.coef_
colors = ['green' if c > 0 else 'red' for c in coefficients]
bars = plt.barh(feature_names, coefficients, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
plt.xlabel('Coefficient Value', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Feature Importance\n(Model Coefficients)', fontsize=13, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
plt.grid(True, alpha=0.3, axis='x')

# Thêm giá trị lên bars
for bar, coef in zip(bars, coefficients):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, 
             f'{coef:.3f}',
             ha='left' if width > 0 else 'right',
             va='center', fontsize=9, fontweight='bold')

# 5. Residuals vs Predicted (Original Scale)
ax5 = plt.subplot(2, 3, 5)
plt.scatter(y_pred_original, residuals_original, alpha=0.6, color='purple', 
            edgecolors='darkviolet', s=60, linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2.5, label='Zero Residual')
plt.xlabel('Predicted Salary (USD)', fontsize=12, fontweight='bold')
plt.ylabel('Residuals (USD)', fontsize=12, fontweight='bold')
plt.title('Residuals vs Predicted Values\n(Original Scale)', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 6. Model Performance Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

metrics_text = f"""
MODEL PERFORMANCE SUMMARY
{'='*42}

Performance Metrics (Original Scale):
  • R² Score:            {r2:.4f}
  • Mean Squared Error:  ${mse:,.0f}
  • Root MSE:            ${rmse:,.0f}
  • Mean Abs Error:      ${mae:,.0f}

{'='*42}

Dataset Information:
  • Training samples:    {len(X_train)}
  • Test samples:        {len(X_test)}
  • Number of features:  {X.shape[1]}

Model Parameters:
  • Intercept (log):     {model.intercept_:.4f}
  • Random State:        40

{'='*42}
Note: Model trained on log-transformed salaries
Metrics calculated on original salary scale
"""

plt.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
         verticalalignment='center', 
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6, pad=1))
plt.title('Model Statistics', fontsize=13, fontweight='bold', pad=20)

plt.suptitle('Salary Prediction Model - Comprehensive Visualization', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('model_visualization_complete.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'model_visualization_complete.png'")
plt.show()

# ========== ADDITIONAL VISUALIZATION: Feature Analysis ==========
fig2, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.ravel()

full_feature_names = [
    'work_models_On-site',
    'work_models_Remote',
    'comp_is_US',
    'company_size_encoded',
    'domain_expertise_Data Engineering',
    'domain_expertise_Data Science',
    'domain_expertise_Machine Learning / AI',
    'domain_expertise_Other',
    'exp_level'
]

for idx, (col, short_name) in enumerate(zip(X.columns, feature_names)):
    ax = axes[idx]
    
    # Tạo dataframe để phân tích
    feature_data = pd.DataFrame({
        'Feature': X_test[col],
        'Actual': y_test_original,
        'Predicted': y_pred_original
    })
    
    # Group by và tính trung bình
    grouped = feature_data.groupby('Feature').mean()
    
    if len(grouped) > 0:
        x_pos = np.arange(len(grouped))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, grouped['Actual'], width, 
                      label='Actual', alpha=0.8, color='steelblue', edgecolor='navy')
        bars2 = ax.bar(x_pos + width/2, grouped['Predicted'], width, 
                      label='Predicted', alpha=0.8, color='coral', edgecolor='darkred')
        
        ax.set_xlabel(short_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Salary (USD)', fontsize=10)
        ax.set_title(f'Salary by {short_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(grouped.index, rotation=45, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.suptitle('Average Salary Analysis by Feature', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('model_feature_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Feature analysis saved as 'model_feature_analysis.png'")
plt.show()

print("\n" + "="*60)
print("VISUALIZATION COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nGenerated files:")
print("  1. model_visualization_complete.png")
print("  2. model_feature_analysis.png")
print(f"\nModel R² Score: {r2:.4f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")
print("="*60)