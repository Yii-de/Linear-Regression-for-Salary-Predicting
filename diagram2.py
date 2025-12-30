import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Đọc dữ liệu
df = pd.read_csv('dataset_proceeded.csv')

# Chuẩn bị dữ liệu
X = df.drop(columns='salary_in_usd')
y = df['salary_in_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Tạo figure với nhiều subplot
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 12))

# 1. Biểu đồ so sánh giá trị thực tế vs dự đoán
ax1 = plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', edgecolors='k', s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Salary (USD)', fontsize=11, fontweight='bold')
plt.ylabel('Predicted Salary (USD)', fontsize=11, fontweight='bold')
plt.title('Actual vs Predicted Salary', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Biểu đồ phân phối residuals
ax2 = plt.subplot(2, 3, 2)
residuals = y_test - y_pred
plt.hist(residuals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
plt.xlabel('Residuals (USD)', fontsize=11, fontweight='bold')
plt.ylabel('Frequency', fontsize=11, fontweight='bold')
plt.title('Distribution of Residuals', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Biểu đồ Q-Q plot cho residuals
ax3 = plt.subplot(2, 3, 3)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)

# 4. Biểu đồ importance của các features
ax4 = plt.subplot(2, 3, 4)
feature_names = ['emp_is_US', 'work_On-site', 'work_Remote', 'comp_is_US', 
                 'company_size', 'job_title_group', 'exp_level']
coefficients = model.coef_
colors = ['green' if c > 0 else 'red' for c in coefficients]
plt.barh(feature_names, coefficients, color=colors, alpha=0.7, edgecolor='black')
plt.xlabel('Coefficient Value', fontsize=11, fontweight='bold')
plt.ylabel('Features', fontsize=11, fontweight='bold')
plt.title('Feature Importance (Coefficients)', fontsize=13, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='x')

# 5. Biểu đồ Residuals vs Predicted Values
ax5 = plt.subplot(2, 3, 5)
plt.scatter(y_pred, residuals, alpha=0.5, color='purple', edgecolors='k', s=50)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
plt.xlabel('Predicted Salary (USD)', fontsize=11, fontweight='bold')
plt.ylabel('Residuals (USD)', fontsize=11, fontweight='bold')
plt.title('Residuals vs Predicted Values', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Metrics Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = np.mean(np.abs(residuals))

metrics_text = f"""
Model Performance Metrics
{'='*35}

R² Score:           {r2:.4f}
Mean Squared Error: ${mse:,.2f}
Root MSE:           ${rmse:,.2f}
Mean Absolute Error: ${mae:,.2f}

{'='*35}

Number of Test Samples: {len(y_test)}
Number of Features:     {X.shape[1]}
Model Intercept:        ${model.intercept_:,.2f}
"""

plt.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.5))
plt.title('Model Performance Summary', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('salary_prediction_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Tạo biểu đồ bổ sung: Phân tích theo từng feature
fig2, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.ravel()

for idx, col in enumerate(X.columns):
    ax = axes[idx]
    feature_data = pd.DataFrame({
        'Feature': X_test[col],
        'Actual': y_test,
        'Predicted': y_pred
    })
    
    # Groupby và tính trung bình
    grouped = feature_data.groupby('Feature').mean()
    
    x_pos = np.arange(len(grouped))
    width = 0.35
    
    ax.bar(x_pos - width/2, grouped['Actual'], width, label='Actual', 
           alpha=0.8, color='steelblue', edgecolor='black')
    ax.bar(x_pos + width/2, grouped['Predicted'], width, label='Predicted', 
           alpha=0.8, color='coral', edgecolor='black')
    
    ax.set_xlabel(col, fontsize=10, fontweight='bold')
    ax.set_ylabel('Average Salary (USD)', fontsize=10)
    ax.set_title(f'Salary by {col}', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped.index, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('salary_by_features.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("VISUALIZATION COMPLETED!")
print("="*50)
print(f"\nGenerated 2 visualization files:")
print("1. salary_prediction_visualization.png")
print("2. salary_by_features.png")
print(f"\nModel R² Score: {r2:.4f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")