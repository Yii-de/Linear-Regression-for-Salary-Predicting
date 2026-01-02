import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats

# 1. ĐỌC DỮ LIỆU
# Giả định: 'salary_in_usd' trong file này đã được Log-transform ở bước tiền xử lý
df = pd.read_csv('dataset_proceeded.csv')

# 2. CHUẨN BỊ DỮ LIỆU
X = df.drop(columns='salary_in_usd')
y = df['salary_in_usd']

# Chia tập dữ liệu (Giữ nguyên random_state để đối chiếu)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# 3. HUẤN LUYỆN MÔ HÌNH
model = LinearRegression()
model.fit(X_train, y_train)

# 4. DỰ ĐOÁN (TRÊN LOG SCALE)
y_pred = model.predict(X_test)

# 5. TÍNH TOÁN METRICS TRÊN LOG SCALE
# Không dùng np.exp(), tính trực tiếp để xem sai số trên đơn vị Log
mse_log = mean_squared_error(y_test, y_pred)
rmse_log = np.sqrt(mse_log)
r2_log = r2_score(y_test, y_pred)
mae_log = mean_absolute_error(y_test, y_pred)
residuals_log = y_test - y_pred

print("="*60)
print("MODEL PERFORMANCE METRICS (LOG SCALE)")
print("="*60)
print(f"R² Score:                {r2_log:.4f}")
print(f"Mean Squared Error:      {mse_log:.4f}")
print(f"Root Mean Squared Error: {rmse_log:.4f}")
print(f"Mean Absolute Error:     {mae_log:.4f}")
print("="*60)

# ========== TRỰC QUAN HÓA 1: HIỆU SUẤT TỔNG THỂ (LOG SCALE) ==========
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(18, 12))

# 1. Actual vs Predicted (Log Scale)
ax1 = plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue', edgecolors='navy', s=60)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2.5, label='Perfect Prediction')
plt.xlabel('Actual Log(Salary)', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Log(Salary)', fontsize=12, fontweight='bold')
plt.title('Actual vs Predicted\n(Log Scale)', fontsize=13, fontweight='bold')
plt.legend()

# 2. Distribution of Residuals (Log Scale)
ax2 = plt.subplot(2, 3, 2)
sns.histplot(residuals_log, kde=True, color='lightcoral', edgecolor='darkred', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Residuals (Log Difference)', fontsize=12, fontweight='bold')
plt.title('Distribution of Residuals\n(Log Scale)', fontsize=13, fontweight='bold')

# 3. Q-Q Plot (Log Scale)
ax3 = plt.subplot(2, 3, 3)
stats.probplot(residuals_log, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals\n(Log Scale)', fontsize=13, fontweight='bold')

# 4. Feature Importance (Coefficients)
ax4 = plt.subplot(2, 3, 4)
feature_names = ['work_On-site', 'work_Remote', 'comp_is_US', 'company_size', 
                 'DE', 'DS', 'ML/AI', 'Other', 'exp_level']
coefficients = model.coef_
colors = ['green' if c > 0 else 'red' for c in coefficients]
bars = plt.barh(feature_names, coefficients, color=colors, alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='black', lw=1)
plt.title('Feature Importance\n(Log Scale Coefficients)', fontsize=13, fontweight='bold')
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.3f}', va='center')

# 5. Residuals vs Predicted (Log Scale) - Kiểm tra Homoscedasticity
ax5 = plt.subplot(2, 3, 5)
plt.scatter(y_pred, residuals_log, alpha=0.6, color='purple', edgecolors='darkviolet', s=60)
plt.axhline(y=0, color='red', linestyle='--', lw=2)
plt.xlabel('Predicted Log(Salary)', fontsize=12, fontweight='bold')
plt.ylabel('Residuals', fontsize=12, fontweight='bold')
plt.title('Residuals vs Predicted\n(Log Scale)', fontsize=13, fontweight='bold')

# 6. Summary Table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = (
    f"SUMMARY (LOG SCALE)\n"
    f"{'-'*30}\n"
    f"R² Score: {r2_log:.4f}\n"
    f"RMSE: {rmse_log:.4f}\n"
    f"MAE: {mae_log:.4f}\n"
    f"Train size: {len(X_train)}\n"
    f"Test size: {len(X_test)}"
)
plt.text(0.1, 0.5, summary_text, fontsize=12, family='monospace', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Salary Prediction Model - Comprehensive Log-Scale Analysis', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('model_analysis_log_scale.png', dpi=300)
plt.show()

# ========== TRỰC QUAN HÓA 2: PHÂN TÍCH ĐẶC TRƯNG (LOG SCALE) ==========
fig2, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.ravel()

for idx, (col, short_name) in enumerate(zip(X.columns, feature_names)):
    if idx >= 9: break # Giới hạn 9 đặc trưng đầu tiên
    ax = axes[idx]
    
    # Tính trung bình log salary thực tế và dự đoán theo từng nhóm đặc trưng
    feat_analysis = pd.DataFrame({'Feature': X_test[col], 'Actual': y_test, 'Predicted': y_pred})
    grouped = feat_analysis.groupby('Feature').mean()
    
    x_pos = np.arange(len(grouped))
    width = 0.35
    ax.bar(x_pos - width/2, grouped['Actual'], width, label='Actual (Log)', color='steelblue')
    ax.bar(x_pos + width/2, grouped['Predicted'], width, label='Predicted (Log)', color='coral')
    
    ax.set_title(f'Avg Log Salary by {short_name}', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped.index, rotation=45)
    ax.set_ylabel('Log Value')
    # Zoom vào vùng giá trị log để thấy rõ sự khác biệt (ví dụ từ 10.0 đến 12.5)
    ax.set_ylim([y_test.min() * 0.95, y_test.max() * 1.05])
    ax.legend(fontsize=8)

plt.suptitle('Feature Analysis: Actual vs Predicted (Log Scale Mean)', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('feature_analysis_log_scale.png', dpi=300)
plt.show()

print("\n✓ Đã lưu biểu đồ: 'model_analysis_log_scale.png' và 'feature_analysis_log_scale.png'")