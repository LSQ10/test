import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
import warnings
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
warnings.filterwarnings('ignore')

data = pd.read_excel('./data.xlsx')

# 查看数据基本信息
print("数据形状:", data.shape)
print("\n前5行数据:")
print(data.head())
print("\n缺失值检查:")
print(data.isnull().sum())

data = data.drop(['序号', '姓名'], axis=1)

# 处理缺失值
print("\n处理缺失值前形状:", data.shape)
data = data.dropna()
print("处理缺失值后形状:", data.shape)

# 准备特征和目标变量
X = data.drop(['嗓音障碍（是=1；否=0）'], axis=1)  # 特征
y = data['嗓音障碍（是=1；否=0）']  # 目标变量

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")
print(f"训练集中各类样本数量:\n{y_train.value_counts()}")
print(f"测试集中各类样本数量:\n{y_test.value_counts()}")

# ===== SHAP特征重要性筛选 =====
import shap
print("\n正在计算SHAP特征重要性...")
# 用XGBoost做特征重要性解释
xgb_shap_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]) if sum(y_train==1) > 0 else 1
)
xgb_shap_model.fit(X_train, y_train)
explainer = shap.TreeExplainer(xgb_shap_model)
shap_values = explainer.shap_values(X_train)
shap_importance = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'shap_importance': shap_importance
}).sort_values('shap_importance', ascending=False)
print("SHAP特征重要性排序:")
print(feature_importance)
# 只选前10重要特征
top_features = feature_importance['feature'].iloc[:10].tolist()
print(f"选用前10重要特征: {top_features}")
X_train = X_train[top_features]
X_test = X_test[top_features]

# 数据标准化（对需要标准化的模型）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义评估函数
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, scaled=False):
    """评估模型性能"""
    if scaled:
        X_train_eval = X_train_scaled
        X_test_eval = X_test_scaled
    else:
        X_train_eval = X_train
        X_test_eval = X_test
    
    # 训练模型
    model.fit(X_train_eval, y_train)

    # 预测
    y_train_pred = model.predict(X_train_eval)
    y_test_pred = model.predict(X_test_eval)

    # 计算AUC（需概率或决策分数）
    try:
        if hasattr(model, "predict_proba"):
            y_test_score = model.predict_proba(X_test_eval)[:, 1]
        elif hasattr(model, "decision_function"):
            y_test_score = model.decision_function(X_test_eval)
        else:
            y_test_score = y_test_pred  # 若无概率则用标签
        auc = roc_auc_score(y_test, y_test_score)
    except Exception as e:
        auc = np.nan

    # 计算指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    # 交叉验证
    cv_scores = cross_val_score(model, X_train_eval, y_train, cv=5, scoring='accuracy')

    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"5折交叉验证平均准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    print(f"\n分类报告:")
    print(classification_report(y_test, y_test_pred, target_names=['无嗓音障碍', '有嗓音障碍']))

    print(f"混淆矩阵:")
    print(confusion_matrix(y_test, y_test_pred))

    return {
        'model_name': model_name,
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

# 模型训练和评估
results = []

# Logistic Regression (基线模型)
print("\n" + "="*60)
print("1. Logistic Regression 模型训练")
print("="*60)
lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)
lr_result = evaluate_model(lr_model, X_train, X_test, y_train, y_test, "Logistic Regression", scaled=True)
results.append(lr_result)

# Random Forest
print("\n" + "="*60)
print("2. Random Forest 模型训练")
print("="*60)
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,  # 限制树深度，防止过拟合
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_result = evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest", scaled=False)
results.append(rf_result)

# XGBoost
print("\n" + "="*60)
print("3. XGBoost 模型训练")
print("="*60)
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,  # 剪枝参数，限制最大深度
    min_child_weight=3,  # 剪枝参数，最小叶子节点样本数
    gamma=0.2,  # 剪枝参数，分裂所需最小损失减少
    subsample=0.8,  # 子采样比例
    colsample_bytree=0.8,  # 特征采样比例
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]) if sum(y_train==1) > 0 else 1
)
xgb_result = evaluate_model(xgb_model, X_train, X_test, y_train, y_test, "XGBoost", scaled=False)
results.append(xgb_result)

# SVM
print("\n" + "="*60)
print("4. SVM 模型训练")
print("="*60)
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    random_state=42,
    class_weight='balanced',
    probability=True
)
svm_result = evaluate_model(svm_model, X_train, X_test, y_train, y_test, "SVM", scaled=True)
results.append(svm_result)

# 决策树
print("\n" + "="*60)
print("5. Decision Tree 模型训练")
print("="*60)
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(
    random_state=42,
    class_weight='balanced',
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    max_leaf_nodes=20,
    ccp_alpha=0.01,
    criterion='gini'
)
dt_result = evaluate_model(dt_model, X_train, X_test, y_train, y_test, "Decision Tree", scaled=False)
results.append(dt_result)

# KNN
print("\n" + "="*60)
print("6. KNN 模型训练")
print("="*60)
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(
    n_neighbors=5,
    n_jobs=-1
)
knn_result = evaluate_model(knn_model, X_train, X_test, y_train, y_test, "KNN", scaled=True)
results.append(knn_result)

# 神经网络（MLP）
print("\n" + "="*60)
print("7. MLP 模型训练")
print("="*60)
from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=500,
    random_state=42,
    alpha=0.01,  # L2正则化强度
    early_stopping=True,  # 提前停止防止过拟合
    validation_fraction=0.2
)
mlp_result = evaluate_model(mlp_model, X_train, X_test, y_train, y_test, "MLP", scaled=True)
results.append(mlp_result)

# ================== 模型堆叠 ==================

from sklearn.ensemble import StackingClassifier
from itertools import combinations
base_models = [
    ('lr', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced', n_jobs=-1)),
    ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, min_child_weight=3, gamma=0.2, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss', scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]) if sum(y_train==1) > 0 else 1)),
    ('svm', SVC(kernel='rbf', C=1.0, random_state=42, class_weight='balanced', probability=True)),
    ('dt', DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5, min_samples_split=10, min_samples_leaf=5, max_leaf_nodes=20, ccp_alpha=0.01, criterion='gini')),
    ('knn', KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, alpha=0.01, early_stopping=True, validation_fraction=0.2))
]
print("\n" + "="*60)
print("8. 多种模型堆叠（Stacking）训练")
print("="*60)
for n in range(2, len(base_models)+1):
    for combo in combinations(base_models, n):
        combo_names = [name for name, _ in combo]
        estimators = [(name, model) for name, model in combo]
        stack_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            cv=5,
            n_jobs=-1,
            passthrough=False
        )
        model_label = f"Stacking({' + '.join(combo_names)})"
        print(f"\n正在训练: {model_label}")
        stack_result = evaluate_model(stack_model, X_train, X_test, y_train, y_test, model_label, scaled=True)
        results.append(stack_result)

# 结果比较
print("\n" + "="*60)
print("模型性能比较")
print("="*60)

results_df = pd.DataFrame(results)
results_df = results_df[['model_name', 'train_accuracy', 'test_accuracy', 'auc', 'precision', 'recall', 'f1', 'cv_mean', 'cv_std']]
results_df = results_df.sort_values('test_accuracy', ascending=False)
print(results_df.to_string(index=False))