from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import Normalizer, normalize,  
X_scaled = scale(X_train)

scaler = StandardScaler().fit(X_train)
scaler.transform(X_test)

min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax = min_max_scaler.transform(X_test)

max_abs_scaler = MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
X_test_maxabs = max_abs_scaler.transform(X_test)  # 此时，训练数据应该是已经零中心化或者是稀疏数据，专为缩放稀疏数据(scipy.sparse)而设计的

quantile_transformer = QuantileTransformer(random_state=24) # 将数据映射到了零到一的均匀分布上，也可以通过设置 output_distribution='normal' 将转换后的数据映射到正态分布
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)
# 验证方法：np.percentile(X_train[:, 0], [0, 25, 50, 75, 100])

normalizer = Normalizer().fit(X)  # 接收来自scipy.sparse的密集类数组数据和稀疏矩阵作为输入，被提交给高效Cython例程前，数据被转化为压缩的稀疏行形式 (参见 scipy.sparse.csr_matrix )，
为了避免不必要的内存复制，推荐在上游选择CSR表示
normalizer.transform(X)


sns...heatmap()
sns...


from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
# generate some data to play with
X, y = samples_generator.make_classification(
n_informative=5, n_redundant=0, random_state=42)
# ANOVA SVM-C
anova_filter = SelectKBest(f_regression, k=5)
clf = svm.SVC(kernel='linear')
anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
# You can set the parameters using the names issued
# For instance, fit using a k of 10 in the SelectKBest
# and a parameter 'C' of the svm
anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
prediction = anova_svm.predict(X)
anova_svm.score(X, y)                        
# getting the selected features chosen by anova_filter
anova_svm.named_steps['anova'].get_support()
# Another way to get selected features chosen by anova_filter
anova_svm.named_steps.anova.get_support()
