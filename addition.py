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
X_test_maxabs = max_abs_scaler.transform(X_test)  # ��ʱ��ѵ������Ӧ�����Ѿ������Ļ�������ϡ�����ݣ�רΪ����ϡ������(scipy.sparse)����Ƶ�

quantile_transformer = QuantileTransformer(random_state=24) # ������ӳ�䵽���㵽һ�ľ��ȷֲ��ϣ�Ҳ����ͨ������ output_distribution='normal' ��ת���������ӳ�䵽��̬�ֲ�
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)
# ��֤������np.percentile(X_train[:, 0], [0, 25, 50, 75, 100])

normalizer = Normalizer().fit(X)  # ��������scipy.sparse���ܼ����������ݺ�ϡ�������Ϊ���룬���ύ����ЧCython����ǰ�����ݱ�ת��Ϊѹ����ϡ������ʽ (�μ� scipy.sparse.csr_matrix )��
Ϊ�˱��ⲻ��Ҫ���ڴ渴�ƣ��Ƽ�������ѡ��CSR��ʾ
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
