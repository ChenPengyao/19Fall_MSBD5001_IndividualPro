# encoding=utf-8
# Author: CPY
# MSBD5001 Individual Project

def MSBD5001_IP(test_size):
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import mean_squared_error

	df_train = pd.read_excel(
		'/Users/chenpengyao/Desktop/HKUST/BDT/MSBD_5001_Foundation_of_DA/individual_project/train.xlsx')  # 原始训练集
	df_pred = pd.read_csv(
		'/Users/chenpengyao/Desktop/HKUST/BDT/MSBD_5001_Foundation_of_DA/individual_project/test.csv')  # 原始测试集

	df_pred.insert(1, 'playtime_forever', 0)  # 给预测的数据集加上需要预测的变量列
	df = pd.concat([df_train, df_pred], axis=0).reset_index(
		drop=True)  # 合并training data and predict，同时重编index为了下面的哑变量矩阵处理

	# split for train dataset
	def df_labels(df):
		df_genres = df['genres'].str.split(',', expand=True)
		df_categories = df['categories'].str.split(',', expand=True)
		df_tags = df['tags'].str.split(',', expand=True)
		df_labels = pd.concat([df_categories, df_genres, df_tags], axis=1)
		df_labels.columns = [str(i) for i in range(len(df_labels.columns))]
		# 把labels拼在一起
		df_labelsT = []
		for i in range(len(df_labels.columns)):
			for j in range(len(df_labels[str(i)].value_counts().index)):
				df_labelsT.append(df_labels[str(i)].value_counts().index[j])
		df_labelsT = set(df_labelsT)
		return df_labels, df_labelsT

	df_labels, df_labelsT = df_labels(df)

	# 将labels变成哑变量
	df_dummy = pd.DataFrame(data=int(0), index=range(len(df)), columns=df_labelsT)
	for i in range(len(df_dummy)):
		for j in [x for x in list(df_labels.loc[i]) if x in df_labelsT]:
			df_dummy.loc[i][j] = int(1)

	# 处理时间类型数据
	df_dropna = df.iloc[:357, :].dropna(axis=0)

	def DateTran_train(x):
		dict_a = {'Jan': '1', 'Feb': '2', 'Mar': '3', 'Apr': '4', 'May': '5', 'Jun': '6',
				  'Jul': '7', 'Aug': '8', 'Sep': '9', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
		# x=str(x)
		y = list(x.split(' '))
		z = str(y[2]) + '/' + str(dict_a[y[1].replace(',', '')]) + '/' + str(y[0])
		return z

	df_dropna['release_date'] = df_dropna['release_date'].apply(DateTran_train)
	df_dropna['release_date'] = pd.to_datetime(df_dropna['release_date'])
	df_dropna['purchase_date'] = pd.to_datetime(df_dropna['purchase_date'])
	df_dropna['date_diff'] = df_dropna['purchase_date'] - df_dropna['release_date']

	def Date2Num(x):
		y = str(x).replace(' days 00:00:00', '')
		z = str(y).replace(' days +00:00:00', '')
		return int(z)

	df_dropna['date_diff'] = df_dropna['date_diff'].map(Date2Num)

	df_test_dropna = df.iloc[357:, :].dropna(axis=0)

	def DateTran_test(x):
		dict_a = {'Jan': '1', 'Feb': '2', 'Mar': '3', 'Apr': '4', 'May': '5', 'Jun': '6',
				  'Jul': '7', 'Aug': '8', 'Sep': '9', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
		# x=str(x)
		y = list(x.split(' '))
		z = str(y[2]) + '/' + str(dict_a[y[0]]) + '/' + str(y[1].replace(',', ''))
		return z

	df_test_dropna['purchase_date'] = df_test_dropna['purchase_date'].apply(DateTran_test)
	df_test_dropna['purchase_date'] = pd.to_datetime(df_test_dropna['purchase_date'])
	df_test_dropna['release_date'] = pd.to_datetime(df_test_dropna['release_date'])
	df_test_dropna['date_diff'] = df_test_dropna['purchase_date'] - df_test_dropna['release_date']
	df_test_dropna['date_diff'] = df_test_dropna['date_diff'].map(Date2Num)

	# 将经处理过事件类型数据的数据命名为df_1
	df_1 = pd.concat([df_dropna, df_test_dropna], axis=0)
	df_1.insert(1, 'play_or_not', 0)

	def trans_class1(x):
		if x != 0:
			return 1
		else:
			return 0

	df_1['play_or_not'] = df_1['playtime_forever'].apply(trans_class1)  # 将是否会玩该游戏，作为一个分类变量

	def trans_class2(x):
		if x:
			return 1
		else:
			return 0

	df_1['is_free'] = df_1['is_free'].apply(trans_class2)  # 把is_free用1和0表示

	"""
	用含有labels的数据，用线性回归分类器做输出，先预测是否会玩，在预测玩的时间
	"""

	# 先做一个分类模型，看看是否会玩该游戏
	df_4 = df_1.drop(
		['play_or_not', 'is_free', 'id', 'playtime_forever', "genres", "tags", "categories", 'release_date',
		 'purchase_date'], axis=1)

	# 归一化
	from sklearn.preprocessing import StandardScaler
	ss_x = StandardScaler()
	df_4 = pd.DataFrame(ss_x.fit_transform(df_4), columns=df_4.columns, index=df_4.index)
	df_4 = df_4.round(6)
	df_4 = df_4.astype(dtype='float32')
	df_4['is_free'] = df_1['is_free']
	df_4['play_or_not'] = df_1['play_or_not']
	df_4['playtime_forever'] = df_1['playtime_forever']
	df_4['id'] = df_1['id']
	df_4_final = pd.concat([df_4.iloc[:, 0:5], df_dummy, df_4.iloc[:, 5:8]], axis=1)
	df_4_final = df_4_final.dropna(axis=0)  # 5和76。 12和45

	# 分割训练集和测试集
	df_4_train = df_4_final.iloc[:355, :]
	df_4_pred = df_4_final.iloc[355:, :]
	cX_train, cX_test, cy_train, cy_test = train_test_split(df_4_train.iloc[:, :-3], df_4_train.iloc[:, -3:-2],
														test_size=test_size, random_state=13)
	# 用GBDT对真正测试集进行是否会玩游戏的判断
	from sklearn.metrics import accuracy_score
	from sklearn.ensemble import GradientBoostingClassifier
	clf = GradientBoostingClassifier(learning_rate=0.15,n_estimators=160)  ### GBDT(Gradient Boosting Decision Tree) Classifier  #0.694

	clf.fit(cX_train, cy_train)
	y_testpred = clf.predict(cX_test)
	print(accuracy_score(cy_test, y_testpred, normalize=True, sample_weight=None))
	df_4_pred['pred_play'] = clf.predict(df_4_pred.iloc[:, :350])

	# 选用非0游戏时间的数据来拟合游戏时间
	df_5 = df_4_final.copy()
	X_train, X_test, y_train, y_test = train_test_split(df_5.iloc[:, :5], df_5.iloc[:, -2:-1], test_size=test_size,random_state=20)

	# 再回归拟合游戏时间
	from sklearn import linear_model
	reg = linear_model.LinearRegression()  # 加载线性逻辑回归模型
	reg.fit(X_train, y_train)
	logi_y_predict = reg.predict(X_test)
	# tttt=reg.predict(df_test_1)
	print(mean_squared_error(y_test, logi_y_predict) ** (1 / 2))
	df_4_pred['pred_playtime'] = reg.predict(df_4_pred.iloc[:, :5])

	def playtime_trans(x):
		if x[1]>0:
			return x[1]
		else:
			return 0

	pred_list = list(zip(list(df_4_pred['pred_play']), list(round(df_4_pred['pred_playtime'], 5))))

	final_list = []
	for i in range(90):
		if i < 12:
			final_list.append([i, playtime_trans(pred_list[i])])
		elif i == 12:
			final_list.append([12, 0])
		elif i > 12 and i < 45:
			final_list.append([i, playtime_trans(pred_list[i - 1])])
		elif i == 45:
			final_list.append([45, 0])
		else:
			final_list.append([i, playtime_trans(pred_list[i - 2])])
	#print(final_list)

	final_df = pd.DataFrame(final_list, columns=['id', 'playtime_forever'])
	save_path='/Users/chenpengyao/Desktop/HKUST/BDT/MSBD_5001_Foundation_of_DA/individual_project/M2_pred_'+str(test_size)+'.csv'
	final_df.to_csv(save_path,index=None)
	print('Final!!')

if __name__ == "__main__":
	#自测试
	MSBD5001_IP(test_size=0.25)
	#用99%的数据训练模型
	MSBD5001_IP(test_size=0.01)





