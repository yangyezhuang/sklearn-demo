from sklearn.preprocessing import LabelBinarizer

# 将文字标签转换为数字
feature = [[0, 1], [1, 1], [0, 0], [1, 0]]
label = ['是', '否', '是', '否']
lb = LabelBinarizer()  # 构建一个转换对象
Y = lb.fit_transform(label)
re_label = lb.inverse_transform(Y)
print(Y)
print(Y.flatten())  # 降维
print(re_label)
