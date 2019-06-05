#导包
import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import feature_column
import tensorflow.keras
from sklearn.model_selection import train_test_split

#pandas 加载数据
URL= "F:\\tensorflow2\\tf2learning\heart_trouble_prediction\heart.csv"
df = pd.read_csv(URL)
#print(df.head(10))
#tes = dict(df.head(3))
#print(tes)

#讲数据划分为测试节和训练集
train, test = train_test_split(df, test_size=0.2)
#将训练集划分为训练集合和测试集
train, val = train_test_split(train, test_size=0.2)
print("train size:"+str(len(train))+"  val size:"+str(len(val))+ "   test size:"+str(len(test)))

#使用tf.data创建输入管道
def df_to_dataset(dataframe, shuffle=True, batch_size =32):
    #copy() 默认参数deep=True ,deep=True原值改变，copy的值也跟着改变, 否则不跟着改变
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds =ds.shuffle(buffer_size= len(dataframe))
    ds =ds.batch(batch_size)
    return ds

batch_size =5
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size= batch_size)

# # 我们将使用此批处理来演示几种类型的特征列
example_batch = next(iter(train_ds))[0]
# #print(example_batch)
#
# # 用于创建特征列和转换批量数据
def demo(feature_column):
  feature_layer = tf.keras.layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())

age = feature_column.numeric_column("age")
# demo(age)

#连续值特征分桶，也即按照不同的不同的阈值one-hot
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)

#分类列，对同一个属性特征的不同类别进行one-hot
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])

thal_one_hot = feature_column.indicator_column(thal)
demo(thal_one_hot)

#嵌入列
'''
假设我们不是只有几个可能的字符串，而是每个类别有数千（或更多）值。由于多种原因，
随着类别数量的增加，使用独热编码训练神经网络变得不可行，我们可以使用嵌入列来克服此
限制。嵌入列不是将数据表示为多维度的独热矢量，而是将数据表示为低维密集向量，其中每
个单元格可以包含任意数字，而不仅仅是0或1.嵌入的大小（在下面的例子中是8）是必须调整
的参数。
关键点：当分类列具有许多可能的值时，最好使用嵌入列，我们在这里使用一个用于演示目的，
因此您有一个完整的示例，您可以在将来修改其他数据集。
'''
thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)

#哈希特征列
'''
表示具有大量值的分类列的另一种方法是使用categorical_column_with_hash_bucket.
此特征列计算输入的哈希值，然后选择一个hash_bucket_size存储桶来编码字符串，使用此列时，
您不需要提供词汇表，并且可以选择使hash_buckets的数量远远小于实际类别的数量以节省空间。
关键点：该技术的一个重要缺点是可能存在冲突，其中不同的字符串被映射到同一个桶，实际上，
无论如何，这对某些数据集都有效。
'''
thal_hashed = feature_column.categorical_column_with_hash_bucket(
      'thal', hash_bucket_size=1000)
demo(feature_column.indicator_column(thal_hashed))

#交叉得到高级特征
'''
将特征组合成单个特征（也称为特征交叉），使模型能够为每个特征组合学习单独的权重。
在这里，我们将创建一个age和thal交叉的新功能，请注意，crossed_column不会构建所有
可能组合的完整表（可能非常大），相反，它由hashed_column支持，因此您可以选择表的大小。
'''
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(feature_column.indicator_column(crossed_feature))

#选择要使用的特征
'''
我们已经了解了如何使用几种类型的特征列，现在我们将使用它们来训练模型。本教程的目标是
向您展示使用特征列所需的完整代码（例如，机制），我们选择了几列来任意训练我们的模型。
关键点：如果您的目标是建立一个准确的模型，请尝试使用您自己的更大数据集，并仔细考虑哪
些特征最有意义，以及如何表示它们。
'''
feature_columns = []
# numeric 数字列
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
  feature_columns.append(feature_column.numeric_column(header))

# bucketized 分桶列
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator 指示符列
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding 嵌入列
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed 交叉列
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

#创建特征层
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

#创建、编译和训练模型
model = tf.keras.Sequential([
  feature_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

#测试
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)