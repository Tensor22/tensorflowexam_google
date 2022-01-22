import tensorflow as tf
import timeit

class SequentialModel(tf.keras.Model):
     def __init__(self,**kwargs):
          super(SequentialModel,self).__init__(kwargs)
          self.flatten = tf.keras.layers.Flatten(input_shape=(28,28))
          self.dense_1 = tf.keras.layers.Dense(128,activation='relu')
          self.dropout = tf.keras.layers.Dropout(0.2)
          self.dense_2 = tf.keras.layers.Dense(10)
     def call(self,x):
          x= self.flatten(x)
          x= self.dense_1(x)
          x= self.dropout(x)
          x= self.dense_2(x)
          return x
input_data=tf.random.uniform([60,28,28])
eager_model = SequentialModel()
graph_model = tf.function(eager_model)

print("eager time : ", timeit.timeit(lambda: eager_model(input_data),number=10000))
print("Graph time : ", timeit.timeit(lambda: graph_model(input_data),number=10000))

class Winter:
  print("겨울")
class Summer:
  print("여름")
W=Winter()
S=Summer()
print(isinstance(W,Winter))
print(isinstance(S,Winter))

class A:
    def __init__(self):
        print('init')
    def __call__(self):
        print('call')
a=A()
a()
#print(isinstance(a,A))
