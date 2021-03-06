# -*- coding: utf-8 -*-
import json
import os
import tensorflow as tf
import numpy as np
import math

def get_in_filename():
  dids = json.loads(os.getenv('DIDS', None))

  for did in dids:
      return f'data/inputs/{did}/0'

j = json.load(open(get_in_filename()))

dec = []

for i in j: dec.append(i['deceduti'])

tf.random.set_seed(42)

x_set_max = len(j)
x_trn_max = int(x_set_max*0.8)

x_res = np.reshape(dec,(-1,1))
X_seq = (x_res-min(x_res))/(max(x_res)-min(x_res))

X_set = []
y_set = []

for i in range(20,x_set_max):
  X_set.append(X_seq[i-20:i,0])
  y_set.append(X_seq[i,0])

X_train = np.array(X_set[:x_trn_max])
y_train = np.array(y_set[:x_trn_max])

X_test = np.array(X_set[x_trn_max:])
y_test = np.array(y_set[x_trn_max:])

model = tf.keras.Sequential([
  tf.keras.layers.LSTM(90,dropout=0.2,return_sequences=True,input_shape=[20,1]),
  tf.keras.layers.LSTM(60,dropout=0.2,return_sequences=True),
  tf.keras.layers.LSTM(50,dropout=0.2),
  tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.mean_squared_error
)

history = model.fit(X_train,y_train,epochs=800,verbose=0)

#model.evaluate(X_test,y_test)

y_pred = model.predict(X_test).flatten()

#f = plt.figure(figsize=(22,7))
#
#f1 = f.add_subplot(121)
#
#f1.scatter([x for x in range(0,x_trn_max)],y_train,c="g",label="train")
#f1.scatter([x+x_trn_max+1 for x in range(0,x_set_max-x_trn_max-20)],[y-(y_pred[0]-y_test[0]) for y in y_pred],c="b",label="pred",alpha=0.2)
#f1.scatter([x+x_trn_max+1 for x in range(0,x_set_max-x_trn_max-20)],y_test,c="r",label="actual",alpha=0.2)

r=dict()

for i in range(0,len(X_test)):
  r[i]=(X_test[i].tolist(),y_test[i],y_pred[i]-(y_pred[0]-y_test[0]))

with open('/data/outputs/result', 'w') as results:
  json.dump(r, results)
