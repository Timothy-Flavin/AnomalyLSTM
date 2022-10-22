import tensorflow as tf
# define model
def get_model(num_inputs, units=50, seq_length=5):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.LSTM(units, activation='relu', input_shape=(seq_length, num_inputs)))
  model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
  model.compile(optimizer='adam', loss='mse') #opt='adam' and loss='mse' optimizer='sgd', loss='mae'
  return model


def get_auto_encoder(num_inputs, units=[64,64], seq_length=5, stateful = False):
  print(num_inputs)
  #input("enter to move on...")
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.LSTM(units[0], input_shape=(None, num_inputs), stateful=stateful)) # seq_length
  model.add(tf.keras.layers.Dropout(rate=0.2))
  #model.add(tf.keras.layers.LSTM(units[1], return_sequences=False)) 
  # to use this layer, set return sequence = True in the previous LSTM layer
  model.add(tf.keras.layers.RepeatVector(seq_length))
  #model.add(tf.keras.layers.LSTM(units[1], return_sequences=True))
  model.add(tf.keras.layers.LSTM(units[0], return_sequences=True))
  model.add(tf.keras.layers.Dropout(rate=0.2))
  model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_inputs)))
  model.compile(optimizer='adam', loss='mae')
  model.summary()
  return model