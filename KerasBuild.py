import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_batch_ops import batch 
import tensorflow_datasets as tfds


def plotGraphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

dataset, info = tfds.load('imdb_reviews', with_info = True, as_supervised = True)
train_data, test_data = dataset['train'], dataset['test']

bufferSize = 10000
batchSize = 64 
vocabSize = 1000

train_data = train_data.shuffle(bufferSize).batch(batchSize).prefetch(tf.data.AUTOTUNE)
test_data = test_data.batch(batchSize).prefetch(tf.data.AUTOTUNE)

encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocabSize)
encoder.adapt(train_data.map(lambda text, label: text))

vocab = np.array(encoder.get_vocabulary()) 

model = tf.keras.Sequential([encoder, tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=64, mask_zero=True), 
                            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)), 
                            tf.keras.layers.Dense(64, activation = 'relu'), 
                            tf.keras.layers.Dense(1)])


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])

#model = tf.keras.models.load_model('kerasModel') Option for loading in the Keras model rather than going through the training iterations
history = model.fit(train_data, epochs=10,
                    validation_data=test_data,
                    validation_steps=30)

# model.save("kerasModel")
test_loss, test_acc = model.evaluate(test_data)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plotGraphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plotGraphs(history, 'loss')
plt.ylim(0, None)
plt.show()

# sample = ('I love this movie. This movie was amazing. my favourite movie')
# predictions = model.predict(np.array([sample]))
# print(predictions)

# Currently gives an output between 2 and -2 so that's fun. Normalise it and graph it!