import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas
from tensorflow import keras
from keras import layers, Input
from sklearn.model_selection import train_test_split
from keras.layers import Rescaling

df = pandas.DataFrame(np.load("Main/Data.npy",allow_pickle=True))
df.columns = ["data", "genre"]
dat = [i.x for i in df["data"].to_list()]

# Specifies inputs as 64x64 images with RGB values. Does not take in any other inputs e.g. image metadata
# as flickr query is enough for the model.

def create_model(lr) -> keras.Model:
    lr = lr
    inputs = keras.Input(shape = (128, 128, 3))
    sub_layers = 4
    x = Rescaling(scale=1.0 / 255)(inputs)

    for i in range(sub_layers):
        x = tf.keras.layers.Conv2D(
            filters = 12,
            kernel_size = 3,
            strides=(1, 1),
            activation="relu",
        )(x)
        if i < sub_layers-1:
            x = layers.MaxPooling2D(pool_size=(3,3))(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    #
    n = 12
    #
    outputs = layers.Dense(n, activation="softmax")(x)
    optimizer, loss = (keras.optimizers.legacy.Adam(learning_rate = lr), tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

    return model

x = np.array(dat)
y = np.array(df['genre'].to_list())
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(x, y)
valdata = (X_test, y_test)
batch_size = 48
lrs = [1e-3, 1e-2]
epochs = [150, 150]
losses = []
accs = []

for lr, epoc in zip(lrs, epochs):
    model = create_model(lr)
    history = model.fit(X_train, y_train, validation_data = valdata, batch_size = batch_size, epochs = epoc)
    model.save("Models/")
    losses.append(history.history['val_loss'])
    accs.append(history.history['val_accuracy'])

fig, ax = plt.subplots(len(lrs) + 1, figsize = (5, 12.5))

slopes = []

for i in range(len(lrs)):
    x = np.log(np.array(range(epochs[i])))
    y = np.polyfit(x, losses[i], 1)
    ax[i].scatter(range(epochs[i]), losses[i], c = 'r')
    ax[i].scatter(range(epochs[i]), np.array(accs[i])*100)
    ax[i].plot(x, y, c = 'yellow')
    ax[i].set_ylabel("Loss")
    ax[i].set_xlabel("Step")
    slopes.append(1/x)
    
ax[len(lrs)].plot(range(len(epochs[1])), slopes[1])
plt.tight_layout()
plt.show()