import numpy as np
from keras.layers import Activation, Dense
from keras.models import Sequential
from numpy import genfromtxt

X = genfromtxt('./data.csv',
               delimiter=',',
               skip_header=1,
               usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
               dtype=int)

Y = genfromtxt('./data.csv',
               delimiter=',',
               skip_header=1,
               usecols=(15),
               dtype=int)

model = Sequential()

model.add(Dense(input_dim=15, output_dim=30))
model.add(Activation("relu"))
model.add(Dense(output_dim=1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

model.fit(X, Y, nb_epoch=1000, batch_size=32)

results = model.predict_classes(np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
))

print(results)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# save model
model.save_weights('./learnedWeights')