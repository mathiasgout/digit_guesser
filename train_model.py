import os
import datetime
from build_model import build_model
from tensorflow import keras
from tensorflow.keras.datasets import mnist

""" Check for 'models' folder"""
# Check if "models" folder exist
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(DIR_PATH, "models")
if os.path.exists(MODELS_DIR) is False:
    os.mkdir(MODELS_DIR)

# Create model file name based on the current time
now = datetime.datetime.now().strftime("%Y%m%d%H%M")
MODEL_PATH = MODELS_DIR + "/model_" + now + ".hdf5"


""" Imports and data preparation"""
# Importation des données
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# (n_ind, n_row, n_col) -> (n_ind, n_row, n_col, 1) (1 car images en noir et blanc)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Intervale de la valeur des variables : {0,1,..,255} -> [0, 1]
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train = X_train / 255
X_test = X_test / 255

# On transforme y_train et y_test en vecteurs one hot
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


""" Training """
model = build_model()
checkpoint = keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=50, restore_best_weights=True,
                                           mode="min")
model.fit(X_train, y_train, epochs=500, batch_size=128, validation_data=(X_test, y_test), use_multiprocessing=True,
          callbacks=[early_stop, checkpoint])

print("Your model file : {}".format(MODEL_PATH))
