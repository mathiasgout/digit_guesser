from tensorflow import keras
from tensorflow.keras.datasets import mnist

MODEL_PATH = "models/model_conv2d.hdf5"


""" Importation des données """
(X_train, y_train), (X_test, y_test) = mnist.load_data()


""" Mise en forme des données"""
def data_preparation(X_train, y_train, X_test, y_test):
    
    # (n_ind, n_row, n_col) -> (n_ind, n_row, n_col, 1) (1 car images en noir et blanc)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) 
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)  

    # Intervalle de la valeur des variables : {0,1,..,255} -> [0, 1] 
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train = X_train / 255
    X_test = X_test / 255

    # On transforme y_train et y_test en vecteurs one hot
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = data_preparation(X_train, y_train, X_test, y_test)


""" Création du modèle """
def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(28,28,1), padding="same"))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))
    
    model.summary()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model
    

""" Entrainement """
model = build_model()
checkpoint = keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=50, restore_best_weights=True, mode="min")
model.fit(X_train, y_train, epochs=500, batch_size=128, validation_data=(X_test,y_test), use_multiprocessing=True, callbacks=[early_stop,checkpoint])
