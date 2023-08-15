import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from eppa_exp1jr_scenario_discovery_main import SD

sd_obj = SD("GLB_RAW", "REF_GLB_RENEW_SHARE")
X = sd_obj.get_X()
print(X.shape)
Y = sd_obj.get_y_by_year(2050)
print(Y.shape)
X_normalized = X / np.max(np.abs(X))
Y_normalized = Y / np.max(np.abs(Y))

def train_and_test(X_normalized, Y_normalized):
    X_train, X_val, Y_train, Y_val = train_test_split(X_normalized, Y_normalized, test_size=0.3)

    model = tf.keras.Sequential([
        layers.Input(shape=(53,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    history = model.fit(X_train, Y_train, epochs=100, validation_data=(X_val, Y_val))

    val_loss, val_mae = model.evaluate(X_val, Y_val)
    print("Validation loss:", val_loss)
    print("Validation mean absolute error:", val_mae)

    predicted = model.predict(X_val)
    print(predicted)
    score = r2_score(Y_val, predicted)

    print(score)

train_and_test(X_normalized, Y_normalized)