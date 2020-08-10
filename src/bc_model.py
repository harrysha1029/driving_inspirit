import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.consts import DIM, INDEX2ACTIONS, N_ACTIONS
from src.driving_utils import get_features_from_env, get_racing_env, run_racing
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Lambda,
    MaxPool2D,
    concatenate,
)
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.optimizers import Adam


# ======MODEL==========
def simple_image_encoder():
    return tf.keras.Sequential(
        [
    Lambda(lambda x: x / 255.0)
    , Conv2D(64, 5, activation="relu")
    , BatchNormalization()
    , Dropout(0.2)
    , Conv2D(64, 3, activation="relu")
    , BatchNormalization()
    , Dropout(0.2)
    , MaxPool2D(2)
    , Conv2D(32, 3, activation="relu")
    , BatchNormalization()
    , Flatten()
        ]
    )

def transfer_image_encoder():
    mobile = tf.keras.applications.MobileNetV2((96, 96, 3), weights='imagenet', include_top=False)
    return tf.keras.Sequential([
        mobile, 
        tf.keras.layers.GlobalAveragePooling2D()
    ])



def get_bc_model(image_encoder=simple_image_encoder):
    imgs_inputs = Input(shape=(96, 96, 3))
    features_inputs = Input(shape=(4,))
    x = image_encoder()(imgs_inputs)
    x =  concatenate([x, features_inputs])
    x =  Dense(64, activation="relu")(x)
    x =  Dropout(0.1)(x)
    x =  Dense(16, activation="relu")(x)
    x =  Dropout(0.1)(x)
    x =  Dense(N_ACTIONS, activation="softmax")(x)
    return Model([imgs_inputs, features_inputs], x)

    

# ========== END MODEL ============


def train_model(model, imgs, features, y, epochs=10):
    model.compile(
        optimizer=Adam(learning_rate=0.0004),
        loss=sparse_categorical_crossentropy,
        metrics=["sparse_categorical_accuracy"],
    )
    (
        imgs_train,
        imgs_test,
        features_train,
        features_test,
        y_train,
        y_test,
    ) = train_test_split(imgs, features, y, test_size=0.2, random_state=1)
    print(imgs_train.shape, features_train.shape, y_train.shape)
    model.fit(
        [imgs_train, features_train],
        y_train,
        epochs=epochs,
        callbacks=[ModelCheckpoint("checkpoints/bc/", save_best_only=True, verbose=1)],
        validation_data=([imgs_test, features_test], y_test),
    )


def model_prob_policy(model, obs, env):
    features = np.array(get_features_from_env(env))[None, ...]
    img = obs[None, ...]
    probs = model.predict((img, features)).squeeze()
    move = np.random.choice(range(N_ACTIONS), p=probs)
    return INDEX2ACTIONS[move]


def model_policy(model, obs, env, speed_min):
    features = get_features_from_env(env)
    print(features[0])
    if features[0] < speed_min:
        return [0, 1, 0]
    features = np.array(features)[None, ...]
    img = obs[None, ...]
    probs = model.predict((img, features)).squeeze()
    move = np.argmax(probs)
    return INDEX2ACTIONS[move]


def run_bc_model(model_path, speed_min=0.5, record_video=False):
    env = get_racing_env(record_video)
    model = load_model(model_path)
    policy = lambda obs, env: model_policy(model, obs, env, speed_min)
    run_racing(env, policy)
