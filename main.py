import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint

from densenet.classifiers.one_d import DenseNet121

model_name = ""


def VGG_custom():
    global model_name
    model_name = "VGG"

    model = Sequential()

    model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu", input_shape=(116, 1)))
    model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool1D(pool_size=2, strides=2))

    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool1D(pool_size=2, strides=2))

    model.add(Conv1D(filters=256, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=256, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool1D(pool_size=2, strides=2))

    model.add(Conv1D(filters=512, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=512, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool1D(pool_size=2, strides=2))

    model.add(Conv1D(filters=512, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=512, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool1D(pool_size=2, strides=2))

    model.add(Flatten())

    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=7, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def DenseNet_custom():
    global model_name
    model_name = "DenseNet"

    model = DenseNet121(input_shape=(116, 1), num_outputs=7)

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


if __name__ == "__main__":
    feature = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
               "urgent",
               "hot",
               "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
               "num_file_creations", "num_shells",
               "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count",
               "serror_rate", "srv_serror_rate",
               "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
               "dst_host_count",
               "dst_host_srv_count",
               "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
               "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
               "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]

    train = "./data/nsl-kdd/KDDTrain+.txt"
    train_data = pd.read_csv(train, names=feature)
    train_data.drop(["difficulty"], axis=1, inplace=True)
    print(train_data["label"].value_counts())

    s = ["normal", "back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb", "processtable", "udpstorm",
         "apache2",
         "worm"]
    train_data = train_data.loc[train_data["label"].isin(s)]
    print(train_data["label"].value_counts())

    multi_data = train_data.copy()
    multi_label = pd.DataFrame(multi_data.label)

    std_scaler = StandardScaler()

    numeric_col = multi_data.select_dtypes(include="number").columns

    le2 = preprocessing.LabelEncoder()
    enc_label = multi_label.apply(le2.fit_transform)
    multi_data["intrusion"] = enc_label

    multi_data.drop(labels=["label"], axis=1, inplace=True)
    multi_data = pd.get_dummies(multi_data, columns=["protocol_type", "service", "flag"], prefix="", prefix_sep="")
    y_train_multi = multi_data[["intrusion"]]
    X_train_multi = multi_data.drop(labels=["intrusion"], axis=1)
    X_train_multi = np.expand_dims(X_train_multi, 2)

    print("X_train has shape:", X_train_multi.shape, "\ny_train has shape:", y_train_multi.shape)

    y_train_multi = LabelBinarizer().fit_transform(y_train_multi)

    net = DenseNet_custom()
    # net = VGG_custom()
    net.summary()

    checkpointer = ModelCheckpoint(
        monitor="val_accuracy",
        filepath="model/" + model_name + ".h5",
        verbose=1,
        save_best_only=True)

    plot_model(net, to_file="image/" + model_name + ".png", show_shapes=True, show_layer_names=True)

    history = net.fit(X_train_multi, y_train_multi, epochs=10, batch_size=512, validation_split=0.2,
                      callbacks=[checkpointer])

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Plot of accuracy vs epoch for train and test dataset")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.savefig("image/" + model_name + "_acc_plot.png")
    plt.show()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Plot of loss vs epoch for train and test dataset")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.savefig("image/" + model_name + "_loss_plot.png")
    plt.show()
