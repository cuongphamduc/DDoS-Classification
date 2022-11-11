import datetime

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

from densenet.classifiers.one_d import DenseNet121

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
# print(train_data["label"].value_counts())

s = ["normal", "back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb", "processtable", "udpstorm",
     "apache2",
     "worm"]
train_data = train_data.loc[train_data["label"].isin(s)]
# print(train_data["label"].value_counts())

multi_data = train_data.copy()
multi_label = pd.DataFrame(multi_data.label)

std_scaler = StandardScaler()


def standardization(df, col):
    for i in col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = std_scaler.fit_transform(arr.reshape(len(arr), 1))
    return df


numeric_col = multi_data.select_dtypes(include="number").columns
data = standardization(multi_data, numeric_col)

le2 = preprocessing.LabelEncoder()
enc_label = multi_label.apply(le2.fit_transform)
multi_data["intrusion"] = enc_label

multi_data.drop(labels=["label"], axis=1, inplace=True)
multi_data = pd.get_dummies(multi_data, columns=["protocol_type", "service", "flag"], prefix="", prefix_sep="")
y_train_multi = multi_data[["intrusion"]]
X_train_multi = multi_data.drop(labels=["intrusion"], axis=1)
X_train_multi = np.expand_dims(X_train_multi, 2)

X_train_multi = X_train_multi[:1, :, :]

VGG = load_model("model/VGG.h5")

# warm up GPU
_ = VGG.predict(X_train_multi)

total = 0
for _ in range(10):
    a = datetime.datetime.now()
    _ = VGG.predict(X_train_multi)
    b = datetime.datetime.now()

    total += (b - a).microseconds / 1000

print("VGG inference time : ", total / 10, "ms")

DenseNet = load_model("model/DenseNet.h5", custom_objects={'DenseNet121': DenseNet121})
_ = DenseNet.predict(X_train_multi)

total = 0
for _ in range(10):
    a = datetime.datetime.now()
    _ = DenseNet.predict(X_train_multi)
    b = datetime.datetime.now()

    total += (b - a).microseconds / 1000

print("DenseNet inference time : ", total / 10, "ms")
