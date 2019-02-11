import json
import warnings
from datetime import datetime

import keras.backend as K
import keras.utils as ku
import numpy as np
import pandas as pd
import tensorflow as tf
from bidict import bidict
from keras.callbacks import EarlyStopping
from keras.layers import (GRU, LSTM, Add, Concatenate, Dense, Dropout,
                          Embedding, Flatten, Layer, Masking, merge,RepeatVector,TimeDistributed)
from keras.layers.core import *
from keras.layers.merge import Concatenate, concatenate
from keras.models import *
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy
from project_utils import *

warnings.filterwarnings("ignore")


raw = pd.read_csv("./data/checkins.txt",names=["User","Location","time"],nrows=500000).sort_values(by="time")
split_num = int(len(raw)*0.8) # split data by time for validation 
check , test = raw.iloc[:split_num] , raw.iloc[split_num:]
# check = check.drop("time",1)
# test  = test.drop("time",1)

def max_len(user_group,max_timestep=298,threshold = 1.5): # 
    ml = 0  #init max_seq_len  
    for _, v in user_group:
        t = 0 #tmp for split 
        i = 0 # index
        for _ ,k in v.iterrows():
            if k["diff"] > threshold :
                data = v.loc[t:i]
                l = data.shape[0]
                if l > ml :
                    ml = l 
                t = i 
            i+=1
        data = v.iloc[t:i]
        l = data.shape[0]
        if l > ml  and l<= max_time_step:
            ml = l 
    return ml 


def precess_time(df):
    df["Next"] = df[["time"]].shift(-1)
    df["Next"] = df["Next"].fillna(df["time"].values[-1])
    df["diff"] = (df["Next"] - df["time"]) / np.timedelta64(1,"D")
    return df 


def generate_timeinfo(token_list):
    res = []
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        res.append(n_gram_sequence)
    pad = np.array(pad_sequences(res,maxlen=max_sequence_len))[:,:-1].tolist()
    onehot_res = []
    for i in pad:
        t = []
        for b in i:
            one_n = [0]*bin_length
            if b == 0: #no input 
                t.append(one_n)
            else:
                one_n[b-1] = 0
                t.append(one_n)
        onehot_res.append(t)
    return onehot_res 


def user_history(df,bin_length,threshold = 2):
    #separate the history with a threshold 
    res = []
    bin_onehot = []
    tmp = 0
    i = 0
    for _ ,v in df.iterrows():
        if v["diff"] > threshold :
            data = df.iloc[tmp:i]
            res.append(data["Location"].values.tolist())
            tmp = i
            #create one hot encoding for time
            tb = data["time_bin"].values
            bin_onehot.append(ku.to_categorical(tb,num_classes=bin_length).tolist())
        i+=1
    data = df.iloc[tmp:i]
    res.append(data["Location"].values.tolist())
    #create one hot encoding for time
    tb = data["time_bin"].values
    bin_onehot.append(ku.to_categorical(tb,num_classes=bin_length).tolist())
    return res,bin_onehot


def time_bin(ts,split_hours = 2):
    ts = (ts - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    hours = datetime.utcfromtimestamp(ts).time().hour
    time_bin = 0  #bin starts from zero 
    current_time = 0 
    while hours > current_time:
        if hours < current_time + split_hours:
            break 
        current_time += split_hours
        time_bin += 1
    return time_bin


def recall_k(y_true,y_predict):
    y_true = list(set(y_true)) #avoid repeat checkins
    hit = sum([1 for i in y_predict if i in y_true ])
    return hit/len(y_true)


def precision_k(y_true,y_predict):
    y_true = list(set(y_true)) #avoid repeat checkins
    hit = sum([1 for i in y_predict if i in y_true ])
    return hit / len(y_predict)

check["time"] =  pd.to_datetime(check["time"])
check = precess_time(check)
check["time_bin"] =  check["time"].apply(time_bin)
user_group = check.groupby(by="User")


#parameters
split_hour = 2 
bin_length = int(24 / split_hour)
threshold = 1.5
max_time_step = 298
max_sequence_len = max_len(user_group,max_time_step,threshold)
# top10 = lambda x, y: sparse_top_k_recall(x, y, k=10)
top10 = lambda x, y: top_k_categorical_accuracy(x, y, k=10)

# add user embedding
user_ind = dict()
total_user = 0 
for i,v in enumerate(check.User.unique()):
    user_ind[v] = i
    total_user+=1

# use pretrain model for user embeddings 
with open('./models/pretrain_emd.json', 'r') as f:
    user_vec = json.load(fp=f)
weight_matrix = np.zeros((total_user,64))
for user , ind in user_ind.items():
    weight_matrix[ind] = user_vec.get(user)


history_str = []
user_embed_inp = []
time_onehot = []
for user,v in user_group:
    his, time_oh= user_history(v,bin_length,threshold=1.5) # split history record with 1.5 days 
    if max([len(i) for i in his]) > max_time_step: # avoid too long history for rnn 
        continue
    time_onehot.extend(np.array(pad_sequences(time_oh,   
                      maxlen=max_sequence_len-1, padding='pre')).tolist())
    for h in his: 
        history_str.append(" ".join(h)) # represent history chechins as contexts             
    size = len(his)
    for _ in range(size):
        user_embed_inp.append([user_ind[user]])

user_embed_inp = np.array(user_embed_inp)
time_onehot =  np.array(time_onehot)

tokenizer = Tokenizer()  
tokenizer.fit_on_texts(history_str)
total_words = len(tokenizer.word_index) + 1


# process the history
input_sequences = []
for line in history_str:
    token_list = tokenizer.texts_to_sequences([line])[0]
    input_sequences.append(token_list)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences,   
                      maxlen=max_sequence_len, padding='pre'))

predictors, sparse_label = input_sequences[:,:-1],input_sequences[:,-1]
each_label = input_sequences[:,1:]
sample_size, INPUT_DIM = predictors.shape

#pretrain matrix
word_index = tokenizer.word_index
loc_emd = np.zeros((len(word_index) + 1, 64))
for word, i in word_index.items():
    embedding_vector = user_vec.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        loc_emd[i] = embedding_vector

print("Total sample %s, Max time steps: %s" %(predictors.shape[0],predictors.shape[1]))
print("User Embedding shape:",user_embed_inp.shape)
print("Time one_hot",time_onehot.shape)
print(total_words)
print(max_sequence_len)

# data for testing set 
future_user = test["User"].unique()
test["time"] =  pd.to_datetime(test["time"])
test = precess_time(test)
test["time_bin"] =  check["time"].apply(time_bin)
future_user_group = test.groupby(by = "User")

test_history_str = []
test_user_embed_inp = []
test_time_onehot = []
test_lbl = []
for user in future_user:
    try:
        v = user_group.get_group(user)
    except:
        continue
    #create labels
    f = future_user_group.get_group(user)
    locs = " ".join(f["Location"].values.tolist()) 
    lbls = tokenizer.texts_to_sequences([locs])[0]
    if lbls == []:
        continue
    # split history record with 1.5 days 
    his, time_oh= user_history(v,bin_length,threshold=1.5) 

    # avoid too long history for rnn 
    if max([len(i) for i in his]) > max_time_step: 
        continue
    for l in lbls[:1]: 
        test_lbl.append(l)
        test_time_onehot.extend(np.array(pad_sequences(time_oh,   
                      maxlen=max_sequence_len-1, padding='pre')).tolist())
        for h in his: 
            test_history_str.append(" ".join(h)) # represent history checkins as contexts             
        size = len(his)
        for _ in range(size):
            test_user_embed_inp.append([user_ind[user]])

predict_sequences = []
for line in test_history_str:
    token_list = tokenizer.texts_to_sequences([line])[0]
    predict_sequences.append(token_list)

max_sequence_len = max([len(x) for x in predict_sequences])
predict_sequences = np.array(pad_sequences(predict_sequences,   
                      maxlen=max_sequence_len, padding='pre'))[:,1:]

test_user_embed_inp = np.array(test_user_embed_inp)
test_time_onehot = np.array(test_time_onehot)
test_lbl = np.array(test_lbl)
print(test_lbl.shape)
print(predict_sequences.shape)
print(test_user_embed_inp.shape)
print(test_time_onehot.shape)


#define user embedding model 
input_len = max_sequence_len - 1
user_emb_inp = Input(shape=(1,))
user_emb = Embedding(total_user,64,input_length=1,weights=[weight_matrix],trainable=False)(user_emb_inp)
user_emb = Flatten()(user_emb)
user_emb = RepeatVector(input_len)(user_emb)

#adding time information 
time_inp = Input(shape=(max_sequence_len-1,bin_length))

# rnn model 
rnn_inp = Input(shape=(input_len,))
rnn_emb = Embedding(total_words, 64, input_length=input_len,mask_zero=True,weights=[loc_emd],trainable=True)(rnn_inp)

#concate time and location 
time_rnn = Concatenate()([rnn_emb,time_inp])
time_rnn = Dense(64)(time_rnn)


# rnn = GRU(150)(time_rnn)
rnn = GRU(150,return_sequences=True)(time_rnn)
# rnn = AttentionWithContext()(rnn)
rnn = TimeDistributed(Dense(64,activation="relu"))(rnn)

#merge 
model = Add()([rnn,user_emb])
model = Dense(total_words, activation='softmax')(model)

#main model 
main = Model(inputs=[rnn_inp,user_emb_inp,time_inp],outputs=model)
main.compile(loss="sparse_categorical_crossentropy", optimizer='adam',metrics=["acc"])
main.summary()
# main = load_model("./models/rnn3.h5")
main.fit([predictors,user_embed_inp,time_onehot], each_label, 
          epochs=20,batch_size=128,
        #   validation_data=([predict_sequences,test_user_embed_inp,test_time_onehot],test_lbl),
          )
predict_result = main.predict([predict_sequences,test_user_embed_inp,test_time_onehot])
# main.save("./models/rnn3.h5")


'''


recalls = []
precisions = []
k = 10
future_user_group = test.groupby(by = "User")
loc_dict = bidict(tokenizer.word_index)
for user, res in zip(future_user,predict_result):
    try:
        v = user_group.get_group(user)
    except:
        continue
    f = future_user_group.get_group(user)
    locs = f["Location"].values.tolist()
    top_k = np.argsort(res)[::-1]
    top_locations = []
    for i in top_k:
        if len(top_locations) == k :
            break
        try:
            l = loc_dict.inv[i]
        except:
            continue
        top_locations.append(l)
    assert len(top_locations) == k
    recall_per = recall_k(locs,top_locations)
    precision_per = precision_k(locs,top_locations)
    recalls.append(recall_per) 
    precisions.append(precision_per)


print(np.mean(recalls))
print(np.mean(precisions))
'''