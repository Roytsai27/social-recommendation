import pandas as pd 
import numpy as np 
import networkx as nx 
from node2vec import Node2Vec
import json

raw = pd.read_csv("./data/checkins.txt",names=["User","Location","time"],nrows=500000).sort_values(by="time")
split_num = int(len(raw)*0.8) # split data by time for validation 
check , test = raw.iloc[:split_num] , raw.iloc[split_num:]
friends = pd.read_csv("./data/social_graph.txt",names=["U1","U2","fri"])
friends = friends.drop("fri",1)

# build social graph
F = nx.Graph()
F.add_edges_from(friends.values)
F = F.to_directed()

#link user and location using checkin data 
user_to_loc = nx.DiGraph()
user_to_loc.add_edges_from(check[["User","Location"]].values)

#combine two graphs
Combined = nx.compose(user_to_loc,F)
Combined = nx.convert_node_labels_to_integers(Combined,label_attribute="old")
node_labels = {i:v["old"] for i ,v in Combined.nodes(data=True)}

#process the embeddings
emd = pd.read_table("./models/combine.embeddings",sep=" ",header=None,skiprows=1,index_col=0)
vec_ind = dict()
for ind , v in emd.iterrows():
    label = node_labels.get(ind)
    vec_ind[label] = v.values.tolist()

with open('./models/pretrain_emd.json', 'w') as fp:
    json.dump(vec_ind,fp)


'''
here's for calculate the embeddings
nx.write_adjlist(Combined,"./models/combine.adjlist")
nx.write_gpickle(Combined,"./models/combine.pkl")
nx.write_edgelist(Combined,"./models/combine.edgelist")
node2vec = Node2Vec(Combined, dimensions=100, walk_length=16, num_walks=100,workers=4)
model = node2vec.fit(window=10, min_count=10, batch_words=64)
'''

