import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data


device = ("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.FloatTensor

#创建语料库
sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
        "dog cat animal", "cat monkey animal", "monkey dog animal"]

#数据处理
sentence_list = " ".join(sentences).split()
vocab = list(set(sentence_list))
word2idx = {w:i for i,w in enumerate(vocab)}
vocab_size = len(vocab)

#构建输入输出
window_size = 2
skip_grams = []
for idx in range(window_size,len(sentence_list)-window_size):
    center = sentence_list[idx]
    contexts_idx = list(range(idx - window_size, idx))+list(range(idx + 1, idx + window_size + 1))
    contexts = [sentence_list[i] for i in contexts_idx]
    for context in contexts:
        skip_grams.append([center, context])

def make_data(skip_grams):
    input_data = []
    output_data = []
    for skip_gram in skip_grams:
        input_data.append(np.eye(vocab_size)[word2idx[skip_gram[0]]])
        output_data.append(word2idx[skip_gram[1]])
    return input_data,output_data
'''数据处理到此完毕。下面开始构建网络'''

batch_size = 4
input_data, output_data = make_data(skip_grams)
input_data, output_data = torch.Tensor(input_data), torch.LongTensor(output_data)
data = Data.TensorDataset(input_data, output_data)
loader = Data.DataLoader(data, batch_size=batch_size, shuffle=True)

embedding_dim = 2
class Word2vec(nn.Module):
    def __init__(self):
        super(Word2vec, self).__init__()
        self.w_1 = nn.Parameter(torch.randn(vocab_size,embedding_dim).type(dtype)).requires_grad_()
        self.w_2 = nn.Parameter(torch.randn(embedding_dim,vocab_size).type(dtype)).requires_grad_()

    def forward(self, x):
        x = torch.mm(x, self.w_1)
        x = torch.mm(x, self.w_2)
        return x

model = Word2vec().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 1e-3)

for epoch in range(10000):
    for i, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if((epoch + 1) % 1000) == 0:
            print(epoch + 1, i, loss.item())

for i, label in enumerate(vocab):
    W, WT = model.parameters()
    x, y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()


