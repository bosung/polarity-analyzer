import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext.vocab as vocab

from random import shuffle
from sklearn.model_selection import KFold

word_pairs = []

###########
# get input
# #########
f = open('polarity_words.txt', 'r')
for line in f.readlines():
    #print(line)
    if line[0] == '#' or line[0] == '' or line[0] == '\n':
        continue
    argm, pred, pol = [x.strip() for x in line.split(" ")]
    word_pairs.append([argm, pred, pol])

f.close()

####################
# get word embedding
####################
glove = vocab.GloVe(name='6B', dim=300)
#fasttext = vocab.FastText(language='en')


def get_glove_vector(word):
    try:
        return glove.vectors[glove.stoi[word]]
    except KeyError:
        return torch.FloatTensor(torch.randn(300))


def get_ft_vector(word):
    try:
        return fasttext.vectors[fasttext.stoi[word]]
    except KeyError:
        return torch.FloatTensor(torch.randn(300))


#################
# define networks
# ###############
class PolarityClassifier(nn.Module):

    def __init__(self, word_embedding_dim):
        super(PolarityClassifier, self).__init__()
        self.linear = nn.Linear(word_embedding_dim, 2)

    def forward(self, inputs):
        out1 = F.relu(self.linear(inputs))
        return F.softmax(out1, dim=0)


#######
# train
# #####
def train(train_index, comp_function, epoch_size):
    model = PolarityClassifier(300)
    loss_function = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(epoch_size):
        total_loss = 0
        for i in train_index:
            args, pred, pol = word_pairs[i]
            args = get_glove_vector(args)
            pred = get_glove_vector(pred)
            pol = float(pol)

            model.zero_grad()
            optimizer.zero_grad()

            # add
            comp = autograd.Variable(comp_function(args, pred))
            tag_score = model(comp)
            pol = torch.FloatTensor((1, 0)) if pol == 1 else torch.FloatTensor((0, 1))
            target = autograd.Variable(pol)

            loss = loss_function(tag_score, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.data
        #print("epoch: %d\ttotal_loss: %.3f" % (epoch, total_loss))

    return model


def test(test_index, model, comp_function):
    ans_cnt = 0
    for i in test_index:
        args, pred, pol = word_pairs[i]
        args = get_glove_vector(args)
        pred = get_glove_vector(pred)
        pol = float(pol)

        comp = autograd.Variable(comp_function(args, pred))
        tag_score = model(comp)

        v, i = torch.max(tag_score, -1)
        if i.stride(0) == 0 and pol == 1:
            ans_cnt += 1
        elif i.stride(0) == 1 and pol == 0:
            ans_cnt += 1

    accuracy = (ans_cnt/len(test_index)*100)
    print("accuracy: %.3f" % accuracy)
    return accuracy


shuffle(word_pairs)

kf = KFold(n_splits=10)
kf.get_n_splits(word_pairs)


#test 1. additive model

#def comp_function(x, y): return torch.add(x, y)
def comp_function(x, y): return torch.FloatTensor(np.multiply(np.array(x), np.array(y)))

total_acr = 0
for train_index, test_index in kf.split(word_pairs):
    #print("TRAIN:", train_index, "TEST:", test_index)
    model = train(train_index, comp_function, 20)
    acr = test(test_index, model, comp_function)
    total_acr += acr
print("test %d\taverage accuracy: %.3f" % (1, total_acr/10))


# test 2. multiplicative

#def comp_function(x, y): return torch.FloatTensor(np.multiply(np.array(x), np.array(y)))

#total_acr = 0
#for train_index, test_index in kf.split(word_pairs):
    #print("TRAIN:", train_index, "TEST:", test_index)
#    model = train(train_index, comp_function, 100)
#    acr = test(test_index, model, comp_function)
#    total_acr += acr
#print("test %d\taverage accuracy: %.3f" % (1, total_acr/10))

