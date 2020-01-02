import dynet as dy
import time
from tqdm import tqdm
import numpy as np
from copy import deepcopy


def iterate_minibatches(data, batch_size=32, percent_masking=0.1, shuffle = True):
    ind = np.arange(len(data))
    if shuffle:
        np.random.shuffle(ind)
    data = np.array(data)
    
    for j in range(0, len(data), batch_size):
        data_batch = data[ind[j:j + batch_size]]
        
        if len(data.shape) == 3:
            # n * 4 * len(sent)
            tokens_batch, parents_batch, children_batch, node_order_batch = np.transpose(data_batch, (1, 0, 2))
            lens = np.cumsum([0] + [tokens_batch.shape[1] for _ in range(tokens_batch.shape[0] - 1)])
            lens = lens.reshape(tokens_batch.shape[0], -1)
            
            tokens = tokens_batch.ravel()
            parents = (parents_batch + lens).ravel()
            children = (children_batch + lens).ravel()
            node_order = (node_order_batch + lens).ravel()
        
        else:
            tokens_batch, parents_batch, children_batch, node_order_batch = data_batch.T
            lens = np.cumsum([0] + [len(item) for item in tokens_batch[:-1]])
        
            tokens = np.hstack(tokens_batch)
            parents = np.hstack(parents_batch + lens)
            children = np.hstack(children_batch + lens)
            node_order = np.hstack(node_order_batch + lens)
        
        if_replace = np.random.random(len(tokens))
        mask1 = (if_replace < percent_masking) * ((np.arange(len(tokens)) + 1))
        mask2 = (if_replace < percent_masking * 1.5) * (np.arange(len(tokens)) + 1)
        inds_for_loss = mask2[mask2 > 0] - 1
        input_tokens = deepcopy(tokens)
        input_tokens[mask1[mask1 > 0] - 1] = 1
        
        yield tokens, parents, children, node_order, input_tokens, inds_for_loss


def train_minibatches(model, optimizer, data, batch_size=32, percent_masking=0.1, shuffle = True):
    t_start = time.time()    
    losses_epoch = []
    for tokens, parents, children, node_order, input_tokens, inds_for_loss in tqdm(iterate_minibatches(data, batch_size, percent_masking, shuffle)):
        if not len(inds_for_loss):
            continue
        dy.renew_cg()
        outs = model.forward(input_tokens, parents, children, node_order, inds_for_loss)
        tokens_for_loss = tokens[inds_for_loss]
        loss_iter = dy.esum(list(map(lambda x: dy.pickneglogsoftmax(outs[x], tokens_for_loss[x]), np.arange(len(inds_for_loss))))) / len(inds_for_loss)
        loss_iter.forward()
        loss_iter.backward()
        optimizer.update()
        losses_epoch.append(loss_iter.value())
    t_end = time.time()
    return f'loss: {np.mean(losses_epoch)}, time: {t_end - t_start}'
