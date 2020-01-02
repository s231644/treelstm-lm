import dynet as dy


class Attention(object):
    """
    A module for both self attention and normal attention with key, query and value
    """
    def __init__(self, model, dm: int, dk: int, dq: int=None):
        # dm = memory dimension
        # dk = key dimension
        # dq = query dimension (None for self-attention)
        dq = dq or dm
        self.w_q = model.add_parameters((dk, dq)) 
        self.w_k = model.add_parameters((dk, dm))
        self.w_v = model.add_parameters((dk, dm))
        self.factor = dk ** 0.5

    def encode(self, memory, query=None):
        query = query or memory # if no query then self attention
        Q = self.w_q * query
        K = self.w_k * memory
        V = self.w_v * memory
        A = dy.softmax(dy.transpose(K) * Q / self.factor)
        out = V * A
        return out


class TreeLSTM(object):
    def __init__(self, model, dm: int, att_type):
        self.model = model
        self.att_type = att_type
        self.WS = [self.model.add_parameters((dm, dm)) for _ in "iouf"]
        self.US = [self.model.add_parameters((dm, dm)) for _ in "iouf"]
        self.BS = [self.model.add_parameters(dm) for _ in "iouf"]
        
        if self.att_type == 'att' or self.att_type == 'selfatt':
            self.attention = Attention(model, dm, dm)
        if self.att_type == 'selfatt':
            self.self_attention = Attention(model, dm, dm)
        
    def state(self, x, hs=[], cs=[]):
        if len(hs) == 0:
            # initial state
            Wi, Wo, Wu, Wf = self.WS
            bi, bo, bu, bf = self.BS

            i = dy.logistic(dy.affine_transform([bi, Wi, x]))
            o = dy.logistic(dy.affine_transform([bo, Wo, x]))
            u = dy.tanh(dy.affine_transform([bu, Wu, x]))
            c = dy.cmult(i, u)
            h = dy.cmult(o, dy.tanh(c))
            return h, c
        else:
            # transduce
            Ui, Uo, Uu, Uf = self.US
            bi, bo, bu, bf = self.BS
            Wi, Wo, Wu, Wf = self.WS

            if self.att_type == 'selfatt':
                hm = dy.concatenate_cols(hs)
                hm = self.self_attention.encode(hm)
                hm = self.attention.encode(hm, x)
            elif self.att_type == 'att':
                hm = dy.concatenate_cols(hs)
                hm = self.attention.encode(hm, x)
            else:
                hm = dy.esum(hs)

            i = dy.logistic(dy.affine_transform([bi, Ui, hm, Wi, x]))
            o = dy.logistic(dy.affine_transform([bo, Uo, hm, Wo, x]))
            u = dy.tanh(dy.affine_transform([bu, Uu, hm, Wu, x]))
            fs = [dy.logistic(dy.affine_transform([bf, Uf, h, Wf, x])) for h in hs]
            c_out = dy.cmult(i, u) + dy.esum([dy.cmult(f, c) for f, c in zip(fs, cs)])
            h_out = dy.cmult(o, dy.tanh(c_out))
            return h_out, c_out


class TreeLSTMLM(object):
    def __init__(self, in_features: int, out_features: int, emb_size: int=10):
        self.model = dy.Model()
        self.in_features = in_features
        self.out_features = out_features
        self.emb_size = emb_size

        self.emb = self.model.add_lookup_parameters((self.in_features, self.emb_size), init='uniform', scale=self.emb_size ** (-0.5))
        
        self.tree_lstm_up = TreeLSTM(self.model, self.emb_size, 'selfatt')
        self.tree_lstm_dn = TreeLSTM(self.model, self.emb_size, None)
        
        self.W = self.model.add_parameters((self.out_features, 2 * self.emb_size))
        self.b = self.model.add_parameters(self.out_features)
    
    def forward(self, tokens, parents, children, node_order, inds_for_loss):
        hs_up = [dy.vecInput(self.emb_size) for _ in range(len(tokens))]
        cs_up = [dy.vecInput(self.emb_size) for _ in range(len(tokens))]
        hs_dn = [dy.vecInput(self.emb_size) for _ in range(len(tokens))]
        cs_dn = [dy.vecInput(self.emb_size) for _ in range(len(tokens))]
        
        for node in node_order:
            h_ch = [hs_up[ch] for ch in children[node]]
            c_ch = [cs_up[ch] for ch in children[node]]
            h_, c_ = self.tree_lstm_up.state(self.emb[tokens[node]], h_ch, c_ch)
            hs_up[node] = h_
            cs_up[node] = c_
        
        for node in reversed(node_order):
            h_pa = [hs_dn[pa] for pa in parents[node]]
            c_pa = [cs_dn[pa] for pa in parents[node]]
            h_, c_ = self.tree_lstm_dn.state(self.emb[tokens[node]], h_pa, c_pa)
            hs_dn[node] = h_
            cs_dn[node] = c_
        
        hs_return = [dy.affine_transform([self.b, self.W, dy.concatenate([hs_up[i], hs_dn[i]])]) for i in inds_for_loss]
        #cs_return = [dy.concatenate([cs_up[i], cs_dn[i]]) for i in inds_for_loss]
        return hs_return
