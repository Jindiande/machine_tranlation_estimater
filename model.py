import torch
import gc
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



UNIT = "word" # unit of tokenization (char, word)

MIN_LEN = 1 # minimum sequence length for training

MAX_LEN = 50 # maximum sequence length for training and decoding

BATCH_SIZE = 64 * 3 # must be divisible by BEAM_SIZE at inference time

EMBED_SIZE = 300

HIDDEN_SIZE = 1000

HIDDEN_SIZE_2=100

NUM_LAYERS = 2

DIMEN_MAXOUT_UNIT=300   # dimension of maxout unit

DIMEN_QUALITY_VECTOR=300  # dimension of quality vector

DROPOUT = 0.5

BIDIRECTIONAL = True

NUM_DIRS = 2 if BIDIRECTIONAL else 1

LEARNING_RATE = 1e-4

BEAM_SIZE = 3

VERBOSE = 0 # 0: None, 1: attention heatmap, 2: beam search

SAVE_EVERY = 10



PAD = "<PAD>" # padding

EOS = "<EOS>" # end of sequence

SOS = "<SOS>" # start of sequence

UNK = "<UNK>" # unknown token



PAD_IDX = 0

SOS_IDX = 1

EOS_IDX = 2

UNK_IDX = 3



torch.manual_seed(1)

CUDA = torch.cuda.is_available()

assert BATCH_SIZE % BEAM_SIZE == 0



class encoder(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()



        # architecture

        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)

        self.rnn = nn.LSTM( # LSTM or GRU

            input_size = EMBED_SIZE,

            hidden_size = HIDDEN_SIZE // NUM_DIRS,

            num_layers = NUM_LAYERS,

            bias = True,

            batch_first = True,

            dropout = DROPOUT,

            bidirectional = BIDIRECTIONAL

        )



        if CUDA:

            self = self.cuda()



    def init_hidden(self, rnn_type): # initialize hidden states

        h = zeros(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS) # hidden states

        if rnn_type == "LSTM":

            c = zeros(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS) # cell states

            return (h, c)

        return h



    def forward(self, x, mask):

        self.hidden = self.init_hidden("GRU") # LSTM or GRU

        x = self.embed(x)

        x = nn.utils.rnn.pack_padded_sequence(x, mask[1], batch_first = True)

        h, _ = self.rnn(x, self.hidden)

        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)

        return h



class decoder(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.feed_input = True # input feeding



        # architecture

        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)

        self.rnn = nn.LSTM( # LSTM or GRU

            input_size = EMBED_SIZE + (HIDDEN_SIZE if self.feed_input else 0),

            hidden_size = HIDDEN_SIZE // NUM_DIRS,

            num_layers = NUM_LAYERS,

            bias = True,

            batch_first = True,

            dropout = DROPOUT,

            bidirectional = BIDIRECTIONAL

        )

        self.attn = attn()

        self.out = nn.Linear(HIDDEN_SIZE, vocab_size)

        self.softmax = nn.LogSoftmax(1)



        if CUDA:

            self = self.cuda()



    def forward(self, dec_in, enc_out, t, mask):

        x = self.embed(dec_in)

        if self.feed_input:

            x = torch.cat((x, self.attn.hidden), 2)

        h, _ = self.rnn(x, self.hidden)

        if self.attn:

            qua_in,contex = self.attn(h, enc_out, t, mask)

        #h = self.out(h).squeeze(1)
        #y = self.softmax(h)

        return qua_in,contex

class quality(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        #self.S=nn.Linear(2*NUM_DIRS*HIDDEN_SIZE,2*DIMEN_MAXOUT_UNIT)
        self.S = nn.Linear(2 *  HIDDEN_SIZE, 2 * DIMEN_MAXOUT_UNIT)
        self.V=nn.Linear(2*EMBED_SIZE,2*DIMEN_MAXOUT_UNIT)
        self.C0=nn.Linear(NUM_DIRS*HIDDEN_SIZE,2*DIMEN_MAXOUT_UNIT)
        #self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx=PAD_IDX)
        #self.t_long=None
        #self.t=None
        #self.quality=None
        #self.attn=attn()
        self.W1=torch.rand(vocab_size,DIMEN_QUALITY_VECTOR)
        self.W2=nn.Linear(DIMEN_MAXOUT_UNIT,DIMEN_QUALITY_VECTOR)
        self.softmax=nn.LogSoftmax(1)

        self.rnn=nn.LSTM(
            input_size=DIMEN_QUALITY_VECTOR,

            hidden_size=HIDDEN_SIZE_2 ,

            num_layers=NUM_LAYERS,

            bias=True,

            batch_first=True,

            dropout=DROPOUT,

            bidirectional=False

        )
        self.Wqe=nn.Linear(HIDDEN_SIZE_2,1)
    def init_hidden(self, rnn_type): # initialize hidden states

        h = zeros(NUM_LAYERS , BATCH_SIZE, HIDDEN_SIZE_2 ) # hidden states

        if rnn_type == "LSTM":

            c = zeros(NUM_LAYERS , BATCH_SIZE, HIDDEN_SIZE_2) # cell states

            return (h, c)

        return h
    def fun1(self,dec_out,y,embed_dic):# dec_out= b*seq_length*(hidden*bi), y=b*seq_length, contex= b*seq_length*(hidden*bi)
                                       # dec_out hidden state: <SOS> w1 w2... w_n
        tensor1=torch.zeros(dec_out.size(0),1,2*dec_out.size(2))
        tensor2=torch.zeros(dec_out.size(0),1,2*EMBED_SIZE)
        embed=embed_dic(y)             # [b seq_length EMBED_SIZE] W1 W2 ... WN <EOS>
        for i in range(dec_out.size(1)):# for all seq_length
            if((i==0)):
                tensor1=torch.cat((dec_out[:,i+1,int(dec_out.size(2)/2):dec_out.size(2)].unsqueeze(1),dec_out[:,i+1,int(dec_out.size(2)/2):dec_out.size(2)].unsqueeze(1)),2)
                tensor2=torch.cat((embed[:,i+1,:].unsqueeze(1),embed[:,i+1,:].unsqueeze(1)),2)
            elif(i==dec_out.size(1)-1):
                tensor1 = torch.cat((tensor1, torch.cat((dec_out[:, i - 1, 0:int(dec_out.size(2)/2)].unsqueeze(1), dec_out[:, i - 1,0:int(dec_out.size(2)/2)].unsqueeze(1)), 2)), 1)
                tensor2=torch.cat((tensor1, torch.cat((embed[:, i - 1, :].unsqueeze(1), embed[:, i - 1, :].unsqueeze(1)), 2)), 1)
            else:
                tensor1 = torch.cat((tensor1, torch.cat((dec_out[:, i - 1, 0:int(dec_out.size(2)/2)].unsqueeze(1), dec_out[:, i + 1, int(dec_out.size(2)/2):dec_out.size(2)].unsqueeze(1)), 2)), 1)
                tensor2 = torch.cat((tensor1, torch.cat((embed[:, i - 1, :].unsqueeze(1), embed[:, i + 1, :].unsqueeze(1)), 2)), 1)

        return tensor1,tensor2
    def fun2(self,dec_out,y,embed_dic):# dec_out hidden state: <SOS> w1 w2... w_n
        tensor2 = torch.zeros(dec_out.size(0), 1, 2 * EMBED_SIZE)
        embed=embed_dic(y)
        for i in range(dec_out.size(1)):  # for all seq_length
            if ((i == 0)):
                tensor2 = torch.cat((embed[:, i + 1, :].unsqueeze(1), embed[:, i + 1, :].unsqueeze(1)), 2)
            elif (i == dec_out.size(1) - 1):
                tensor2 = torch.cat(
                    (tensor2, torch.cat((embed[:, i - 1, :].unsqueeze(1), embed[:, i - 1, :].unsqueeze(1)), 2)), 1)
            else:
                tensor2 = torch.cat(
                    (tensor2, torch.cat((embed[:, i - 1, :].unsqueeze(1), embed[:, i + 1, :].unsqueeze(1)), 2)), 1)
        return dec_out.cuda(),tensor2.cuda()
    def forward(self, dec_out,y,contex,embed_dic):

        tensor1,tensor2=self.fun2(dec_out,y,embed_dic)
        t_long=self.S(tensor1)+self.V(tensor2)+self.C0(contex.cuda())   #b*s*(2DIMEN_MAXOUT_UNIT)
        del tensor1,tensor2
        gc.collect()
        t=torch.ones(t_long.size(0),t_long.size(1),DIMEN_MAXOUT_UNIT)  # b*s*DIMEN_MAXOUT_UNIT
        for i in range(DIMEN_MAXOUT_UNIT):
            t[:,:,i]=torch.where(t_long[:,:,2*i]>t_long[:,:,2*i+1],t_long[:,:,2*i],t_long[:,:,2*i+1])
        a=self.W2(t.cuda())   #b*s*DIMEN_QUALITY_VECTOR
        b=a.matmul(self.W1.permute(1,0).cuda())        #b*s*vocab_size
        del t_long,t
        gc.collect()
        b=b.view(b.size(0)*b.size(1),b.size(2))# (b*s)*vocab_size
        b=self.softmax(b.cuda())              # for first training part loss calculation

        row_W1=torch.zeros(a.size(0),a.size(1),DIMEN_QUALITY_VECTOR)# b*s*DIMEN_QUALITY_VECTOR
        for i in range(a.size(0)):
            for j in range(a.size(1)):
                row_W1[i,j,:]=self.W1[y[i,j],:]
        quality=row_W1.mul(a)                                  # b*s*DIMEN_QUALITY_VECTOR
        del a,row_W1
        gc.collect()
        mask1=y==0
        index1=np.where(mask1.cpu().detach().numpy().any(axis=1),mask1.cpu().detach().numpy().argmax(axis=1),mask1.size(1)-1)# [b 1] first index=pad_index
        index1=torch.from_numpy(index1)
        self.hidden=self.init_hidden("LSTM")
        h,_=self.rnn(quality,self.hidden)                      # b*s*hidden_size_2
        s=h.cpu().detach().numpy()[range(0,h.size(0)),index1.long(),:]  # b*hidden_size_2
        s=torch.from_numpy(s)
        s=s.squeeze(1)                                              # b*hidden_size_2
        QE_score=self.Wqe(s.cuda())                                        # b*1, qe score in one batch
        QE_score=torch.sigmoid(QE_score).cuda()
        del quality,h,s,index1
        gc.collect()
        return b,QE_score



        #using t to forward cross_entropy








class attn(nn.Module): # attention layer (Luong et al 2015)

    def __init__(self):

        super().__init__()

        self.type = "global" # global, local-m, local-p

        self.method = "dot" # dot, general, concat

        self.hidden = None # attentional hidden state for input feeding

        self.Va = None # attention weights



        # architecture

        if self.type[:5] == "local":

            self.window_size = 5

            if self.type[-1] == "p":

                self.Wp = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

                self.Vp = nn.Linear(HIDDEN_SIZE, 1)

        if self.method == "general":

            self.Wa = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

        elif self.method  == "concat":

            pass # TODO

        self.softmax = nn.Softmax(2)

        self.Wc = nn.Linear(HIDDEN_SIZE * 2 *NUM_DIRS, HIDDEN_SIZE*NUM_DIRS)

        self.Wd=nn.Linear(HIDDEN_SIZE*NUM_DIRS,HIDDEN_SIZE)



    def window(self, ht, hs, t, mask): # for local attention

        if self.type[-1] == "m": # monotonic

            p0 = max(0, min(t - self.window_size, hs.size(1) - self.window_size))

            p1 = min(hs.size(1), t + 1 + self.window_size)

            return hs[:, p0:p1], mask[0][:, p0:p1]

        if self.type[-1] == "p": # predicative

            S = Tensor(mask[1]) # source sequence length

            pt = S * torch.sigmoid(self.Vp(torch.tanh(self.Wp(ht)))).view(-1) # aligned position

            hs_w = []

            mask_w = []

            k = [] # truncated Gaussian distribution as kernel function

            for i in range(BATCH_SIZE):

                p = int(S[i].item())

                seq_len = mask[1][i]

                min_len = mask[1][-1]

                p0 = max(0, min(p - self.window_size, seq_len - self.window_size))

                p1 = min(seq_len, p + 1 + self.window_size)

                if min_len < p1 - p0:

                    p0 = 0

                    p1 = min_len

                hs_w.append(hs[i, p0:p1])

                mask_w.append(mask[0][i, p0:p1])

                sd = (p1 - p0) / 2 # standard deviation

                v = [torch.exp(-(j - pt[i]).pow(2) / (2 * sd ** 2)) for j in range(p0, p1)]

                k.append(torch.cat(v))

            hs_w = torch.cat(hs_w).view(BATCH_SIZE, -1, HIDDEN_SIZE)

            mask_w = torch.cat(mask_w).view(BATCH_SIZE, -1)

            k = torch.cat(k).view(BATCH_SIZE, 1, -1)

            return hs_w, mask_w, pt, k



    def align(self, ht, hs, mask, k):

        if self.method == "dot":

            a = ht.bmm(hs.transpose(1, 2))

        elif self.method == "general":

            a = ht.bmm(self.Wa(hs).transpose(1, 2))

        elif self.method == "concat":

            pass # TODO

        a = a.masked_fill(mask.unsqueeze(1), -10000) # masking in log space

        a = self.softmax(a) # [B, 1, H] @ [B, H, L] = [B, 1, L]

        if self.type == "local-p":

            a = a * k

        return a # alignment weights



    def forward(self, ht, hs, t, mask):

        if self.type == "local-p":

            hs, mask, pt, k = self.window(ht, hs, t, mask)

        else:

            if self.type == "local-m":

                hs, mask = self.window(ht, hs, t, mask)

            else:

                mask = mask[0]

            k = None

        a = self.Va = self.align(ht, hs, mask, k) # alignment vector

        c = a.bmm(hs) # context vector [B, S, L] @ [B, L, 2H] = [B, S, H]

        h = torch.cat((c, ht), 2)# [B S 4H]

        qua_in=self.Wc(h)#[B S 2H]

        qua_in=torch.tanh(qua_in)

        self.hidden = torch.tanh(self.Wc(h)) # attentional vector [B S H]

        del a,h
        gc.collect()

        return qua_in,c



def Tensor(*args):

    x = torch.Tensor(*args)

    return x.cuda() if CUDA else x



def LongTensor(*args):

    x = torch.LongTensor(*args)

    return x.cuda() if CUDA else x



def zeros(*args):

    x = torch.zeros(*args)

    return x.cuda() if CUDA else x



def maskset(x):

    mask = x.data.eq(PAD_IDX)

    return (mask, x.size(1) - mask.sum(1)) # set of mask and lengths