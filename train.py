import sys
from utils import *
from os.path import isfile
filename_hter=""
train_model="first"
def load_data_whole():
    data = []
    src_batch = []
    tgt_batch = []
    Hter_batch=[]
    src_batch_len = 0
    tgt_batch_len = 0
    print("loading data...")
    src_vocab = load_vocab(sys.argv[2], "src")
    tgt_vocab = load_vocab(sys.argv[3], "tgt")
    fo = open(sys.argv[4], "r")
    f1=open(filename_hter,"r")
    for line in fo:
        line = line.strip()
        line1= line1.strip()
        src, tgt = line.split("\t")
        src = [int(i) for i in src.split(" ")] + [EOS_IDX]
        tgt = [int(i) for i in tgt.split(" ")] + [EOS_IDX]
        # src.reverse() # reversing source sequence
        if len(src) > src_batch_len:
            src_batch_len = len(src)
        if len(tgt) > tgt_batch_len:
            tgt_batch_len = len(tgt)
        src_batch.append(src)
        tgt_batch.append(tgt)
        try:
            Hter_batch.append([float(line1)])
        except ValueError as e:
            print("error")
        if len(src_batch) == BATCH_SIZE:
            for seq in src_batch:
                seq.extend([PAD_IDX] * (src_batch_len - len(seq)))
            for seq in tgt_batch:
                seq.extend([PAD_IDX] * (tgt_batch_len - len(seq)))
            data.append((LongTensor(src_batch), LongTensor(tgt_batch),Hter_batch))
            src_batch = []
            tgt_batch = []
            Hter_batch=[]
            src_batch_len = 0
            tgt_batch_len = 0
        line1=f1.readline()
    fo.close()
    f1.close()
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, src_vocab, tgt_vocab


def load_data_first():
    data = []
    src_batch = []
    tgt_batch = []
    Hter_batch=[]
    src_batch_len = 0
    tgt_batch_len = 0
    print("loading data...")
    src_vocab = load_vocab(sys.argv[2], "src")
    tgt_vocab = load_vocab(sys.argv[3], "tgt")
    fo = open(sys.argv[4], "r")
    f1=open(filename_hter,"r")
    for line in fo:
        line = line.strip()
        line1= line1.strip()
        src, tgt = line.split("\t")
        src = [int(i) for i in src.split(" ")] + [EOS_IDX]
        tgt = [int(i) for i in tgt.split(" ")] + [EOS_IDX]
        # src.reverse() # reversing source sequence
        if len(src) > src_batch_len:
            src_batch_len = len(src)
        if len(tgt) > tgt_batch_len:
            tgt_batch_len = len(tgt)
        src_batch.append(src)
        tgt_batch.append(tgt)
        try:
            Hter_batch.append([float(line1)])
        except ValueError as e:
            print("error")
        if len(src_batch) == BATCH_SIZE:
            for seq in src_batch:
                seq.extend([PAD_IDX] * (src_batch_len - len(seq)))
            for seq in tgt_batch:
                seq.extend([PAD_IDX] * (tgt_batch_len - len(seq)))
            data.append((LongTensor(src_batch), LongTensor(tgt_batch)))
            src_batch = []
            tgt_batch = []
            Hter_batch=[]
            src_batch_len = 0
            tgt_batch_len = 0
        line1=f1.readline()
    fo.close()
    f1.close()
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, src_vocab, tgt_vocab


def train_whole():
    print("cuda: %s" % CUDA)
    num_epochs = int(sys.argv[5])
    data, src_vocab, tgt_vocab = load_data_whole()
    enc = encoder(len(src_vocab))
    dec = decoder(len(tgt_vocab))
    qua = quality(len(tgt_vocab))
    enc_optim = torch.optim.Adam(enc.parameters(), lr = LEARNING_RATE)
    dec_optim = torch.optim.Adam(dec.parameters(), lr = LEARNING_RATE)
    qua_optim=  torch.optim.Adam(qua.parameters(), lr = LEARNING_RATE)
    epoch = load_checkpoint(sys.argv[1], enc, dec,qua) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print(enc)
    print(dec)
    print(qua)
    print("training whole model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        ii = 0
        mem_out_count=0
        #loss1_sum = 0
        #loss2_sum=0
        loss3_sum=0
        #timer = time.time()
        for x, y,Hter_batch in data:
            try:
                ii += 1
                #loss1 = 0
                #loss2 = 0
                loss3 = 0
                enc.zero_grad()
                dec.zero_grad()
                qua.zero_grad()
                mask = maskset(x)
                enc_out = enc(x, mask)
                dec_in = LongTensor([SOS_IDX] * BATCH_SIZE).unsqueeze(1)
                dec.hidden = enc.hidden
                if dec.feed_input:
                    dec.attn.hidden = zeros(BATCH_SIZE, 1, HIDDEN_SIZE)
                """    
                qua_in = torch.zeros(BATCH_SIZE, y.size(1), HIDDEN_SIZE * NUM_DIRS)
                contex = torch.zeros(BATCH_SIZE, y.size(1), HIDDEN_SIZE * NUM_DIRS)
                """
                dec_in=torch.cat((dec_in,y),dim=1)#[B S+1]
                qua_in,contex=dec(dec_in[:,:y.size(1)],enc_out,1,mask)#[B S H]
                """
                for t in range(y.size(1)):
                    dec_out, contex_part = dec(dec_in, enc_out, t, mask)
                    qua_in[:, t, :] = dec_out
                    contex[:, t, :] = contex_part
                    loss1 += F.nll_loss(dec_out, y[:, t], ignore_index=PAD_IDX, reduction="sum")
                    dec_in = y[:, t].unsqueeze(1)  # teacher forcing
                """
                _, QE_score = qua(qua_in, y, contex, dec.embed)
                del qua_in,contex,dec_in,enc_out
                gc.collect()
                #loss2 = F.nll_loss(b, y.view(y.size(0) * y.size(1), y.size(2)), ignore_index=PAD_IDX, reduction="sum")
                # loss1 /= y.data.gt(0).sum().float() # divide by the number of unpadded tokens

                loss=nn.L1Loss()
                loss3=loss(QE_score.cuda(),torch.from_numpy(np.array(Hter_batch)))

                # loss1.backward()
                #loss2.backward()
                loss3.backward()
                enc_optim.step()
                dec_optim.step()
                qua_optim.step()
                # loss1 = loss1.item()
                #loss2 = loss2.item()
                loss3 = loss3.item()
                # loss1_sum += loss1
                #loss2_sum += loss2
                loss3_sum += loss3
                if(ii%100==0):
                    print("ii=",ii,"loss=",loss3_sum/(ii-mem_out_count))
                # print("epoch = %d, iteration = %d, loss = %f" % (ei, ii, loss))
            except RuntimeError as e:
                mem_out_count+=1
                if 'out of memory' in str(e):
                    print('|WARNING : ran out of memory')
                    if hasattr(torch.cuda,'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                        raise e

        #timer = time.time() - timer
        #loss1_sum /= len(data)
        loss3_sum /= len(data)
        if((ei-epoch)%1==0):
            save_checkpoint("ii=",str(ii)+",ei="+str(ei),enc,dec,qua,epoch,loss3_sum/(len(data)-mem_out_count))


def train_first():
    print("cuda: %s" % CUDA)
    num_epochs = int(sys.argv[5])
    data, src_vocab, tgt_vocab = load_data_first()
    enc = encoder(len(src_vocab))
    dec = decoder(len(tgt_vocab))
    qua = quality(len(tgt_vocab))
    enc_optim = torch.optim.Adam(enc.parameters(), lr = LEARNING_RATE)
    dec_optim = torch.optim.Adam(dec.parameters(), lr = LEARNING_RATE)
    qua_optim=  torch.optim.Adam(qua.parameters(), lr = LEARNING_RATE)
    epoch = load_checkpoint(sys.argv[1], enc, dec,qua) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print(enc)
    print(dec)
    print(qua)
    print("training whole model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        ii = 0
        mem_out_count=0
        loss2_sum=0
        #timer = time.time()
        for x, y in data:
            try:
                ii += 1

                loss2 = 0

                enc.zero_grad()
                dec.zero_grad()
                qua.zero_grad()
                mask = maskset(x)
                enc_out = enc(x, mask)
                dec_in = LongTensor([SOS_IDX] * BATCH_SIZE).unsqueeze(1)
                dec.hidden = enc.hidden
                if dec.feed_input:
                    dec.attn.hidden = zeros(BATCH_SIZE, 1, HIDDEN_SIZE)
                """    
                qua_in = torch.zeros(BATCH_SIZE, y.size(1), HIDDEN_SIZE * NUM_DIRS)
                contex = torch.zeros(BATCH_SIZE, y.size(1), HIDDEN_SIZE * NUM_DIRS)
                """
                dec_in=torch.cat((dec_in,y),dim=1)#[B S+1]
                qua_in,contex=dec(dec_in[:,:y.size(1)],enc_out,1,mask)#[B S H]
                """
                for t in range(y.size(1)):
                    dec_out, contex_part = dec(dec_in, enc_out, t, mask)
                    qua_in[:, t, :] = dec_out
                    contex[:, t, :] = contex_part
                    loss1 += F.nll_loss(dec_out, y[:, t], ignore_index=PAD_IDX, reduction="sum")
                    dec_in = y[:, t].unsqueeze(1)  # teacher forcing
                """
                b, _ = qua(qua_in, y, contex, dec.embed)
                del qua_in,contex,dec_in,enc_out
                gc.collect()
                loss2 = F.nll_loss(b, y.view(y.size(0) * y.size(1), y.size(2)), ignore_index=PAD_IDX, reduction="sum")


                loss2.backward()

                enc_optim.step()
                dec_optim.step()
                qua_optim.step()
                loss2 = loss2.item()
                # loss1_sum += loss1
                loss2_sum += loss2
                if(ii%100==0):
                    print("ii=",ii,"loss=",loss2_sum/(ii-mem_out_count))
                # print("epoch = %d, iteration = %d, loss = %f" % (ei, ii, loss))
            except RuntimeError as e:
                mem_out_count+=1
                if 'out of memory' in str(e):
                    print('|WARNING : ran out of memory')
                    if hasattr(torch.cuda,'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                        raise e

        #timer = time.time() - timer
        #loss1_sum /= len(data)
        loss2_sum /= len(data)
        if((ei-epoch)%1==0):
            save_checkpoint("ii=",str(ii)+",ei="+str(ei),enc,dec,qua,epoch,loss2_sum/(len(data)-mem_out_count))
if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model vocab.src vocab.tgt training_data num_epoch" % sys.argv[0])
    if(train_model=="first"):
        train_first()
    else:
        train_whole()