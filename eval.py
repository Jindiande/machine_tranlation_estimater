import numpy as np
import pandas as pd
from scipy import stats
from utils import *
from train import *
filename_hter=""
def eval():
    print("cuda: %s" % CUDA)

    data, src_vocab, tgt_vocab = load_data_whole()
    enc = encoder(len(src_vocab))
    dec = decoder(len(tgt_vocab))
    qua = quality(len(tgt_vocab))
    with torch.no_grad():
        enc.eval()
        dec.eval()
        qua.eval()
    epoch = load_checkpoint(sys.argv[1], enc, dec,qua) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print(enc)
    print(dec)
    print(qua)
    print("evaluating model...")

    ii = 0
    QE_score_set=[]
    Hter_set=[]
    for x, y, Hter_batch in data:
        try:
            ii += 1
            # loss1 = 0
            # loss2 = 0
            loss3 = 0

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
            dec_in = torch.cat((dec_in, y), dim=1)  # [B S+1]
            qua_in, contex = dec(dec_in[:, :y.size(1)], enc_out, 1, mask)  # [B S H]
            """
            for t in range(y.size(1)):
                dec_out, contex_part = dec(dec_in, enc_out, t, mask)
                qua_in[:, t, :] = dec_out
                contex[:, t, :] = contex_part
                loss1 += F.nll_loss(dec_out, y[:, t], ignore_index=PAD_IDX, reduction="sum")
                dec_in = y[:, t].unsqueeze(1)  # teacher forcing
            """
            b, QE_score = qua(qua_in, y, contex, dec.embed)
            del qua_in, contex, dec_in, enc_out
            gc.collect()
            QE_score_set.append(QE_score.cpu().detach().numpy())

        except RuntimeError as e:

            if 'out of memory' in str(e):
                print('|WARNING : ran out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise e
    print("over, now calculating result")
    QE_score_set=np.reshape(QE_score_set,[np.array(QE_score_set).shape[0]*np.array(QE_score_set).shape[1]])
    Hter_set=np.reshape(Hter_set,[np.array(Hter_set).shape[0]*np.array(Hter_set).shape[1]])
    print("Pearson correlation=", np.corrcoef(QE_score_set,Hter_set))
    print("Spearman correlation=", stats.corrcoef(QE_score_set, Hter_set))
    print("MAE=",np.average(np.absolute(QE_score_set-Hter_set)))
    if __name__ == "__main__":
        if(len(sys.argv)!=5):
            sys.exit("Usage: %s model vocab.src vocab.tgt training_data num_epoch" % sys.argv[0])
        with torch.no_grad():
            eval()

