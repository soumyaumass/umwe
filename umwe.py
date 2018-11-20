import os
import io
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from dictionary import Dictionary
from discriminator import Discriminator

class UMWE(nn.Module):
    
    def __init__(self, dtype=torch.float32, device=torch.device('cpu')):
        super(UMWE, self).__init__()
        self.dtype = dtype
        self.device = device
        self.src_langs = {0:'es', 1:'fa'}
        self.tgt_lang = 'en'
        self.langs = ['en', 'es', 'fa']
        self.dim = 300
        
    def load_embeddings(self, lang, emb_path):
        word2id = {}
        vectors = np.zeros((1,300))
        dico = []
        embeddings = []
        
        with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            for i, line in enumerate(f):
                if i > 0:
                    word, vect = line.rstrip().split(' ', 1)
                    vect = np.fromstring(vect, sep=' ')
                    
                    if not np.any(vect):
                        vect[0] = 0.01
                    
                    if word in word2id:
                        print("word found twice")
                    else:
                        if vect.shape[0] != self.dim:
                            print("Dimension not 300")
                    
                    word2id[word] = len(word2id)
                    vectors = np.vstack((vectors, vect))
                if i == 3000:
                    break
                   
        id2word = {v:k for k,v in word2id.items()}
        dico = Dictionary(id2word, word2id, lang)
        embeddings = vectors[1:,]
        embeddings = torch.tensor(embeddings, dtype=self.dtype)
        
        print("Embeddings Loaded")
        return dico, embeddings
                    
    
    def build_model(self):
        
        _src_embs = {}
        src_embs = []
        embs = {}
        vocabs = {}
        EMB_DIR = './'
        for lang in self.langs:
            src_embs.append(os.path.join(EMB_DIR, f'wiki.{lang}.vec'))
            
        for i in range(len(self.src_langs)):
            dico, emb = self.load_embeddings(self.src_langs[i], src_embs[i])
            vocabs[self.src_langs[i]] = dico
            _src_embs[self.src_langs[i]] = emb
            
        for lang in self.src_langs.values():
            src_emb = nn.Embedding(len(vocabs[lang]), self.dim, sparse=True)
            src_emb.weight.data.copy_(_src_embs[lang])
            embs[lang] = src_emb.to(self.device)
            
        dico, emb = self.load_embeddings(self.tgt_lang, src_embs[0])
        vocabs[self.tgt_lang] = dico
        tgt_emb = nn.Embedding(len(vocabs[self.tgt_lang]), self.dim, sparse = True)
        tgt_emb.weight.data.copy_(emb)
        embs[self.tgt_lang] = tgt_emb.to(self.device)
        
        
        encdec = {lang : nn.Linear(self.dim, self.dim).to(self.device) for lang in self.langs}
        
        for p in encdec[self.tgt_lang].parameters():
            p.requires_grad = False
        
        for ed in encdec.values():
            ed.weight.data.copy_(torch.eye(self.dim))
        
        disc = {lang: Discriminator(lang).to(self.device) for lang in self.langs}
        
        print('done')        
        return embs, encdec, disc
    
    def fit(self, embs, encdec, discriminators):
        batch = 128
        freq = 1024
        criterion = nn.BCELoss()
        optimizer = {lang: optim.Adam(discriminators[lang].parameters(), lr=0.001) for lang in self.langs}
        prev_loss = 1.
        for epoch in range(5000):
            for disc in discriminators.values():
                disc.train()
        
            loss = 0.
            if prev_loss - loss < 1e-3:
                print(epoch)
                break
            for dec_lang in self.langs:
                
                enc_lang = np.random.choice(self.langs)
                src_id = torch.LongTensor(batch).random_(freq).to(self.device)
                tgt_id = torch.LongTensor(batch).random_(freq).to(self.device)
                
                with torch.set_grad_enabled(False):
                    src_emb = embs[enc_lang](src_id).detach()
                    tgt_emb = embs[dec_lang](tgt_id).detach()
                    src_emb = encdec[enc_lang](src_emb)
                    src_emb = F.linear(src_emb, encdec[dec_lang].weight.t())
                
                x_to_disc = torch.cat([src_emb, tgt_emb], 0)
                y_true = torch.FloatTensor(2 * batch).zero_()
                
                y_true[:batch] = 1
                y_true = y_true.to(self.device)
                
                preds = discriminators[dec_lang](x_to_disc).flatten()
                loss += criterion(preds, y_true)
                
                optimizer[dec_lang].zero_grad()
                loss.backward(retain_graph=True)
                optimizer[dec_lang].step()
            print(loss)
            prev_loss = loss
            
def main():
    
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print('using device:', device)
    dtype = torch.float32
    
    model = UMWE(dtype, device)
    embs, ed, d = model.build_model()
    model.fit(embs, ed, d)


if __name__ == '__main__':
    main()

            
        