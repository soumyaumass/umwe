import os
import io
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from dictionary import Dictionary
from discriminator import Discriminator
from evaluation import Evaluator

class UMWE(nn.Module):
    def __init__(self, dtype=torch.float32, device=torch.device('cpu'), batch=128):
        super(UMWE, self).__init__()
        self.dtype = dtype
        self.device = device
        self.src_langs = {0:'es', 1:'fa'}
        self.tgt_lang = 'en'
        self.langs = ['en', 'es', 'fa']
        self.dim = 300
        self.batch = batch
        self.embs = None
        self.vocabs = None
        self.encdec  = None
        self.discriminators = None
        
    def load_embeddings(self, lang, emb_path):
        if emb_path.endswith('.pth'):
            data = torch.load(emb_path)
            dico = data['dico']
            embeddings = data['vectors']
            return dico, embeddings
        else:
            word2id = {}
            dico = []
            embeddings = []
            vectors = []
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
                        vectors.append(vect[None])
                        if i == 200000:
                            break    
            
            id2word = {v:k for k,v in word2id.items()}
            dico = Dictionary(id2word, word2id, lang)
            embeddings = np.concatenate(vectors, 0)
            embeddings = torch.tensor(embeddings, dtype=self.dtype)
            
            print(f"Text Embeddings Loaded for language = {lang}")
            return dico, embeddings
    
    def export_embeddings(self, lang, export_embs, filetype):
        if filetype == "pth":
            torch.save({'dico': self.vocabs[lang], 'vectors': export_embs[lang]}, 'wiki.{}.pth'.format(lang))
    
    def build_model(self):
        _src_embs = {}
        src_embs = []
        embs = {}
        vocabs = {}
        EMB_DIR = './'
        pth_or_vec = "pth"
        if os.path.exists("wiki.en.pth") == False:
            pth_or_vec = "vec"
            print("Pretrained Pytorch embedding files not found")
            
        for lang in self.langs:
            src_embs.append(os.path.join(EMB_DIR, f'wiki.{lang}.{pth_or_vec}'))
            
        for i in range(len(self.src_langs)):
            dico, emb = self.load_embeddings(self.src_langs[i], src_embs[i+1])
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
        
        self.embs = embs
        self.encdec = encdec
        self.discriminators = disc
        self.vocabs = vocabs
                 
        for lang in self.langs:
            export_embs = _src_embs.copy()
            export_embs[self.tgt_lang] = emb
            self.export_embeddings(lang, export_embs, "pth")
        
    def discrim_step(self, freq):
        
        for disc in self.discriminators.values():
            disc.train()
            
        discrim_loss = 0.
        criterion = nn.BCELoss()
        discrim_optimizer = {lang: optim.SGD(self.discriminators[lang].parameters(), lr=0.01) for lang in self.langs}
        
        for dec_lang in self.langs:
            enc_lang = np.random.choice(self.langs)
            src_id = torch.LongTensor(self.batch).random_(freq).to(self.device)
            tgt_id = torch.LongTensor(self.batch).random_(freq).to(self.device)
            
            with torch.set_grad_enabled(False):
                src_emb = self.embs[enc_lang](src_id).detach()
                tgt_emb = self.embs[dec_lang](tgt_id).detach()
                src_emb = self.encdec[enc_lang](src_emb)
                src_emb = F.linear(src_emb, self.encdec[dec_lang].weight.t())
            
            x_to_disc = torch.cat([src_emb, tgt_emb], 0)
            y_true = torch.FloatTensor(2 * self.batch).zero_()
            
            y_true[:self.batch] = 1
            y_true = y_true.to(self.device)
            
            preds = self.discriminators[dec_lang](x_to_disc).flatten()
            # Calculate discriminator loss - assign 0 to fake and 1 to real embeddings
            discrim_loss += criterion(preds, y_true)
            
        discrim_optimizer[dec_lang].zero_grad()
        discrim_loss.backward(retain_graph=True)
        discrim_optimizer[dec_lang].step()
        
        return discrim_loss.data.item()
         
    def mapping_step(self, freq):
        
        for disc in self.discriminators.values():
            disc.eval()
            
        mapping_loss = 0
        criterion = nn.BCELoss()
        mapping_optimizer = {lang: optim.SGD(self.discriminators[lang].parameters(), lr=0.01) for lang in self.langs}
        
        # Loop over all languages
        for dec_lang in self.langs:
            # Select a random input language (dec_lang and enc_lang can be same - allowed - Adversarial Autoencoder)
            enc_lang = np.random.choice(self.langs)
            # Select a batch of random word IDs both for input and target langs
            src_id = torch.LongTensor(self.batch).random_(freq).to(self.device)
            tgt_id = torch.LongTensor(self.batch).random_(freq).to(self.device)
            
            with torch.set_grad_enabled(False):
                # Get corresponding random word embeddings
                src_emb = self.embs[enc_lang](src_id).detach()
                tgt_emb = self.embs[dec_lang](tgt_id).detach()
                # Encode to target space
                src_emb = self.encdec[enc_lang](src_emb)
                # Decode to target language space
                src_emb = F.linear(src_emb, self.encdec[dec_lang].weight.t())
            
            # Concatenate real and mapped embeddings
            x_to_disc = torch.cat([src_emb, tgt_emb], 0)
            # Create label space for binary classification of embeddings as real (1) or mapped (0)
            y_true = torch.FloatTensor(2 * self.batch).zero_()
            
            # Classify real embeddings as 1, keep others 0
            y_true[:self.batch] = 1
            y_true = y_true.to(self.device)
            preds = self.discriminators[dec_lang](x_to_disc).flatten()
            mapping_loss += criterion(preds, 1 - y_true)
            
            
        mapping_optimizer[dec_lang].zero_grad()
        mapping_loss.backward(retain_graph=True)
        mapping_optimizer[dec_lang].step()
        
        beta = 0.001
        for mapping in self.encdec.values():
            W = mapping.weight.detach()
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))
    
    def discrim_fit(self):
        freq = 10000
        
        for epoch in range(2):    
            discrim_loss_list = []
            start = time.time()
            for n_iter in range(0,200000, self.batch):
                
                for n in range(5):
                    discrim_loss_list.append(self.discrim_step(freq))
                discrim_loss = np.array(discrim_loss_list)
                
                if n_iter % 500 == 0:  
                    print(f'n_iter = {n_iter}',end=' ')
                    print("Discriminator Loss = ", end=' ')
                    print(f'{np.mean(discrim_loss):.4f}', end=' ')
                    end = time.time()
                    print(f'Time = {(end-start):.2f}')
                    start = end
                    discrim_loss_list = []
                
                self.mapping_step(freq)
            
            
def main():
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print('using device:', device)
    dtype = torch.float32
    
    model = UMWE(dtype, device, 128)
    model.build_model()
    model.discrim_fit()
    eval_ = Evaluator(model)
    print(eval_.clws('es', 'en'))

if __name__ == '__main__':
    main()