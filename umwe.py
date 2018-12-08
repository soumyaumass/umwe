import os
import pickle
import io
import time
import itertools
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
from lexica import build_lexicon

class UMWE(nn.Module):
    def __init__(self, dtype=torch.float32, device=torch.device('cpu'), batch=128, epoch=2):
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
        self.max_rank = 15000
        self.freq = 75000
        self.lexica_method = 'nn'
        self.lexica = {}
        self.discrim_optimizer = None
        self.mapping_optimizer = None
        self.mpsr_optimizer = None
        self.epoch = epoch
        
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
    
    def export_embeddings(self, lang, export_embs, filetype, after=None):
        if filetype == "pth":
            torch.save({'dico': self.vocabs[lang], 'vectors': export_embs[lang]}, 'wiki.{}.pth'.format(lang))
        elif filetype == "txt":
            emb_lang = self.encdec[lang](self.embs[lang].weight)
            with io.open(f'vectors-{lang}-{after}.txt', 'w', encoding='utf-8') as f:
                f.write(u"%i %i\n" % emb_lang.shape)
                for i in range(len(self.vocabs[lang])):
                    if not i % 5000:
                        print(i)
                    if i == 50000:
                        break
                    f.write(u"%s %s\n" % (self.vocabs[lang][i], " ".join('%.5f' % x for x in emb_lang[i])))
    
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
        
        if pth_or_vec != 'pth':
            for lang in self.langs:
                export_embs = _src_embs.copy()
                export_embs[self.tgt_lang] = emb
                self.export_embeddings(lang, export_embs, "pth")
 
        self.discrim_optimizer = optim.SGD(itertools.chain(*[d.parameters() for d in self.discriminators.values()]), lr=0.1)
        self.mapping_optimizer = optim.SGD(itertools.chain(*[ed.parameters() for lang,ed in self.encdec.items() if lang!=self.tgt_lang]), lr=0.1)
        self.mpsr_optimizer = {lang: optim.Adam(self.encdec[lang].parameters()) for lang in self.langs}
        
    def discrim_step(self):
        
        for disc in self.discriminators.values():
            disc.train()
            
        discrim_loss = 0.
        criterion = nn.BCELoss()
        
        for dec_lang in self.langs:
            enc_lang = np.random.choice(self.langs)
            src_id = torch.LongTensor(self.batch).random_(self.freq).to(self.device)
            tgt_id = torch.LongTensor(self.batch).random_(self.freq).to(self.device)
            
            with torch.set_grad_enabled(False):
                src_emb = self.embs[enc_lang](src_id).detach()
                tgt_emb = self.embs[dec_lang](tgt_id).detach()
                src_emb = self.encdec[enc_lang](src_emb)
                src_emb = F.linear(src_emb, self.encdec[dec_lang].weight.t())
            
            x_to_disc = torch.cat([src_emb, tgt_emb], 0)
            y_true = torch.FloatTensor(2 * self.batch).zero_()
            
            y_true[:self.batch] = 1 - 0.1
            y_true[self.batch:] = 0.1
            y_true = y_true.to(self.device)
            
            preds = self.discriminators[dec_lang](x_to_disc).flatten()
            # Calculate discriminator loss - assign 0 to fake and 1 to real embeddings
            discrim_loss += criterion(preds, y_true)
            
        self.discrim_optimizer.zero_grad()
        discrim_loss.backward(retain_graph=True)
        self.discrim_optimizer.step()
        
        return discrim_loss.data.item()
         
    def mapping_step(self):
        
        for disc in self.discriminators.values():
            disc.eval()
            
        mapping_loss = 0
        criterion = nn.BCELoss()
        
        # Loop over all languages
        for dec_lang in self.langs:
            # Select a random input language (dec_lang and enc_lang can be same - allowed - Adversarial Autoencoder)
            enc_lang = np.random.choice(self.langs)
            # Select a batch of random word IDs both for input and target langs
            src_id = torch.LongTensor(self.batch).random_(self.freq).to(self.device)
            tgt_id = torch.LongTensor(self.batch).random_(self.freq).to(self.device)
            
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
            y_true[:self.batch] = 1 - 0.1
            y_true[self.batch:] = 0.1
            y_true = y_true.to(self.device)
            preds = self.discriminators[enc_lang](x_to_disc).flatten()
            mapping_loss += criterion(preds, 1 - y_true)
            
            
        self.mapping_optimizer.zero_grad()
        mapping_loss.backward(retain_graph=True)
        self.mapping_optimizer.step()
        
        beta = 0.001
        for mapping in self.encdec.values():
            W = mapping.weight.detach()
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))
        
        return mapping_loss.data.item()
    
    def discrim_fit(self):
        
        for epoch in range(5):    
            discrim_loss_list = []
            start = time.time()
            for n_iter in range(0, 400000, self.batch):
                
                for n in range(5):
                    discrim_loss_list.append(self.discrim_step())
                discrim_loss = np.array(discrim_loss_list)
                # discrim_loss = self.discrim_step()
                map_loss = self.mapping_step()
                
                if n_iter % 500 == 0:  
                    print(f'n_iter = {n_iter}',end=' ')
                    print("Discriminator Loss = ", end=' ')
                    print(f'{np.mean(discrim_loss):.4f}', end=' ')
                    # print(f'{discrim_loss:.4f}', end=' ')
                    print("Mappings Loss = ", end=' ')
                    print(f'{map_loss:.4f}', end=' ')
                    end = time.time()
                    print(f'Time = {(end-start):.2f}')
                    start = end
                    discrim_loss_list = []
                
    
    def mpsr_dictionary(self):
        for i, lang1 in enumerate(self.langs):
            for j, lang2 in enumerate(self.langs):
                # If the pair occurs for the first time, create the lexicon
                if i < j:
                    # Get embeddings for lang1
                    src_emb = self.embs[lang1].weight
                    # Apply lang1 encoder to embeddings
                    src_emb = self.encdec[lang1](src_emb).detach()
                    # Get embeddings for lang2
                    tgt_emb = self.embs[lang2].weight
                    # Apply lang2 encoder to embeddings
                    tgt_emb = self.encdec[lang2](tgt_emb).detach()
                    # Normalize output embeddings
                    src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
                    tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
                    # Build lexicon Lex(lang1, lang2)
                    self.lexica[(lang1, lang2)] = build_lexicon(self, src_emb, tgt_emb)
                # If the pair has already occurred before, swap the entries in the lexicon
                elif i > j:
                    self.lexica[(lang1, lang2)] = self.lexica[(lang2, lang1)][:, [1,0]]
                # If lang1 == lang2, each word is the nearest neighbor of itself
                else:
                    idx = torch.arange(self.max_rank).long().view(self.max_rank, 1)
                    self.lexica[(lang1, lang2)] = torch.cat([idx, idx], dim=1).to(self.device)
    
    def mpsr_step(self):
        mpsr_loss = 0
        # Loop on all languages
        for lang1 in self.langs:
            # Randomly select a language
            lang2 = np.random.choice(self.langs)

            ### Sample word IDs from both lang1 and lang2
            # Get the lexicon corresponding to both langs
            lexicon = self.lexica[(lang1, lang2)]
            # Generate random indices for sampling pairs from lexicon
            idx = torch.LongTensor(self.batch).random_(len(lexicon)).to(self.device)
            # Sample the lexicon with the generated indices
            sample_ids = lexicon.index_select(0, idx)
            # Get lang1 IDs
            lang1_ids = sample_ids[:, 0].to(self.device)
            # Get lang2 IDs
            lang2_ids = sample_ids[:, 1].to(self.device)
            ###

            ### Get corresponding word embeddings
            with torch.set_grad_enabled(True):
                lang1_emb = self.embs[lang1](lang1_ids).detach()
                lang2_emb = self.embs[lang2](lang2_ids).detach()
                # Encode to target space
                lang1_emb = self.encdec[lang1](lang1_emb)
                # Decode to target language space
                lang1_emb = F.linear(lang1_emb, self.encdec[lang2].weight.t())
            ###

            mpsr_loss += F.mse_loss(lang1_emb, lang2_emb)
        
        self.mpsr_optimizer[lang2].zero_grad()
        mpsr_loss.backward(retain_graph=True)
        self.mpsr_optimizer[lang2].step()

        beta = 0.001
        for mapping in self.encdec.values():
            W = mapping.weight.detach()
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))
        
        return mpsr_loss.data.item()
    
    def mpsr_refine(self):
        for epoch in range(5):
            # Create lexica from embeddings aligned using MAT in the previous step
            self.mpsr_dictionary()

            # Optimize MPSR
            start = time.time()
            mpsr_loss_list = []
            for n_iter in range(10000):
                # MPSR train step
                mpsr_loss_list.append(self.mpsr_step())
                # Log loss and other stats
                if n_iter % 500 == 0:
                    print(f'n_iter = {n_iter}',end=' ')
                    print("MPSR Loss = ", end=' ')
                    print(f'{np.mean(mpsr_loss_list):.4f}', end=' ')
                    end = time.time()
                    print(f'Time = {(end-start):.2f}')
                    start = end
                    mpsr_loss_list = []

            
def main():
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print('using device:', device)
    dtype = torch.float32
    
# =============================================================================
#     filename = 'curr_model'
#     f = open(filename, 'rb')
#     model = pickle.load(f)
#     f.close()
#     
# =============================================================================
    model = UMWE(dtype, device, 128, 2)
    model.build_model()
    model.discrim_fit()
    filename = 'curr_model'
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
# =============================================================================
#     model.mpsr_refine()
# =============================================================================
# =============================================================================
#     for lang in model.src_langs.values():
#         model.export_embeddings(lang, model.embs, "txt")
# =============================================================================
    eval_ = Evaluator(model)
    print(eval_.clws('es', 'en'))
    eval_.word_translation('es', 'en')

if __name__ == '__main__':
    main()