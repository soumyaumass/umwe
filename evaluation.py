import torch
import os
import io
import gc
# import path
import numpy as np
from scipy.stats import spearmanr

class Evaluator():
    
    def __init__(self, model):
        self.embs = model.embs
        self.vocab = model.vocabs
        self.encdec = model.encdec
        self.discriminators = model.discriminators
    
    def encode_decode(self, encdec, embs):
        batch = 1024
        with torch.no_grad():
            op = embs.clone()
            for i, k in enumerate(range(0, len(embs), batch)):
                x = embs[k:k+batch]
                op[k:k+batch] = encdec(x.to(encdec.weight.device)).to(embs.device)
            return op
    
    def get_pairs(self, file):
        word_pairs = []
        
        with io.open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = ((line.rstrip()).lower()).split()
                if len(line) != 3:
                    continue
                word_pairs.append((line[0], line[1], float(line[2])))
            return word_pairs
    
    def get_id(self, word, word2id):
        wid = word2id.get(word)
        if wid is None:
            wid = word2id.get(word.capitalize())
            if wid is None:
                wid = word2id.get(word.title())
        return wid
            
    
    def spearman_correlation(self, word2id1, emb1, path, word2id2, emb2):
        word_pairs = self.get_pairs(path)
        not_found = 0
        preds = []
        true = []
        for w2, w1, sim in word_pairs:
            wid1 = self.get_id(w1, word2id1)
            wid2 = self.get_id(w2, word2id2)
            if wid1 is None or wid2 is None:
                not_found += 1
                true.append(sim)
                preds.append(0.5)
                #ignore word not in both
                continue
            
            e1 = emb1[wid1]
            e2 = emb2[wid2]
            s = e1.dot(e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-5)
            true.append(sim)
            preds.append(s)
            
        return spearmanr(true, preds)[0], len(true), not_found
    
    def get_cross_lingual_score(self, src_lang, src_word2id, src_emb, tgt_lang, tgt_word2id, tgt_emb):
        path = './crosslingual/wordsim'
        src_tgt = os.path.join(path, f'{src_lang}-{tgt_lang}-SEMEVAL17.txt')
        tgt_src = os.path.join(path, f'{tgt_lang}-{src_lang}-SEMEVAL17.txt')
        rho = 0.
        
        if os.path.exists(src_tgt):
            rho, found, not_found = self.spearman_correlation(src_word2id, src_emb, src_tgt, tgt_word2id, tgt_emb)
        elif os.path.exists(tgt_src):
            rho, found, not_found = self.spearman_correlation(tgt_word2id, tgt_emb, tgt_src, src_word2id, src_emb)
        
        return rho

    def load_dict(self, path, w2id1, w2id2):
        
        pairs = []
        with io.open(path, 'r', encoding='utf-8') as f:
            for _, line in enumerate(f):
                line = line.lower()
                w1, w2 = line.rstrip().split()
                if w1 in w2id1 and w2 in w2id2:
                    pairs.append((w1,w2))
                
        pairs = sorted(pairs, key = lambda x: w2id1[x[0]])
        
        my_dict = torch.LongTensor(len(pairs), 2)
        for i, (w1, w2) in enumerate(pairs):
            my_dict[i, 0] = w2id1[w1]
            my_dict[i, 1] = w2id2[w2]
                
        return my_dict

    def translation_accuracy(self, src_lang, word2id1, src_emb, tgt_lang, word2id2, tgt_emb):
        path = os.path.join('./', f'{tgt_lang}-{src_lang}.5000-6500.txt')
        my_dict = self.load_dict(path, word2id1, word2id2)
        
# =============================================================================
#         for i, (k,v) in enumerate(my_dict):
#             if i==15:
#                 break
#             print(k,v)
# =============================================================================
        
        src_emb = src_emb.to(torch.device('cpu'))
        tgt_emb = tgt_emb.to(torch.device('cpu'))
        src_emb /= src_emb.norm(2, 1, keepdim = True).expand_as(src_emb)
        tgt_emb /= tgt_emb.norm(2, 1, keepdim = True).expand_as(tgt_emb)
        
        q = src_emb[my_dict[:,0]].to(torch.device('cpu'))
        del src_emb
        gc.collect()
        score = q.mm(tgt_emb.transpose(0,1))
        
        result = []
        top_match = score.topk(10, 1, True)[1]
        
        for k in [5]:
            top_match = top_match[:, :k]
            
            match = (top_match == my_dict[:, 1][:, None].expand_as(top_match)).sum(1)
            m = {}
            for i, src in enumerate(my_dict[:, 0].numpy()):
                m[src] = min(m.get(src, 0) + match[i], 1)
                precision = 100 * np.mean(list(m.values()))
                result.append(('k = %i' % k, precision))
        return result
        
    def clws(self, src_lang, tgt_lang):
        src_emb = self.encode_decode(self.encdec[src_lang], self.embs[src_lang].weight).cpu().numpy()
        tgt_emb = self.encode_decode(self.encdec[tgt_lang], self.embs[tgt_lang].weight).cpu().numpy()
      
        scores = self.get_cross_lingual_score(src_lang, self.vocab[src_lang].word2id, src_emb, tgt_lang, self.vocab[tgt_lang].word2id, tgt_emb)
        
        return f'{src_lang}_{tgt_lang}_CLWS_SCORE = {scores}'
    
    def word_translation(self, src_lang, tgt_lang):
        src_emb = self.encode_decode(self.encdec[src_lang], self.embs[src_lang].weight)
        tgt_emb = self.encode_decode(self.encdec[tgt_lang], self.embs[tgt_lang].weight)
        scores = self.translation_accuracy(src_lang, self.vocab[src_lang].word2id, src_emb, tgt_lang, self.vocab[tgt_lang].word2id, tgt_emb)
        print(scores)
        
    
    
          
