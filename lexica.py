# Original work Copyright (c) 2017-present, Facebook, Inc.
# Modified work Copyright (c) 2018, Xilun Chen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import torch

# from .utils import get_nn_avg_dist


logger = getLogger()


def get_candidates(params, emb1, emb2):
    """
    Get best translation pairs candidates.
    """
    bs = 128

    all_scores = []
    all_targets = []

    # nearest neighbors
    if params.lexica_method == 'nn':

        # for every source word
        for i in range(0, params.max_rank, bs):

            # compute target words scores
            # 200000x300 X 300x128 -> 200000x128 (This gives the proximity of each of the 128 embeddings to each of the 200000 embeddings)
            # More proximity, higher score
            scores = emb2.mm(emb1[i:min(params.max_rank, i + bs)].transpose(0, 1)).transpose(0, 1)
            # Pick the top 2 largest scores and their indices
            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)
        print(all_scores.shape)
        print(all_targets.shape)

    # contextual dissimilarity measure
    elif params.lexica_method.startswith('csls_knn_'):

        knn = params.lexica_method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)

        # average distances to k nearest neighbors
        average_dist1 = torch.from_numpy(get_nn_avg_dist(emb2, emb1, knn))
        average_dist2 = torch.from_numpy(get_nn_avg_dist(emb1, emb2, knn))
        average_dist1 = average_dist1.type_as(emb1)
        average_dist2 = average_dist2.type_as(emb2)

        # for every source word
        for i in range(0, params.max_rank, bs):

            # compute target words scores
            scores = emb2.mm(emb1[i:min(params.max_rank, i + bs)].transpose(0, 1)).transpose(0, 1)
            scores.mul_(2)
            scores.sub_(average_dist1[i:min(params.max_rank, i + bs)][:, None] + average_dist2[None, :])
            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    all_pairs = torch.cat([
        torch.arange(0, all_targets.size(0)).long().unsqueeze(1),
        all_targets[:, 0].unsqueeze(1)
    ], 1)
    print(all_pairs.shape)

    # sanity check
    assert all_scores.size() == all_pairs.size() == (params.max_rank, 2)

    # sort pairs by score confidence
    diff = all_scores[:, 0] - all_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    all_scores = all_scores[reordered]
    all_pairs = all_pairs[reordered]

    # max dico words rank
    if params.max_rank > 0:
        selected = all_pairs.max(1)[0] <= params.max_rank
        mask = selected.unsqueeze(1).expand_as(all_scores).clone()
        all_scores = all_scores.masked_select(mask).view(-1, 2)
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)
        if len(all_pairs) == 0:
            return []

    return all_pairs


def build_lexicon(params, src_emb, tgt_emb):
    """
    Build a training dictionary given current embeddings / mapping.
    """
    logger.info("Building the train dictionary ...")

    s2t_candidates = get_candidates(params, src_emb, tgt_emb)
    t2s_candidates = get_candidates(params, tgt_emb, src_emb)
    t2s_candidates = torch.cat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)

    s2t_candidates = set([(a, b) for a, b in s2t_candidates.numpy()])
    t2s_candidates = set([(a, b) for a, b in t2s_candidates.numpy()])

    final_pairs = s2t_candidates & t2s_candidates
    if len(final_pairs) == 0:
        logger.warning("Empty intersection ...")
        return None
    dico = torch.LongTensor(list([[int(a), int(b)] for (a, b) in final_pairs]))

    if len(dico) == 0:
        return None
    logger.info('New train dictionary of %i pairs.' % dico.size(0))
    return dico.to(params.device)