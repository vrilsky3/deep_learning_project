import pickle
import torch

with open('p_theta_logits.pickle', 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    p_theta_logits = pickle.load(f)
with open('p0_logits.pickle', 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    p0_logits = pickle.load(f)

K = 10


_, top_k_tokens_by_p0 = p0_logits.topk(K, dim=1)

def get_probs_of_top_k_tokens_by_p0_and_sum_rest(p_logits):
    # we are doing softmax first, because if we try to sum all the logits, we  overflow
    p_prob = torch.softmax(p_logits, dim=1)
    top_k_p_prob = p_prob.gather(dim=1, index=top_k_tokens_by_p0)
    sum_rest_of_p_prob = p_prob.sum(dim=1) - top_k_p_prob.sum(dim=1)
    p_theta_probs = torch.cat((top_k_p_prob, sum_rest_of_p_prob.view(sum_rest_of_p_prob.shape[0], 1)), dim=1)
    return p_theta_probs

p_0_probs = get_probs_of_top_k_tokens_by_p0_and_sum_rest(p0_logits)
p_theta_probs = get_probs_of_top_k_tokens_by_p0_and_sum_rest(p_theta_logits)

kl_div_loss_calculator = torch.nn.KLDivLoss(reduction="batchmean")

# KLDivLoss expects the (first) input to be log'd
p_theta_probs_logged = torch.log(p_theta_probs)
loss = kl_div_loss_calculator(p_theta_probs_logged, p_0_probs)

print()