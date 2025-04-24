import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__()
        self.w_gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        logits = self.w_gate(x)
        scores = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        return topk_vals, topk_idx, scores

class MoELayer(nn.Module):
    def __init__(self, experts, gating_network):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gating = gating_network

    def forward(self, x):
        topk_vals, topk_idx, _ = self.gating(x)
        expert_outputs = []
        for k in range(topk_idx.size(-1)):
            expert = self.experts[topk_idx[..., k]]
            expert_outputs.append(expert(x) * topk_vals[..., k].unsqueeze(-1))
        return sum(expert_outputs)