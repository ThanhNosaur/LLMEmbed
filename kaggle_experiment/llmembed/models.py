"""Table 1 fusion operators and the draft's learned chunk/source fusion."""
import torch
from torch import nn


class BaselineFusion(nn.Module):
    def __init__(self, classes, dims=(4096, 1024, 1024), strategy=15, hidden=1024, sigma=0.1, author=False):
        super().__init__()
        if strategy not in range(1, 16):
            raise ValueError('Table 1 strategy must be 1..15.')
        self.strategy, self.sigma, self.author = strategy, sigma, author
        dl, db, dr = dims
        sizes = {1: dl, 2: dl, 3: dl, 4: 5 * dl,
                 5: dl + db, 6: dl + db, 7: 5 * dl + db,
                 8: dl + dr, 9: dl + dr, 10: 5 * dl + dr,
                 11: dl + db + dr, 12: dl + db + dr, 13: 5 * dl + db + dr,
                 14: 49, 15: 49 + dl}
        if strategy >= 14:
            self.compress_layers = nn.ModuleList([
                nn.Sequential(nn.Linear(dl, db), nn.ReLU(), nn.Dropout(0.5)) for _ in range(5)])
            self.roberta_align = nn.Identity() if db == dr else nn.Linear(dr, db)
        self.fc1 = nn.Linear(sizes[strategy], hidden)
        self.fc2 = nn.Linear(hidden, max(1, hidden // 4))
        self.fc3 = nn.Linear(max(1, hidden // 4), classes)
        self.dropout = nn.Dropout(0.5)

    def fuse(self, llama, bert, roberta):
        s = self.strategy
        if s >= 14:
            sources = torch.stack([self.compress_layers[i](llama[:, i]) for i in range(5)] +
                                  [bert, self.roberta_align(roberta)], dim=1)
            gram = (sources @ sources.transpose(1, 2)).flatten(1)
            # Match released code; the paper's tanh(2*sigma*x) equality is a typo.
            co = 2 * torch.sigmoid(self.sigma * gram) - 1
            return torch.cat([co, llama.mean(1)], 1) if s == 15 else co
        if s == 1:
            return llama[:, 0]  # Features are ordered deepest to shallowest.
        l = llama.flatten(1) if s in (4, 7, 10, 13) else llama.amax(1) if s in (3, 6, 9, 12) else llama.mean(1)
        parts = [l]
        if s in (5, 6, 7, 11, 12, 13):
            parts.append(bert)
        if s in (8, 9, 10, 11, 12, 13):
            parts.append(roberta)
        return torch.cat(parts, 1)

    def forward(self, llama, bert, roberta):
        x = self.dropout(torch.relu(self.fc1(self.fuse(llama, bert, roberta))))
        x = self.dropout(torch.relu(self.fc2(x)))
        logits = self.fc3(x)
        return (logits.softmax(-1) if self.author else logits), None


class AdaptiveFusion(nn.Module):
    def __init__(self, classes, dims=(4096, 1024, 1024), hidden=1024, heads=8, sigma=0.2):
        super().__init__()
        if hidden % heads or hidden < 4:
            raise ValueError('Hidden size must be >=4 and divisible by the number of heads.')
        self.sigma = sigma
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.LayerNorm(hidden))
            for d in [dims[0]] * 5 + list(dims[1:])])
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=0.2, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.gate = nn.Sequential(nn.Linear(hidden, hidden // 4), nn.Tanh(), nn.Linear(hidden // 4, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * hidden, hidden // 2), nn.GELU(),
                                        nn.LayerNorm(hidden // 2), nn.Dropout(0.2), nn.Linear(hidden // 2, classes))

    def forward(self, llama, bert, roberta):
        inputs = [llama[:, i] for i in range(5)] + [bert, roberta]
        x = torch.stack([projection(value) for projection, value in zip(self.projections, inputs)], 1)
        attention, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm(x + attention)
        weights = self.gate(x).softmax(1)
        pooled = (weights * x).sum(1)
        interaction = torch.tanh(2 * self.sigma * x[:, :5].mean(1) * x[:, 5:].mean(1))
        return self.classifier(torch.cat([pooled, interaction], -1)), weights.squeeze(-1)


class ChunkPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.features = nn.Linear(dim, 64)
        self.position = nn.Linear(3, 64, bias=False)
        self.score = nn.Linear(64, 1, bias=False)

    def forward(self, x, mask):
        # [batch, chunks, depths, dimension]; tokenizers may yield different chunk counts.
        if not mask.any(1).all():
            raise ValueError('Every document must contain at least one chunk.')
        indices = torch.arange(x.shape[1], device=x.device)[None, :]
        last = mask.sum(1, keepdim=True) - 1
        positions = torch.stack([(indices / last.clamp_min(1)).expand_as(mask),
                                 (indices == 0).expand_as(mask), indices == last], -1).float()
        scores = self.score(torch.tanh(self.features(x) + self.position(positions).unsqueeze(2))).squeeze(-1)
        weights = scores.masked_fill(~mask.unsqueeze(-1), -torch.inf).softmax(1)
        return (weights.unsqueeze(-1) * x).sum(1), weights


class ExperimentModel(nn.Module):
    def __init__(self, classes, dims, chunked=False, adaptive=False, strategy=15, hidden=1024, heads=8, sigma=0.1, author=False):
        super().__init__()
        self.chunked = chunked
        self.pools = nn.ModuleList([ChunkPool(dim) for dim in dims]) if chunked else nn.ModuleList()
        self.fusion = (AdaptiveFusion(classes, dims, hidden, heads, sigma) if adaptive else
                       BaselineFusion(classes, dims, strategy, hidden, sigma, author))

    def forward(self, features, masks):
        if self.chunked:
            pooled = [pool(x, mask) for pool, x, mask in zip(self.pools, features, masks)]
            sources, chunks = zip(*pooled)
        else:
            sources, chunks = [x[:, 0] for x in features], None
        logits, gate = self.fusion(sources[0], sources[1][:, 0], sources[2][:, 0])
        return logits, gate, chunks
