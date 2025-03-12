import torch
from torch import nn
from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from torch import optim

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class Log_softmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        x_max = torch.max(x)
        return x - x_max - torch.log(torch.sum(torch.exp(x - x_max), dim=self.dim)).unsqueeze(-1)

class PreNorm(nn.Module):
    def __init__(self, dim, fn, eps):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = Log_softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., eps=1e-5):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout), eps),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout), eps)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class transformer_model(nn.Module):
    def __init__(self, *, input_dim, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0.05, emb_dropout = 0.05, eps=1e-5):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.embed = nn.Linear(input_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, eps)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim, eps),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embed(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)
    
    
def train_epoch(model, optimizer, data_loader, loss_history, accuracy_history):
    total_samples = len(data_loader.dataset)
    model.train()

    correct_samples = 0
    total_loss = 0

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Accuracy computation
        _, pred = torch.max(output, dim=1)
        correct_samples += pred.eq(target).sum().item()
        total_loss += loss.item()

        # if i % 100 == 0:
            # print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
            #     ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
            #     '{:6.4f}'.format(loss.item()))
            

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct_samples / total_samples
    accuracy_history.append(accuracy)
    loss_history.append(avg_loss)
    print(f'Avg Train Loss: {avg_loss:.4f}  Train Accuracy: {accuracy:.2f}%')
    
    return  loss_history, accuracy_history

def evaluate(model, data_loader, loss_history, accuracy_history):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct_samples / total_samples
    loss_history.append(avg_loss)
    accuracy_history.append(accuracy)

    print('Avg Test loss: {:.4f}   Test Accuracy: {:.2f}%\n'.format(avg_loss, accuracy))
    return avg_loss, accuracy, loss_history, accuracy_history