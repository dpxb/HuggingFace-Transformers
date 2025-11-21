import os
import requests
import math
import tiktoken
import  torch
import torch.nn as nn
from torch.nn import functional as F



batch_size = 4
context_length = 16
d_model = 64
num_blocks = 8
num_head = 4
learning_rate = 3e-4
dropout = 0.1
max_iters = 500
eval_interval = 50
eval_iters = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'

TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

if not os.path.exists('transform/sales_testbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true'
    with open('transform/sales_testbook.txt', 'wb') as f:
        f.write(requests.get(url,verify=False).content)
with open('transform/sales_testbook.txt', 'r',encoding='utf-8') as f:
    text = f.read()

#Tokenize the text
encoding = tiktoken.get_encoding('cl100k_base')
tokenized_text = encoding.encode(text)
max_token_value = max(tokenized_text)+1#最大的数
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)
#区分训练和测试
train_size = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_size]
valid_data = tokenized_text[train_size:]

#
class FeedforwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model,d_model*4)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(d_model*4,d_model)
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        x = self.linear1(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.head_dim = d_model // num_head
        self.context_length = context_length
        self.dropout = dropout
        self.Wq = nn.Linear(d_model,self.head_dim)
        self.Wk = nn.Linear(d_model,self.head_dim)
        self.Wv = nn.Linear(d_model,self.head_dim)
        #self.register_buffer('mask',torch.tril(torch.ones(self.context_length,self.context_length)).bool())
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        B,T,C =Q.shape
        mask = torch.tril(torch.ones(T,T,device=x.device)).bool()

        attention = Q @ K.transpose(-2,-1)/math.sqrt(d_model//num_head)
        attention = attention.masked_fill(~mask,float('-inf'))
        attention = F.softmax(attention,dim=-1)
        attention = attention @ V
        return attention

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([ScaledDotProductAttention()for _ in range(num_head)])
        self.projection = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self,x):
        heads_output = [head(x) for head in self.heads]
        out = torch.cat(heads_output,dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention()
        self.feedforward_network = FeedforwardNet()

    def forward(self,x):
        x = x + self.multi_head_attention(self.layer_norm1(x))
        x = x + self.feedforward_network(self.layer_norm2(x))
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.context_length = context_length
        self.num_blocks = num_blocks
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.max_token_value = max_token_value
        self.token_embedding_lookup_table = nn.Embedding(self.max_token_value+1,self.d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock() for _ in range(self.num_blocks)])
        self.model_out_linear_layer  = nn.Linear(self.d_model,self.max_token_value)

    def forward(self,idx,targets=None): #这个可以当做测试也可以当作训练
        B,T = idx.shape
        position_encoding_lookup_table = torch.zeros(context_length,d_model,device=device)
        position = torch.arange(0,context_length,dtype=torch.float,device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2,device=device).float() * -(math.log(10000.0)/d_model))
        position_encoding_lookup_table[:,0::2] = torch.sin(position*div_term)
        position_encoding_lookup_table[:,1::2] = torch.cos(position*div_term)
        position_embedding = position_encoding_lookup_table[:T,:].to(device)
        x = self.token_embedding_lookup_table(idx)+position_embedding
        for block in self.transformer_blocks:
            x = block(x)
        logits = self.model_out_linear_layer(x)


        if targets is not None:
            B,T,C = logits.shape
            logits_reshaped = logits.view(B*T,C)#降维
            targets_reshaped = targets.view(B*T)
            loss = F.cross_entropy(logits_reshaped,targets_reshaped)
        else:
            loss = None
        return logits,loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # 裁剪输入到模型的最大上下文长度
            idx_crop = idx[:, -self.context_length:]
            # 前向传播获取预测
            logits, loss = self(idx_crop)
            # 只取最后一个时间步的logits
            logits_last_timestep = logits[:, -1, :]
            # 通过softmax转换为概率
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # 从概率分布中采样
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # 将新生成的token添加到序列中
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = Model().to(device)

#得到数据段x 和 与它相差1位的数据段y
def get_batch(split:str):
    data = train_data if split =='train' else valid_data
    idxs = torch.randint(low=0,high=len(data) - context_length,size=(batch_size,) )
    x = torch.stack([data[idx:idx+context_length]for idx in idxs]).to(device)
    y = torch.stack([data[idx+1:idx+context_length+1]for idx in idxs]).to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch,y_batch = get_batch(split)
            logits,loss = model(x_batch,y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out#返回包含两个数据集平均损失的字典

#创建optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
tracked_losses = list()
for step in range(max_iters):
    if step%eval_iters==0 or step == max_iters-1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))
    xb,yb = get_batch('train')
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(),'model.pt')#用于保存PyTorch模型的参数
model.eval()
start = 'The product is'
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids,dtype=torch.long,device=device)[None,...])#[None,...]相当于unsqueeze(0) 1*3
y = model.generate(x,max_new_tokens=100)
print('------------------')
print(encoding.decode(y[0].tolist()))
print('------------------')







