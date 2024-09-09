import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from dataclasses import dataclass
from typing import Optional, Tuple
from torch import nn

# 모델 인자 설정
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # 토크나이저에 의해 나중에 정의됨
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 32768
    depth_init: bool = True

# 분산 학습 초기화 함수
def setup_distributed_training():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())

# 주 주기 처리 함수
def train(model, dataloader, optimizer):
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

# Transformer 모델 정의 (기존 코드 그대로 사용)
class Transformer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.model_dim = model_args.dim
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.layers = nn.ModuleList([TransformerBlock(layer_id, model_args) for layer_id in range(model_args.n_layers)])
        self.norm = RMSNorm(dim=model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.init_weights()

    # 체크포인트 저장을 위한 초기화
    def init_weights(self):
        nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers:
            layer.init_weights()
        self.norm.reset_parameters()

    def forward(self, tokens: torch.Tensor):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h, self.freqs_cis[0:seqlen])
        h = self.norm(h)
        output = self.output(h).float()
        return output

# 데이터로더 설정
def get_dataloader(dataset, batch_size):
    sampler = DistributedSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)

def main():
    setup_distributed_training()

    # 모델 인자 설정 및 모델 생성
    model_args = ModelArgs(vocab_size=50000, dim=2048, n_heads=16, n_layers=12)
    model = Transformer(model_args).cuda()

    # DDP로 모델 감싸기
    model = DDP(model, device_ids=[int(os.environ['RANK']) % torch.cuda.device_count()])

    # 옵티마이저 정의
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 데이터로더 가져오기 (예시로 가상 데이터셋 사용)
    dataset = ...  # 데이터셋 정의 부분
    dataloader = get_dataloader(dataset, batch_size=32)

    # 학습 주기 실행
    train(model, dataloader, optimizer)

    # 마스터 노드에서만 체크포인트 저장
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), 'model_checkpoint.pth')

if __name__ == "__main__":
    main()
