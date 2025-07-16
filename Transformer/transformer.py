import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
            d_model >> 모델 차원->임베딩 벡터 크기
            max_len >> 최대 처리 길이
        """
        super(PositionalEncoding, self).__init__()

        # 위치 인코딩 행렬 초기화
        pe = torch.zeros(max_len, d_model)
        
        # 각 위치(pos)와 차원(i)에 대한 값을 계산
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 짝수 차원에는 sin 함수 적용
        pe[:, 0::2] = torch.sin(position * div_term)
        # 홀수 차원에는 cos 함수 적용
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # (max_len, d_model) -> (1, max_len, d_model) 형태로 변경하여 배치 처리에 용이하게 함
        pe = pe.unsqueeze(0)
        
        # pe를 모델의 파라미터가 아닌 버퍼로 등록 (학습되지 않음)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 임베딩 (batch_size, seq_len, d_model)
        
        Returns:
            torch.Tensor: 위치 정보가 더해진 임베딩
        """
        # 입력 임베딩에 위치 인코딩 값을 더함
        x = x + self.pe[:, :x.size(1), :]
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        """
        Args:
            d_model (int): 모델의 차원
            h (int): 헤드의 개수
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        
        self.d_k = d_model // h # 각 헤드의 차원
        self.h = h # 헤드의 개수
        
        # Q, K, V 및 최종 출력을 위한 선형 계층
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value (torch.Tensor): (batch_size, seq_len, d_model)
            mask (torch.Tensor): 어텐션을 가리기 위한 마스크 (optional)

        Returns:
            torch.Tensor: 어텐션 결과 (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)

        # 1. Q, K, V를 선형 변환 후 헤드 수(h)만큼 차원을 나눔
        # (batch_size, seq_len, d_model) -> (batch_size, h, seq_len, d_k)
        query, key, value = \
            [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2. 스케일드 닷-프로덕트 어텐션 계산
        # scores: (batch_size, h, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 마스크가 있으면, 마스크 위치에 아주 작은 값(-1e9)을 채워 softmax 후 0이 되게 함
        if mask is not None:
            mask = mask.unsqueeze(1) # (batch_size, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        # 소프트맥스를 통해 어텐션 가중치(p_attn) 계산
        p_attn = self.softmax(scores)
        
        # 가중치와 Value를 곱하여 최종 어텐션 값 계산
        # (batch_size, h, seq_len, d_k)
        x = torch.matmul(p_attn, value)

        # 3. 헤드들을 다시 합치고 선형 변환
        # (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        
        return self.linears[-1](x)