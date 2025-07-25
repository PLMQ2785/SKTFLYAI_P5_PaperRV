import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from konlpy.tag import Mecab
from tqdm import tqdm
import math

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# --- 0. 설정 및 파일 경로 ---
# 파일들이 스크립트와 동일한 폴더에 미리 준비되어 있어야 합니다.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILES = {
    "train": {
        "ko": "korean-english-park.train.ko",
        "en": "korean-english-park.train.en",
    },
    "dev": {"ko": "korean-english-park.dev.ko", "en": "korean-english-park.dev.en"},
    "test": {"ko": "korean-english-park.test.ko", "en": "korean-english-park.test.en"},
}

MIN_FREQ = 2  # 어휘집 구축 시 최소 등장 빈도
PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

# --- 1. 모델 및 유틸리티 함수 정의 ---
# PositionalEncoding, Seq2SeqTransformer, generate_square_subsequent_mask, create_mask
# 이 함수들은 이전 답변의 코드와 동일합니다.


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        emb_size,
        nhead,
        src_vocab_size,
        tgt_vocab_size,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(
        self,
        src,
        trg,
        src_mask,
        tgt_mask,
        src_padding_mask,
        tgt_padding_mask,
        memory_key_padding_mask,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz, device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    src_seq_len, tgt_seq_len = src.shape[0], tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def build_vocab(sentences, tokenizer, min_freq):
    """주어진 문장들과 토크나이저를 사용해 어휘집을 구축합니다."""
    # 1. 모든 문장에 대해 토큰 빈도 계산
    counter = Counter(
        token
        for sentence in tqdm(sentences, desc="단어 빈도 계산 중")
        for token in tokenizer(sentence)
    )
    print(f"고유 토큰 수 (필터링 전): {len(counter)}")

    # 2. 특별 토큰을 먼저 추가
    vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}

    # 3. 최소 등장 빈도를 만족하는 단어만 필터링하여 추가
    #    별도의 카운터(idx)를 사용하여 4번부터 순차적으로 인덱스 부여
    idx = 4
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = idx
            idx += 1

    print(f"고유 토큰 수 (최소 빈도 {min_freq}회 이상): {len(vocab) - 4}")
    print(f"최종 단어장 크기 (특별 토큰 포함): {len(vocab)}")

    return vocab


# --- 2. 데이터 처리 클래스 및 함수 ---
class ParkCorpusDataset(Dataset):
    def __init__(
        self, ko_path, en_path, ko_vocab, en_vocab, ko_tokenizer, en_tokenizer
    ):
        with open(ko_path, "r", encoding="utf-8") as f:
            self.ko_sents = [line.strip() for line in f.readlines()]
        with open(en_path, "r", encoding="utf-8") as f:
            self.en_sents = [line.strip() for line in f.readlines()]

        self.ko_vocab = ko_vocab
        self.en_vocab = en_vocab
        self.ko_tokenizer = ko_tokenizer
        self.en_tokenizer = en_tokenizer

    def __len__(self):
        return len(self.ko_sents)

    def __getitem__(self, idx):
        ko_tokens = [
            self.ko_vocab.get(token, UNK_IDX)
            for token in self.ko_tokenizer(self.ko_sents[idx])
        ]
        en_tokens = [
            self.en_vocab.get(token, UNK_IDX)
            for token in self.en_tokenizer(self.en_sents[idx])
        ]
        return torch.tensor([SOS_IDX] + ko_tokens + [EOS_IDX]), torch.tensor(
            [SOS_IDX] + en_tokens + [EOS_IDX]
        )


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    return pad_sequence(src_batch, padding_value=PAD_IDX), pad_sequence(
        tgt_batch, padding_value=PAD_IDX
    )


# --- 3. 학습 및 평가 함수 ---
def train_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        # --- 디버깅 코드 추가 ---
        # 입력 텐서의 인덱스가 어휘집 크기 범위 내에 있는지 확인
        if src.max().item() >= model.src_tok_emb.num_embeddings:
            print(
                f"오류: Source 텐서의 최댓값({src.max().item()})이 어휘집 크기({model.src_tok_emb.num_embeddings})를 벗어났습니다."
            )
            # 문제가 되는 배치를 확인하기 위해 텐서 출력
            print("문제가 발생한 Source 텐서:", src)
            # 여기서 프로그램을 중단하거나 추가적인 디버깅을 할 수 있습니다.

        if tgt.max().item() >= model.tgt_tok_emb.num_embeddings:
            print(
                f"오류: Target 텐서의 최댓값({tgt.max().item()})이 어휘집 크기({model.tgt_tok_emb.num_embeddings})를 벗어났습니다."
            )
            print("문제가 발생한 Target 텐서:", tgt)
        # --- 디버깅 코드 끝 ---

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )
        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )
        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(list(dataloader))


def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating"):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input
            )
            logits = model(
                src,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask,
            )
            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(list(dataloader))


def en_tokenizer_func(text):
    return text.lower().split()


# --- 4. 메인 실행 로직 ---
if __name__ == "__main__":
    # 토크나이저 정의
    ko_tokenizer = Mecab(dicpath="/usr/local/lib/mecab/dic/mecab-ko-dic").morphs
    # en_tokenizer = lambda text: text.lower().split()
    en_tokenizer = en_tokenizer_func

    # **훈련 데이터(.train)로만 어휘집 구축**
    print("훈련 데이터로 어휘집을 구축합니다...")
    with open(FILES["train"]["ko"], "r", encoding="utf-8") as f:
        train_ko_sents = [line.strip() for line in f.readlines()]
    with open(FILES["train"]["en"], "r", encoding="utf-8") as f:
        train_en_sents = [line.strip() for line in f.readlines()]

    ko_vocab = build_vocab(train_ko_sents, ko_tokenizer, MIN_FREQ)
    en_vocab = build_vocab(train_en_sents, en_tokenizer, MIN_FREQ)

    SRC_VOCAB_SIZE = len(ko_vocab)
    TGT_VOCAB_SIZE = len(en_vocab)
    print(f"Source Vocab Size: {SRC_VOCAB_SIZE}, Target Vocab Size: {TGT_VOCAB_SIZE}")

    # 데이터셋 및 데이터로더 생성
    print("\n데이터셋을 생성합니다...")
    train_dataset = ParkCorpusDataset(
        FILES["train"]["ko"],
        FILES["train"]["en"],
        ko_vocab,
        en_vocab,
        ko_tokenizer,
        en_tokenizer,
    )
    dev_dataset = ParkCorpusDataset(
        FILES["dev"]["ko"],
        FILES["dev"]["en"],
        ko_vocab,
        en_vocab,
        ko_tokenizer,
        en_tokenizer,
    )
    test_dataset = ParkCorpusDataset(
        FILES["test"]["ko"],
        FILES["test"]["en"],
        ko_vocab,
        en_vocab,
        ko_tokenizer,
        en_tokenizer,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=256, collate_fn=collate_fn, shuffle=True, num_workers=8
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=256, collate_fn=collate_fn, num_workers=8
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=256, collate_fn=collate_fn, num_workers=8
    )

    # 모델, 손실함수, 옵티마이저 초기화
    model = Seq2SeqTransformer(3, 3, 256, 8, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, 512).to(
        DEVICE
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 학습 시작
    print("\n학습을 시작합니다...")
    for epoch in range(1, 251):
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_fn)
        val_loss = evaluate(model, dev_dataloader, loss_fn)
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 최종 평가
    print("\n최종 모델을 테스트 데이터로 평가합니다...")
    test_loss = evaluate(model, test_dataloader, loss_fn)
    print(f"Test Loss: {test_loss:.4f}")

    # 모델 가중치와 어휘집 저장
    print("\n모델과 어휘집을 저장합니다...")

    # 1. 모델의 학습된 가중치 저장
    torch.save(model.state_dict(), "transformer_model.pth")

    # 2. 소스 및 타겟 언어 어휘집 저장
    # (주의: dataset 객체에서 vocab을 가져와야 합니다)
    torch.save(train_dataset.ko_vocab, "ko_vocab.pth")
    torch.save(train_dataset.en_vocab, "en_vocab.pth")

    print("저장 완료: transformer_model.pth, ko_vocab.pth, en_vocab.pth")
