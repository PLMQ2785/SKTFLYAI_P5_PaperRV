import torch
import torch.nn as nn
import math
from konlpy.tag import Mecab

import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    font_path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc("font", family=font_name)
except FileNotFoundError:
    print("한글 폰트를 찾을 수 없습니다. 영문으로 표시될 수 있습니다.")
    pass

# --- 0. 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3


attention_weights = {}


def get_attention_hook(layer_name):
    """특정 레이어의 어텐션 가중치를 딕셔너리에 저장하는 Hook을 반환합니다."""

    def hook(model, input, output):
        # output[1]이 어텐션 가중치 텐서입니다. (shape: [batch_size, num_heads, seq_len_q, seq_len_k])
        # 여기서는 배치 크기와 헤드 수가 1개 또는 평균내어 처리한다고 가정합니다.
        attention_weights[layer_name] = output[1].squeeze().cpu().detach()

    return hook


# 기존의 translate_and_get_attention 함수를 아래 코드로 완전히 대체합니다.


def translate_and_get_attention(model, src_sentence, ko_vocab, en_vocab, ko_tokenizer):
    """
    번역을 수행하고, 각 디코딩 스텝에서 마지막 디코더 레이어의
    인코더-디코더 어텐션 가중치를 추출하여 반환합니다.
    (Residual Connection이 올바르게 수정된 버전)
    """
    model.eval()

    # --- 토큰화 및 텐서 변환 ---
    src_tokens_str = [token for token in ko_tokenizer(src_sentence)]
    src_tokens = [ko_vocab.get(token, UNK_IDX) for token in src_tokens_str]
    src_tensor = (
        torch.LongTensor([SOS_IDX] + src_tokens + [EOS_IDX]).unsqueeze(1).to(DEVICE)
    )
    src_mask = (
        (torch.zeros(src_tensor.shape[0], src_tensor.shape[0]))
        .type(torch.bool)
        .to(DEVICE)
    )

    # --- 인코딩 ---
    with torch.no_grad():
        memory = model.encode(src_tensor, src_mask)

    # --- Greedy Decode 수동 수행 ---
    ys = torch.ones(1, 1).fill_(SOS_IDX).type(torch.long).to(DEVICE)
    tgt_tokens_str = []
    all_attentions = []

    for i in range(src_tensor.shape[0] + 10):
        tgt_mask = generate_square_subsequent_mask(ys.size(0)).to(DEVICE)
        tgt_emb = model.positional_encoding(model.tgt_tok_emb(ys))

        # 디코더 레이어 수동 실행 (Residual Connection 추가)
        output = tgt_emb
        attention_this_step = None
        for layer in model.transformer.decoder.layers:
            # Self-Attention + Residual & Norm
            self_attn_output, _ = layer.self_attn(
                output, output, output, attn_mask=tgt_mask, need_weights=False
            )
            output = layer.norm1(output + layer.dropout1(self_attn_output))

            # Encoder-Decoder Attention + Residual & Norm
            cross_attn_output, attention_this_step = layer.multihead_attn(
                output, memory, memory, need_weights=True
            )
            output = layer.norm2(output + layer.dropout2(cross_attn_output))

            # Feed Forward + Residual & Norm
            ff_output = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(output)))
            )
            output = layer.norm3(output + layer.dropout3(ff_output))

        # 다음 단어 예측
        prob = model.generator(output[-1, :, :])
        _, next_word_idx = torch.max(prob, dim=1)
        next_word_idx = next_word_idx.item()

        if attention_this_step is not None:
            all_attentions.append(
                attention_this_step.squeeze(0).mean(dim=0).cpu().detach()
            )

        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word_idx)], dim=0
        )
        rev_en_vocab = {i: word for word, i in en_vocab.items()}
        tgt_tokens_str.append(rev_en_vocab.get(next_word_idx, "<unk>"))

        if next_word_idx == EOS_IDX:
            break

    final_attention = torch.stack(all_attentions)
    full_src_tokens = ["<sos>"] + src_tokens_str + ["<eos>"]
    full_tgt_tokens = ["<sos>"] + tgt_tokens_str

    return full_src_tokens, full_tgt_tokens, final_attention


def plot_attention_map(src_tokens, tgt_tokens, attention):
    """어텐션 가중치를 히트맵으로 시각화합니다."""
    fig, ax = plt.subplots(figsize=(10, 12))  # figsize도 약간 키우면 좋습니다.
    im = ax.imshow(attention.numpy(), cmap="Pastel1", aspect="auto")

    # 축 설정 (fontsize 추가)
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=90, fontsize=16)  # x축 눈금 글자 크기
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens, fontsize=16)  # y축 눈금 글자 크기

    # 레이블 및 타이틀 (fontsize 추가)
    plt.xlabel("Source (Korean)", fontsize=16)  # x축 레이블 크기
    plt.ylabel("Target (English)", fontsize=16)  # y축 레이블 크기
    plt.title("Encoder-Decoder Attention", fontsize=16)  # 제목 크기

    fig.colorbar(im)
    plt.tight_layout()
    plt.show()


# --- 1. 모델 정의 (학습 코드와 완전히 동일해야 함) ---
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

    def encode(self, src, src_mask):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )


# --- 2. 번역 함수 ---
def translate(
    model, src_sentence, ko_vocab, en_vocab, ko_tokenizer, en_tokenizer, device
):
    model.eval()  # 모델을 평가 모드로 설정

    # 입력 문장 토큰화 및 수치화
    src_tokens = [ko_vocab.get(token, UNK_IDX) for token in ko_tokenizer(src_sentence)]
    src_tensor = (
        torch.LongTensor([SOS_IDX] + src_tokens + [EOS_IDX]).unsqueeze(1).to(device)
    )

    # 소스 마스크 생성
    num_src_tokens = src_tensor.shape[0]
    src_mask = (torch.zeros(num_src_tokens, num_src_tokens)).type(torch.bool).to(device)

    # 인코더 실행 (메모리 생성)
    with torch.no_grad():
        memory = model.encode(src_tensor, src_mask)
    memory = memory.to(device)

    # 디코더 시작 토큰(<sos>) 준비
    ys = torch.ones(1, 1).fill_(SOS_IDX).type(torch.long).to(device)

    # 순차적으로 단어 생성 (Greedy Search)
    for i in range(100):  # 최대 100개 단어까지 생성
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(
            device
        )

        with torch.no_grad():
            out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        # 생성된 단어를 다음 입력으로 추가
        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word)], dim=0
        )

        # 종료 토큰(<eos>)이 생성되면 중단
        if next_word == EOS_IDX:
            break

    # 생성된 인덱스 시퀀스를 타겟 언어의 단어로 변환
    # <sos> 토큰은 제외
    tgt_tokens = [
        list(en_vocab.keys())[list(en_vocab.values()).index(i)]
        for i in ys.squeeze().tolist()
        if i in list(en_vocab.values())
    ][1:]

    return " ".join(tgt_tokens)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz, device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


# --- 3. 메인 실행 로직 ---
if __name__ == "__main__":
    # 저장된 어휘집 불러오기
    print("어휘집을 불러옵니다...")
    ko_vocab = torch.load("ko_vocab.pth")
    en_vocab = torch.load("en_vocab.pth")
    SRC_VOCAB_SIZE = len(ko_vocab)
    TGT_VOCAB_SIZE = len(en_vocab)

    # 토크나이저 정의
    ko_tokenizer = Mecab(dicpath="C:/mecab/mecab-ko-dic").morphs
    en_tokenizer = lambda text: text.lower().split()

    # 하이퍼파라미터 (모델 구조는 학습 때와 동일해야 함)
    EMB_SIZE = 256
    NHEAD = 8
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    # 모델 초기화 및 가중치 불러오기
    print("모델 가중치를 불러옵니다...")
    model = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS,
        EMB_SIZE,
        NHEAD,
        SRC_VOCAB_SIZE,
        TGT_VOCAB_SIZE,
        FFN_HID_DIM,
    )
    model.load_state_dict(torch.load("transformer_model.pth", map_location=DEVICE))
    model.to(DEVICE)

    # 번역할 문장
    kor_sentence = "석유 거래상들이 고조되고 있는 미국의 이라크 공격 가능성을 고려함에따라 전쟁 열기가 미국의 유가를 19개월 만에 최고치로 밀어 올렸다."

    # 번역 실행
    translated_sentence = translate(
        model, kor_sentence, ko_vocab, en_vocab, ko_tokenizer, en_tokenizer, DEVICE
    )

    print("\n=====================================")
    print(f"입력 (한국어): {kor_sentence}")
    print(f"번역 (영어): {translated_sentence}")
    print("=====================================")

    src_tokens, tgt_tokens, attention = translate_and_get_attention(
        model, kor_sentence, ko_vocab, en_vocab, ko_tokenizer
    )

    translated_sentence = " ".join(tgt_tokens[1:-1])  # <sos>, <eos> 제외

    plot_attention_map(
        src_tokens, tgt_tokens[:-1], attention
    )  # 마지막 <eos> 토큰은 제외하고 그림
