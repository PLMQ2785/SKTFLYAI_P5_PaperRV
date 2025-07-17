import torch
import torch.nn as nn
import math
from konlpy.tag import Mecab
import time
# from torchtext.data.metrics import bleu_score
import sacrebleu
from thop import profile
import warnings
from tqdm import tqdm

# thop 라이브러리의 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning)


# --- 0. 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3


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
        self, num_encoder_layers, num_decoder_layers, emb_size, nhead,
        src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout,
        )
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(
        self, src, trg, src_mask, tgt_mask,
        src_padding_mask, tgt_padding_mask, memory_key_padding_mask,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb, tgt_emb, src_mask, tgt_mask, None,
            src_padding_mask, tgt_padding_mask, memory_key_padding_mask,
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


# --- 2. 번역 및 평가 함수 ---
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz, device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float().masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """Greedy Search를 사용하여 번역 시퀀스를 생성하는 헬퍼 함수"""
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    with torch.no_grad():
        memory = model.encode(src, src_mask)
    memory = memory.to(DEVICE)

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
        with torch.no_grad():
            out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def translate(model, src_sentence, ko_vocab, en_vocab, ko_tokenizer):
    model.eval()
    src_tokens = [ko_vocab.get(token, UNK_IDX) for token in ko_tokenizer(src_sentence)]
    src_tensor = (torch.LongTensor([SOS_IDX] + src_tokens + [EOS_IDX]).unsqueeze(1))
    
    num_tokens = src_tensor.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    
    tgt_tokens_indices = greedy_decode(model, src_tensor, src_mask, max_len=num_tokens + 5, start_symbol=SOS_IDX).flatten()
    
    # 인덱스를 단어로 변환
    # 어휘집에 없는 인덱스는 무시
    rev_en_vocab = {i: word for word, i in en_vocab.items()}
    tgt_tokens = [rev_en_vocab.get(i, "") for i in tgt_tokens_indices.cpu().numpy()][1:] # <sos> 제외
    
    return " ".join(tgt_tokens).replace(" <eos>", "")


def calculate_bleu(model, test_data_path, ko_vocab, en_vocab, ko_tokenizer):
    """테스트 데이터셋 전체에 대한 BLEU 스코어를 계산 (sacrebleu 사용)"""
    
    with open(test_data_path['ko'], 'r', encoding='utf-8') as f:
        test_ko_sents = [line.strip() for line in f.readlines()]
    with open(test_data_path['en'], 'r', encoding='utf-8') as f:
        # sacrebleu는 참조(정답) 문장을 리스트의 리스트 형태로 받습니다. [[ref1], [ref2], ...]
        targets = [[line.strip()] for line in f.readlines()]
        
    predictions = []
    
    for sentence in tqdm(test_ko_sents, desc="Calculating BLEU"):
        prediction_str = translate(model, sentence, ko_vocab, en_vocab, ko_tokenizer)
        predictions.append(prediction_str)
        
    # sacrebleu.corpus_bleu 함수로 점수 계산
    bleu = sacrebleu.corpus_bleu(predictions, targets)
    
    return bleu.score


# --- 3. 메인 실행 로직 ---
if __name__ == "__main__":
    # 저장된 어휘집 불러오기
    print("어휘집을 불러옵니다...")
    ko_vocab = torch.load("ko_vocab.pth")
    en_vocab = torch.load("en_vocab.pth")
    SRC_VOCAB_SIZE = len(ko_vocab)
    TGT_VOCAB_SIZE = len(en_vocab)

    # 토크나이저 정의
    def en_tokenizer_func(text): return text.lower().split()
    ko_tokenizer = Mecab().morphs
    en_tokenizer = en_tokenizer_func

    # 하이퍼파라미터
    EMB_SIZE, NHEAD = 256, 8
    NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS = 3, 3

    # 모델 초기화 및 가중치 불러오기
    print("모델 가중치를 불러옵니다...")
    model = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD,
        SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, dim_feedforward=512
    )
    model.load_state_dict(torch.load("transformer_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    
    # --- 단일 문장 번역 ---
    kor_sentence = "소프트웨어에 의한 암호화는 아주 가격이 저렴하고, 일단 상품을 구입한 이후에는 자연히 돈이 들지 않게 됩니다."
    
    start_time = time.time()
    translated_sentence = translate(model, kor_sentence, ko_vocab, en_vocab, ko_tokenizer)
    end_time = time.time()
    
    print("\n=====================================")
    print(f"입력 (한국어): {kor_sentence}")
    print(f"번역 (영어): {translated_sentence}")
    print("=====================================")

    # --- 연산량 추정 ---
    print("\n--- 연산량 추정 ---")
    # 1. 파라미터 수
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델 파라미터 수: {num_params:,}")

    # 2. FLOPs (입력 크기를 가정해야 함)
    # 예시로 (시퀀스 길이, 배치 크기) = (25, 1)로 가정
    dummy_src = torch.randint(0, SRC_VOCAB_SIZE, (25, 1), device=DEVICE)
    dummy_tgt = torch.randint(0, TGT_VOCAB_SIZE, (25, 1), device=DEVICE)
    # 모델의 forward 함수에 맞는 더미 마스크 생성
    dummy_src_mask = torch.zeros(25, 25, device=DEVICE).type(torch.bool)
    dummy_tgt_mask = generate_square_subsequent_mask(25).type(torch.bool)
    dummy_src_padding_mask = torch.zeros(1, 25, device=DEVICE).type(torch.bool)
    dummy_tgt_padding_mask = torch.zeros(1, 25, device=DEVICE).type(torch.bool)

    # thop.profile을 사용하여 FLOPs 계산
    macs, _ = profile(model, inputs=(dummy_src, dummy_tgt, dummy_src_mask, dummy_tgt_mask,
                                       dummy_src_padding_mask, dummy_tgt_padding_mask, dummy_src_padding_mask), verbose=False)
    # MACs(Multiply-Accumulate)는 FLOPs의 약 2배이므로, GFLOPs는 Giga-MACs * 2
    gflops = (macs * 2) / 1e9
    print(f"FLOPs (추정): {gflops:.2f} GFLOPs")
    
    # 3. 추론 시간
    print(f"단일 문장 추론 시간: {end_time - start_time:.4f} 초")
    
    # --- BLEU 스코어 계산 ---
    test_files = {'ko': 'korean-english-park.test.ko', 'en': 'korean-english-park.test.en'}
    bleu = calculate_bleu(model, test_files, ko_vocab, en_vocab, ko_tokenizer)
    print(f"\n테스트셋 BLEU 스코어: {bleu:.2f}")