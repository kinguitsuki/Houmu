from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "rinna/japanese-gpt2-small"

print("トークナイザ読み込み中...")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

print("モデル読み込み中...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    offload_folder="./offload"
)

print("生成テスト...")
input_text = "日本の司法制度について教えてください。"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# attention_mask の作成
attention_mask = torch.ones_like(input_ids)

# GPUがあれば移動
if torch.cuda.is_available():
    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")
    model = model.to("cuda")

# テキスト生成
output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=10,
    pad_token_id=tokenizer.eos_token_id  # 警告を避けるため
)

# 結果表示
print(tokenizer.decode(output[0], skip_special_tokens=True))


