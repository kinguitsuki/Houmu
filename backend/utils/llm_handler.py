# utils/llm_handler.py

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

if torch.cuda.is_available():
    model = model.to("cuda")

def extract_keywords_from_problem(problem_text: str) -> list:
    prompt = f"""
    あなたは司法試験に詳しい法律アシスタントです。
    以下の問題文から、関係する法律論点を3つ程度、キーワード形式で抽出してください。
    問題文：
    {problem_text}
    キーワード：
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)

    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,  # キーワードなので適度に
        pad_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # プロンプトの入力文を含むので、それ以降の生成テキストだけ抜き出す
    extracted = output_text.split("キーワード：")[-1]

    # 改行で区切り、箇条書きや・があれば除去
    keywords = [kw.strip("・ ・\n\r") for kw in extracted.strip().split("\n") if kw.strip()]
    return keywords

