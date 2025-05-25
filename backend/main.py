# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import json
from utils.llm_handler import extract_keywords_from_problem

app = FastAPI()

# モデル
class ProblemRequest(BaseModel):
    text: str

# 判例データのロード
with open("data/hanrei.json", encoding="utf-8") as f:
    hanrei_data = json.load(f)

@app.post("/search")
def search_hanrei(req: ProblemRequest):
    keywords = extract_keywords_from_problem(req.text)
    matched = []

    for hanrei in hanrei_data:
        if any(kw in hanrei["keywords"] for kw in keywords):
            matched.append({
                "title": hanrei["title"],
                "summary": hanrei["summary"],
                "matched_keywords": [kw for kw in keywords if kw in hanrei["keywords"]]
            })

    return {
        "input_keywords": keywords,
        "matched_hanrei": matched if matched else "該当する判例が見つかりませんでした。"
    }
