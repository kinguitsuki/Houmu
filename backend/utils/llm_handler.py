# utils/llm_handler.py
import openai

openai.api_key = "YOUR_API_KEY"  # 環境変数で管理推奨

def extract_keywords_from_problem(problem_text: str) -> list:
    prompt = f"""
    あなたは司法試験に詳しい法律アシスタントです。
    以下の問題文から、関係する法律論点を3つ程度、キーワード形式で抽出してください。
    問題文：
    {problem_text}
    キーワード：
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    keywords_text = response.choices[0].message.content
    return [kw.strip("・ ") for kw in keywords_text.strip().split("\n") if kw.strip()]
