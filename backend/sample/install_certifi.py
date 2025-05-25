# install_certifi.py
import certifi
import ssl
import urllib.request

# ダミーのリクエストを送って、証明書バンドルを使えることを確認
print("証明書の場所:", certifi.where())
context = ssl.create_default_context(cafile=certifi.where())

# テスト接続
with urllib.request.urlopen("https://huggingface.co", context=context) as response:
    print("HTTPS 通信に成功しました:", response.status)
