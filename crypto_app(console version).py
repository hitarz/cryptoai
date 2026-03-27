import json
import feedparser
import os
import sys
from openai import OpenAI
import google.generativeai as genai

def fetch_crypto_news(coin_name):
    """Отримує 5 останніх новин через Google News RSS"""
    query = f"{coin_name} cryptocurrency news".replace(" ", "%20")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    
    feed = feedparser.parse(url)
    entries = feed.entries[:5]
    
    text_for_ai = "\n".join([f"- {e.title} ({e.source.title})" for e in entries])
    return text_for_ai, entries

def parse_ai_json(response_text):
    """Очищає відповідь ШІ від маркдауну та парсить JSON"""
    clean_text = response_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        return {"action": "NEUTRAL", "confidence": 0, "summary": "Помилка парсингу відповіді від ШІ.", "risk_level": "UNKNOWN"}

def analyze_with_openai(news_text, coin_name, api_key, model):
    client = OpenAI(api_key=api_key)
    prompt = f"""
    Analyze crypto news for '{coin_name}'. Return ONLY valid JSON:
    {{ "action": "BUY"|"SELL"|"HOLD"|"NEUTRAL", "confidence": 0-100, "summary": "Ukrainian, max 3 sentences", "risk_level": "LOW"|"MEDIUM"|"HIGH" }}
    News: {news_text}
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a financial analyst outputting strictly in JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    return parse_ai_json(response.choices[0].message.content)

def analyze_with_gemini(news_text, coin_name, api_key, model_id):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_id)
    
    prompt = f"""
    Analyze market sentiment from these news headlines about '{coin_name}'.
    Respond ONLY with a valid JSON object matching this structure:
    {{
        "action": "BUY" or "SELL" or "HOLD" or "NEUTRAL",
        "confidence": <integer 0-100>,
        "summary": "<Твій аналіз українською мовою, максимум 3 речення>",
        "risk_level": "LOW" or "MEDIUM" or "HIGH"
    }}
    News:
    {news_text}
    """
    response = model.generate_content(prompt)
    return parse_ai_json(response.text)

def main():
    print("=== AI Crypto Predictor (CLI) ===")
    
    provider_choice = input("Оберіть провайдера (1 - Google Gemini, 2 - OpenAI) [1]: ").strip() or "1"
    
    if provider_choice == "1":
        provider = "Gemini"
        model_name = input("Модель (1 - gemini-2.5-flash, 2 - gemini-2.5-pro) [1]: ").strip()
        model_name = "models/gemini-2.5-pro" if model_name == "2" else "models/gemini-2.5-flash"
        api_key = os.getenv("GEMINI_API_KEY") or input("Введіть Gemini API Key: ").strip()
    elif provider_choice == "2":
        provider = "OpenAI"
        model_name = "gpt-4o"
        api_key = os.getenv("OPENAI_API_KEY") or input("Введіть OpenAI API Key: ").strip()
    else:
        print("Помилка: Невірний вибір провайдера.")
        sys.exit(1)

    if not api_key:
        print("Помилка: API ключ не надано.")
        sys.exit(1)

    coin_input = input("\nВведіть назву криптовалюти [Bitcoin]: ").strip() or "Bitcoin"
    
    print(f"\n🔍 Збір даних та аналіз {coin_input} ({model_name})...")
    
    news_text, articles = fetch_crypto_news(coin_input)
    
    if not articles:
        print("Помилка: Не вдалося знайти свіжі новини за цим запитом.")
        sys.exit(1)

    try:
        if provider == "OpenAI":
            result = analyze_with_openai(news_text, coin_input, api_key, model_name)
        else:
            result = analyze_with_gemini(news_text, coin_input, api_key, model_name)
    except Exception as e:
        print(f"Помилка при зверненні до ШІ API: {e}")
        sys.exit(1)

    # Виведення результатів
    print("\n" + "="*40)
    print(f"РЕЗУЛЬТАТ АНАЛІЗУ: {coin_input.upper()}")
    print("="*40)
    print(f"Дія:         {result.get('action', 'NEUTRAL').upper()}")
    print(f"Впевненість: {result.get('confidence', 0)}%")
    print(f"Ризик:       {result.get('risk_level', 'UNKNOWN')}")
    print("-" * 40)
    print(f"Висновок ШІ:\n{result.get('summary', 'Немає опису')}")
    print("="*40)
    
    print("\nДжерела новин:")
    for i, article in enumerate(articles, 1):
        print(f"{i}. {article.title}")
        print(f"   Джерело: {article.source.title} | {article.link}")

if __name__ == "__main__":
    main()