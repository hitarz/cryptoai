import json
import feedparser
import streamlit as st
from openai import OpenAI
import google.generativeai as genai

# Базові налаштування
st.set_page_config(page_title="AI Crypto Analyzer", page_icon="💰", layout="centered")

# Кастомний CSS для карток новин та кнопки
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #FF4B4B; color: white; border: none; font-weight: bold; }
    .news-card { background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid #FF4B4B; }
    .news-title { font-weight: bold; font-size: 16px; margin-bottom: 6px; display: block; color: #1f1f1f; }
    .news-link { color: #0068c9; text-decoration: none; font-size: 14px; }
    .news-link:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)


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
    # Промпт залишається англійською для кращого розуміння моделлю, але вимагає українську мову у відповіді
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


# --- Бічна панель: Налаштування ---
with st.sidebar:
    st.header("⚙️ Налаштування")
    provider = st.selectbox("Провайдер", ["Google Gemini", "OpenAI"])
    
    if provider == "Google Gemini":
        model_name = st.selectbox("Модель", ["models/gemini-2.5-flash", "models/gemini-2.5-pro"])
        api_key = st.text_input("Gemini API Key", type="password")
    else:
        model_name = "gpt-4o"
        api_key = st.text_input("OpenAI API Key", type="password")

    st.info("Ключ зберігається лише локально в поточній сесії.")


# --- Основний UI ---
st.title("💰 AI Crypto Predictor")

# Використовуємо форму для логічного сабміту через Enter
with st.form("crypto_form"):
    coin_input = st.text_input("Введіть назву криптовалюти:", value="Bitcoin")
    submitted = st.form_submit_button("Аналізувати")

if submitted:
    if not api_key:
        st.error("⚠️ Введіть API ключ у бічній панелі!")
        st.stop()

    with st.status(f"🔍 Збір даних та аналіз {coin_input} ({model_name})...", expanded=True) as status:
        news_text, articles = fetch_crypto_news(coin_input)
        
        if not articles:
            status.update(label="Немає даних", state="error")
            st.warning("Не вдалося знайти свіжі новини за цим запитом.")
            st.stop()

        try:
            if provider == "OpenAI":
                result = analyze_with_openai(news_text, coin_input, api_key, model_name)
            else:
                result = analyze_with_gemini(news_text, coin_input, api_key, model_name)
                
            status.update(label="Аналіз завершено", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Помилка API", state="error")
            st.error(f"Збій при зверненні до ШІ: {e}")
            st.stop()

    # --- Виведення результатів ---
    action = result.get('action', 'NEUTRAL').upper()
    color_map = {"BUY": "#09ab3b", "SELL": "#ff2b2b", "HOLD": "#ffa500", "NEUTRAL": "#808080"}
    accent_color = color_map.get(action, "#808080")
    
    st.markdown("---")
    
    # Метрики
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<h2 style='color: {accent_color}; text-align: center; margin: 0;'>{action}</h2>", unsafe_allow_html=True)
    col1.markdown("<p style='text-align: center; color: gray;'>Рекомендація</p>", unsafe_allow_html=True)
    col2.metric("Впевненість", f"{result.get('confidence', 0)}%")
    col3.metric("Рівень ризику", result.get('risk_level', 'UNKNOWN'))

    # Резюме
    st.info(f"💡 **Висновок ШІ:** {result.get('summary', 'Немає опису')}")
    
    # Посилання на новини
    st.subheader("📰 Джерела")
    for article in articles:
        st.markdown(f"""
        <div class="news-card">
            <span class="news-title">{article.title}</span>
            <small style="color: #666;">Джерело: {article.source.title}</small><br>
            <a href="{article.link}" target="_blank" class="news-link">Читати в оригіналі →</a>
        </div>
        """, unsafe_allow_html=True)