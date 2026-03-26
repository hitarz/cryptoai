import streamlit as st
import feedparser
from openai import OpenAI
import google.generativeai as genai
import json

#  Настройка страницы 
st.set_page_config(page_title="AI Crypto Analyzer", page_icon="💰", layout="centered")

# CSS Стили
st.markdown("""
    <style>
    .stButton>button { width: 100%; background-color: #FF4B4B; color: white; border: none; font-weight: bold; }
    .news-box { background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #FF4B4B; color: #31333F !important; }
    .news-title { font-weight: bold; font-size: 16px; margin-bottom: 5px; display: block; color: #31333F !important; }
    .news-link { color: #0068c9 !important; text-decoration: none; font-size: 14px; }
    .news-link:hover { text-decoration: underline; }
    </style>
    """, unsafe_allow_html=True)

#Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    ai_provider = st.selectbox("Выберите ИИ провайдера", ["Google Gemini", "OpenAI"])
    
    if ai_provider == "Google Gemini":
        gemini_model_choice = st.selectbox(
            "Выберите модель Gemini", 
            ["models/gemini-2.5-flash", "models/gemini-2.5-pro"]
        )
        model_name = gemini_model_choice
        api_key = st.text_input("Введите Gemini API Key", type="password")
    else:
        model_name = "gpt-4o"
        api_key = st.text_input("Введите OpenAI API Key", type="password")

    st.info("Ключ используется только для этой сессии.")
    st.markdown("---")
    st.caption("v1.4 | Поддержка Gemini 2.5")

#Функции 

def get_specific_news(coin_name):
    query = f"{coin_name} crypto currency news"
    formatted_query = query.replace(" ", "%20")
    rss_url = f"https://news.google.com/rss/search?q={formatted_query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    headlines = [f"- {entry.title} (Source: {entry.source.title})" for entry in feed.entries[:5]]
    return "\n".join(headlines), feed.entries[:5]

def analyze_with_openai(news_text, coin_name, api_key, model):
    client = OpenAI(api_key=api_key)
    prompt = f"""
    Analyze crypto news for '{coin_name}'. Return ONLY JSON:
    {{ "action": "BUY"|"SELL"|"HOLD"|"NEUTRAL", "confidence": 0-100, "summary": "Ukrainian, max 3 sentences", "risk_level": "LOW"|"MEDIUM"|"HIGH" }}
    News: {news_text}
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a JSON financial assistant."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def analyze_with_gemini(news_text, coin_name, api_key, model_id):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_id)
    
    prompt = f"""
    Analyze the following news headlines regarding the cryptocurrency '{coin_name}'.
    Determine market sentiment and suggest action.
    Return ONLY a valid JSON object:
    {{
        "action": "BUY",
        "confidence": 85,
        "summary": "Ваш аналіз українською мовою тут (3 речення).",
        "risk_level": "MEDIUM"
    }}
    News:
    {news_text}
    """
    response = model.generate_content(prompt)
    text_response = response.text
    if "```json" in text_response:
        text_response = text_response.split("```json")[1].split("```")[0]
    elif "```" in text_response:
        text_response = text_response.split("```")[1].split("```")[0]
        
    return json.loads(text_response.strip())

#Главный Интерфейс

st.title("💰 AI Crypto Predictor")
st.markdown(f"Анализ через: **{model_name}**")

col1, col2 = st.columns([3, 1])
with col1:
    coin_input = st.text_input("Название криптовалюты", "Bitcoin")
with col2:
    st.write("") 
    st.write("") 
    analyze_btn = st.button("АНАЛИЗИРОВАТЬ")

if analyze_btn:
    if not api_key:
        st.error(f"⚠️ Пожалуйста, введите API Key для {ai_provider}!")
    else:
        with st.status(f"🔍 Анализирую {coin_input} используя {model_name}...", expanded=True) as status:
            news_text, top_articles = get_specific_news(coin_input)
            
            if not top_articles:
                status.update(label="Новости не найдены", state="error")
                st.error("Не удалось найти свежих новостей.")
            else:
                try:
                    if ai_provider == "OpenAI":
                        result = analyze_with_openai(news_text, coin_input, api_key, model_name)
                    else:
                        result = analyze_with_gemini(news_text, coin_input, api_key, model_name)
                    
                    status.update(label="Анализ завершен!", state="complete", expanded=False)

                    # --- Отображение результатов ---
                    st.divider()
                    
                    action = result.get('action', 'NEUTRAL')
                    colors = {"BUY": "#09ab3b", "SELL": "#ff2b2b", "HOLD": "#ffa500", "NEUTRAL": "#ffa500"}
                    color = colors.get(action, "#ffa500")
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"<h2 style='color: {color}; text-align: center; margin: 0;'>{action}</h2>", unsafe_allow_html=True)
                        st.markdown("<p style='text-align: center; color: gray;'>Рекомендация</p>", unsafe_allow_html=True)
                    with c2:
                        st.metric("Уверенность AI", f"{result.get('confidence', 0)}%")
                    with c3:
                        st.metric("Риск", result.get('risk_level', 'UNKNOWN'))

                    st.info(f"💡 **Анализ:** {result.get('summary', 'Нет данных')}")
                    
                    st.subheader("📰 Источники:")
                    for article in top_articles:
                        st.markdown(f"""
                        <div class="news-box">
                            <span class="news-title">{article.title}</span>
                            <small>Источник: {article.source.title}</small><br>
                            <a href="{article.link}" target="_blank" class="news-link">Читать статью →</a>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    status.update(label="Ошибка анализа", state="error")
                    st.error(f"Произошла ошибка: {str(e)}")