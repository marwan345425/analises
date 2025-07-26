import asyncio
from telethon import TelegramClient
from telethon.sessions import StringSession
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# بيانات حسابك
api_id = 20476193
api_hash = "170c89a01fb7265b35ebad7fc3cc72e3"
string_session = "1BJWap1sBu4Wb4J5vtBXfOJp8mGggDXMudBZC1SBkZW3FKFN8wKiyWkZ93sB4wes8WxFTdR8vz67pFxIldY6y6QfKgrNnsnlwiPufdnEMmpfH4teM4ww2MrGF9H2Ef5CIlAGT94JkkJLuDSJn0PGFCzK2tqyofnk4_N0gy45qhgWaPpEta8tah6Eabql5KQJ-sWakwELyzQdVF3weZBpBf94QS5cmGHzISvc_ED-z3xjYFdAMbKHdPEWOxWqiKcEUK4KY-x0xa6tETh5zriMOLm8QK2wOMRS2uToO1qaYctTn_5dip5qJQYLSrqqgXrzbUEgHn8PI9e55d_KUcnb9EFYYQ8ael40="

client = TelegramClient(StringSession(string_session), api_id, api_hash)

# تهيئة موديل تحليل المشاعر
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# كلمات أسلوب الكلام
formal_words = set(["حضرتك", "سيدي", "رجاءً", "معذرة", "شكراً", "السلام", "وعليكم"])
informal_words = set(["يا", "ليش", "أبد", "حلو", "تمام", "مو"])
joke_words = set(["هههه", "هاها", "lol", "مزح"])
stress_words = set(["ضيق", "تعب", "مزعج", "مستعجل"])
bored_words = set(["ملل", "تعبت", "ما في", "براحتك"])

# كلمات MBTI تقريبي
sensing_words = set(['تفاصيل', 'حقيقي', 'واقع', 'معلوم', 'واضح', 'مباشر'])
intuition_words = set(['أفكار', 'إبداع', 'خيال', 'نظري', 'ممكن', 'فرص'])
thinking_words = set(['منطقي', 'تحليل', 'قرار', 'حق', 'خطأ', 'دليل'])
feeling_words = set(['حب', 'مشاعر', 'قلق', 'حزن', 'تأثير', 'تقدير'])

async def fetch_all_messages_with_user(target_username_or_id):
    await client.start()
    me = await client.get_me()

    if target_username_or_id.isdigit():
        target = await client.get_entity(int(target_username_or_id))
    else:
        target = await client.get_entity(target_username_or_id)

    print(f"تحليل الرسائل مع: {target.first_name} (ID: {target.id})")

    all_messages = []

    # رسائل الخاص
    async for message in client.iter_messages(target, limit=None):
        if message.text:
            all_messages.append({
                "date": message.date,
                "sender_id": message.sender_id,
                "text": message.text
            })

    # رسائل القروبات المشتركة
    dialogs = await client.get_dialogs()
    for dialog in dialogs:
        entity = dialog.entity
        try:
            participants = await client.get_participants(entity)
            user_ids = [p.id for p in participants]
            if target.id in user_ids:
                async for msg in client.iter_messages(entity, limit=None):
                    if msg.text and (msg.sender_id == target.id or msg.sender_id == me.id):
                        all_messages.append({
                            "date": msg.date,
                            "sender_id": msg.sender_id,
                            "text": msg.text
                        })
        except Exception:
            continue

    print(f"تم جمع {len(all_messages)} رسالة.")

    return pd.DataFrame(all_messages), target, me

def analyze_style(text):
    words = set(text.lower().split())
    return {
        "formal": len(words & formal_words),
        "informal": len(words & informal_words),
        "joke": len(words & joke_words),
        "stress": len(words & stress_words),
        "bored": len(words & bored_words),
    }

def mbti_analysis(df, target_id):
    total_msgs = len(df)
    user_msgs = df[df.sender_id == target_id]
    user_msg_ratio = len(user_msgs) / total_msgs if total_msgs > 0 else 0

    ei = 'E' if user_msg_ratio > 0.4 else 'I'

    sensing_count = 0
    intuition_count = 0
    for text in user_msgs['text']:
        words = set(text.lower().split())
        sensing_count += len(words & sensing_words)
        intuition_count += len(words & intuition_words)
    sn = 'S' if sensing_count >= intuition_count else 'N'

    thinking_count = 0
    feeling_count = 0
    for text in user_msgs['text']:
        words = set(text.lower().split())
        thinking_count += len(words & thinking_words)
        feeling_count += len(words & feeling_words)
    tf = 'T' if thinking_count >= feeling_count else 'F'

    df_user = user_msgs.sort_values('date')
    time_diffs = df_user['date'].diff().dt.total_seconds().dropna()
    avg_diff = time_diffs.mean() if len(time_diffs) > 0 else 1000000
    jp = 'J' if avg_diff < 3600 * 12 else 'P'

    return ei + sn + tf + jp

def analyze_messages(df, target, me):
    sentiments = []
    styles = []
    for text in df['text']:
        try:
            result = sentiment_analyzer(text[:512])[0]
        except Exception:
            result = {"label": "NEUTRAL", "score": 0.0}
        sentiments.append(result)
        styles.append(analyze_style(text))

    df['sentiment'] = [s['label'] for s in sentiments]
    df['sentiment_score'] = [s['score'] for s in sentiments]

    style_df = pd.DataFrame(styles)
    df = pd.concat([df.reset_index(drop=True), style_df.reset_index(drop=True)], axis=1)

    sender_counts = df['sender_id'].value_counts()

    df['date_only'] = df['date'].dt.date
    first_msgs = df.sort_values('date').groupby('date_only').first()
    starters = first_msgs['sender_id'].value_counts()

    return df, sender_counts, starters

def generate_report(df, sender_counts, starters, target, me):
    print("\n--- تقرير تحليلي مفصل ---")
    print(f"الشخص المستهدف: {target.first_name} (ID: {target.id})")
    print(f"عدد الرسائل الكلي: {len(df)}")
    print(f"عدد الرسائل منك: {sender_counts.get(me.id,0)}")
    print(f"عدد الرسائل من {target.first_name}: {sender_counts.get(target.id,0)}")

    print("\nتوزيع المشاعر:")
    print(df['sentiment'].value_counts())

    print("\nمن يبدأ المحادثة يوميًا:")
    for sender_id, count in starters.items():
        name = me.first_name if sender_id == me.id else target.first_name
        print(f"{name}: {count} مرات")

    mbti = mbti_analysis(df, target.id)
    print(f"\n🔍 نمط شخصية MBTI المُقدّر: {mbti}")

    daily_sentiment = df.groupby(['date_only', 'sentiment']).size().unstack(fill_value=0)
    plt.figure(figsize=(14,7))
    sns.lineplot(data=daily_sentiment)
    plt.title('توزيع المشاعر عبر الزمن')
    plt.xlabel('التاريخ')
    plt.ylabel('عدد الرسائل')
    plt.legend(title='المشاعر')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    starters_names = [me.first_name, target.first_name]
    starters_counts = [starters.get(me.id, 0), starters.get(target.id, 0)]
    sns.barplot(x=starters_names, y=starters_counts)
    plt.title('عدد مرات بدء المحادثة يوميًا')
    plt.ylabel('عدد المرات')
    plt.tight_layout()
    plt.show()
  
# إرسال ملخص التحليل إلى يوزر معيّن
    summary = f"""
📊 تحليل المحادثة مع {target.first_name}:

- مجموع الرسائل: {len(df)}
- رسائلك: {sender_counts.get(me.id,0)}
- رسائل {target.first_name}: {sender_counts.get(target.id,0)}

🔁 من يبدأ المحادثة:
{me.first_name}: {starters.get(me.id, 0)} مرة
{target.first_name}: {starters.get(target.id, 0)} مرة

❤ توزيع المشاعر:
{df['sentiment'].value_counts().to_string()}

🧠 نمط شخصية MBTI المتوقع: {mbti}
"""
    try:
        asyncio.create_task(client.send_message("@Leeo71", summary))
        print("✅ تم إرسال ملخص التحليل إلى @Leeo71")
    except Exception as e:
        print(f"❌ فشل إرسال الرسالة: {e}")
      
async def main():
    target_input = input("ادخل يوزر الشخص أو الآيدي: ").strip()
    df, target, me = await fetch_all_messages_with_user(target_input)
    if df.empty:
        print("لا توجد رسائل لتحليلها.")
        return
    df, sender_counts, starters = analyze_messages(df, target, me)
    generate_report(df, sender_counts, starters, target, me)

if __name__ == "__main__":
    asyncio.run(main())
