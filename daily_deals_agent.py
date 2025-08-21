import os
import smtplib
import asyncio
from email.mime.text import MIMEText
from playwright.async_api import async_playwright
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# -----------------------------
# 🔐 ENV VARIABLES (set in Render or Replit)
# -----------------------------
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMAIL_USER = os.environ["EMAIL_USERNAME"]
EMAIL_PASS = os.environ["EMAIL_PASSWORD"]

# -----------------------------
# 📦 PRODUCT CATEGORIES & CRITERIA
# -----------------------------
CATEGORIES = [
    {"search": "men's polos", "max_price": 15, "min_discount": 60},
    {"search": "men's chinos", "max_price": 25, "min_discount": 60},
    {"search": "men's dress pants", "max_price": 25, "min_discount": 60},
    {"search": "men's long sleeve button up shirt", "max_price": 20, "min_discount": 70},
    {"search": "men's sunglasses", "max_price": 10, "min_discount": 0},
    {"search": "men's oxford shoes", "max_price": 40, "min_discount": 70},
]

# -----------------------------
# 🔍 SCRAPE AMAZON (Playwright)
# -----------------------------
async def scrape_amazon():
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for cat in CATEGORIES:
            query = cat["search"].replace(" ", "+")
            url = f"https://www.amazon.com/s?k={query}&s=price-asc-rank"
            await page.goto(url, timeout=60000)

            items = await page.query_selector_all("div.s-result-item")

            for item in items[:10]:
                title = await item.query_selector_eval("h2", "e => e.innerText", strict=False) if await item.query_selector("h2") else None
                price_whole = await item.query_selector_eval(".a-price-whole", "e => e.innerText", strict=False) if await item.query_selector(".a-price-whole") else None
                price_fraction = await item.query_selector_eval(".a-price-fraction", "e => e.innerText", strict=False) if await item.query_selector(".a-price-fraction") else "00"
                link = await item.query_selector_eval("h2 a", "e => e.href", strict=False) if await item.query_selector("h2 a") else None
                original_price = await item.query_selector_eval(".a-text-price .a-offscreen", "e => e.innerText", strict=False) if await item.query_selector(".a-text-price .a-offscreen") else None

                if title and price_whole:
                    price = float(price_whole.replace(",", "") + "." + price_fraction)
                    orig_price = float(original_price.replace("$", "").replace(",", "")) if original_price else price
                    discount = round(100 * (orig_price - price) / orig_price) if orig_price > price else 0

                    if (price <= cat["max_price"] or discount >= cat["min_discount"]):
                        results.append({
                            "title": title,
                            "price": f"${price:.2f}",
                            "original_price": f"${orig_price:.2f}",
                            "discount": f"{discount}%",
                            "link": link,
                            "category": cat["search"]
                        })

        await browser.close()
    return results

# -----------------------------
# 💬 FORMAT USING LLM (LangChain)
# -----------------------------
def summarize_results(items):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.3)

    item_lines = "\n".join([f"""
- **{item['title']}**
  - Price: {item['price']} (was {item['original_price']} | {item['discount']} off)
  - Category: {item['category']}
  - Link: {item['link']}
""" for item in items])

    template = PromptTemplate.from_template("""
You are a fashion assistant. You’ve collected a list of discounted men's fashion items. Write a concise, stylish daily summary email for a shopper interested in recent deals. Keep it clean and easy to skim.

Here are the items:
{item_lines}

If no items are present, return: "No qualifying deals found today. Will retry tomorrow."
""")

    final_prompt = template.format(item_lines=item_lines)
    response = llm.predict(final_prompt)
    return response

# -----------------------------
# 📧 SEND EMAIL
# -----------------------------
def send_email(subject, body):
    msg = MIMEText(body, "html")
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = "matthewlfarley@gmail.com"

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)

# -----------------------------
# ✅ MAIN FUNCTION
# -----------------------------
def run_agent():
    print("Running Amazon fashion agent...")

    results = asyncio.run(scrape_amazon())
    summary = summarize_results(results)
    subject = "🛍️ Daily Men's Fashion Deals – Amazon"
    send_email(subject, summary)

    print("Email sent successfully.")

if __name__ == "__main__":
    run_agent()


