import os
import json
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import requests
import openai
from serpapi import GoogleSearch
from functools import lru_cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_KEY")

app = Flask(__name__)

# Rate limiting: 5 requests/minute per IP
limiter = Limiter(get_remote_address, app=app, default_limits=["5 per minute"])

# Cache scraper results to avoid duplicate requests
@lru_cache(maxsize=100)
def cached_scrape(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return ' '.join(paragraphs)[:3000]
    except Exception as e:
        print(f"Scraping error at {url}: {e}")
        return ""

def search_company_pages(company_name, domain):
    query = f"{company_name} site:{domain} (about OR mission OR leadership OR team OR vision)"
    search = GoogleSearch({
        "engine": "google",
        "q": query,
        "api_key": serpapi_key
    })
    results = search.get_dict()
    urls = [res['link'] for res in results.get('organic_results', [])[:5]]
    return urls

def summarize_with_gpt(company_name, combined_text):
    prompt = f"""
You are an assistant helping analyze company information.

Extract and return a JSON object with the following fields:
- "goals": A concise summary of the company's mission or goals.
- "outlook": Strategic future direction or plans.
- "titles": Any leadership roles or executive titles mentioned.

If information is missing, return "Not Found" for that field.

TEXT TO ANALYZE:
{combined_text}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response['choices'][0]['message']['content']

@app.route('/run', methods=['POST'])
@limiter.limit("5 per minute")
def run_agent():
    try:
        data = request.get_json()
        company = data.get('companyName')
        website = data.get('website')

        if not company or not website:
            return jsonify({"error": "Missing companyName or website"}), 400

        domain = website.replace("https://", "").replace("http://", "").split("/")[0]
        urls = search_company_pages(company, domain)

        combined_text = ""
        for url in urls:
            combined_text += cached_scrape(url) + "\n"

        if not combined_text.strip():
            return jsonify({"error": "No meaningful content found"}), 404

        summary = summarize_with_gpt(company, combined_text)

        return jsonify({
            "companyName": company,
            "website": website,
            "goals": extract_section(summary, "goals"),
            "outlook": extract_section(summary, "outlook"),
            "titles": extract_section(summary, "titles"),
            "raw_summary": summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def extract_section(text, keyword):
    pattern = rf"{keyword}[^:]*[:\-–]\s*(.*?)(?=(\n|$|\w+:))"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else "Not Found"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
