import os
import re
import json
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import requests
from openai import OpenAI
from serpapi import GoogleSearch
from functools import lru_cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from urllib.parse import urlparse
from typing import List, Dict, Optional

# Load environment variables from .env
load_dotenv()

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
serpapi_key = os.getenv("SERPAPI_KEY")

app = Flask(__name__)

# Rate limiting: 5 requests/minute per IP
limiter = Limiter(get_remote_address, app=app, default_limits=["5 per minute"])

@app.route('/')
def health_check():
    return "Agent is alive", 200

def normalize_domain(website):
    if not website.startswith("http"):
        website = "https://" + website
    domain = website.replace("https://", "").replace("http://", "").split("/")[0]
    return website, domain

def fallback_urls(domain):
    return [
        f"https://{domain}/about",
        f"https://{domain}/team",
        f"https://{domain}/leadership",
        f"https://{domain}/who-we-are",
        f"https://{domain}/who-we-serve",
        f"https://{domain}/services",
        f"https://{domain}/contact-us",
        f"https://{domain}/who-we-serve",
        f"https://{domain}/fees",
        f"https://{domain}/investment-philosophy",
        f"https://{domain}/investment-strategy",
        f"https://{domain}/investment-approach",
        f"https://{domain}/about/team",
        f"https://{domain}/our-story"
    ]

@lru_cache(maxsize=100)
def smart_scrape(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, 'html.parser')
        parts = []
        parts += [p.get_text(separator=" ", strip=True) for p in soup.find_all('p')]
        for tag in ['h1','h2','h3','h4','h5','h6']:
            parts += [h.get_text(separator=" ", strip=True) for h in soup.find_all(tag)]
        parts += [li.get_text(separator=" ", strip=True) for li in soup.find_all('li')]
        for row in soup.find_all('tr'):
            cells = [td.get_text(separator=" ", strip=True) for td in row.find_all(['td', 'th'])]
            if cells:
                parts.append(" | ".join(cells))
        emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", res.text)
        parts += emails
        return '\n'.join(parts)[:5000]
    except Exception as e:
        print(f"Smart scraping error at {url}: {e}")
        return ""

def search_company_pages(company_name, domain):
    query = f"{company_name} site:{domain} (about OR mission OR leadership OR team OR vision)"
    search = GoogleSearch({
        "engine": "google",
        "q": query,
        "api_key": serpapi_key
    })
    results = search.get_dict()
    return [res['link'] for res in results.get('organic_results', [])[:5]]

def summarize_with_gpt(company_name, combined_text):
    prompt = f"""
You are an assistant analyzing company websites.

Extract and return ONLY a JSON object in this format, inside a markdown code block:

{{
  "goals": "...",
  "outlook": "..."
}}

If you cannot find information, set the field to "Not Found".

TEXT TO ANALYZE:
{combined_text}

Return only the JSON, inside a markdown code block (triple backticks), and nothing else.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

def parse_summary(summary):
    json_pattern = r"```(?:json)?\s*(\{[\s\S]*\})\s*```"
    match = re.search(json_pattern, summary)
    raw_json = match.group(1) if match else summary.strip()
    try:
        data = json.loads(raw_json)
        goals = data.get("goals", "Not Found")
        outlook = data.get("outlook", "Not Found")
        return goals, outlook
    except Exception as e:
        print("JSON parsing failed! Raw summary below:")
        print(repr(summary))
        print("Error:", e)
        return "Not Found", "Not Found"

def validate_inputs(company_name: str, website: str) -> tuple[bool, Optional[str]]:
    if len(company_name) > 200:
        return False, "Company name too long"
    try:
        result = urlparse(website)
        if not all([result.scheme, result.netloc]):
            return False, "Invalid website URL"
    except Exception:
        return False, "Invalid website format"
    return True, None

@app.route('/run', methods=['POST'])
@limiter.limit("5 per minute")
def run_agent():
    try:
        data = request.get_json(force=True)
        print(f"Received payload: {data}")

        company = data.get('companyName')
        website = data.get('website')

        if not company or not website:
            return jsonify({"error": "Missing companyName or website"}), 400

        is_valid, error = validate_inputs(company, website)
        if not is_valid:
            return jsonify({"error": error}), 400

        website, domain = normalize_domain(website)
        print(f"Normalized website: {website}")
        print(f"Final domain used: {domain}")

        urls = search_company_pages(company, domain)
        print(f"SerpAPI URLs for {company}: {urls}")

        if not urls:
            print(f"[Fallback] Using hardcoded URLs for domain {domain}")
            urls = fallback_urls(domain)

        combined_text = ""
        for url in urls:
            combined_text += smart_scrape(url) + "\n"

        if not combined_text.strip():
            return jsonify({
                "error": "No meaningful content found after scraping",
                "scraped_urls": urls
            }), 404

        summary = summarize_with_gpt(company, combined_text)
        goals, outlook = parse_summary(summary)

        return jsonify({
            "companyName": company,
            "website": website,
            "urlsUsed": urls,
            "goals": goals,
            "outlook": outlook,
            "raw_summary": summary
        })

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
