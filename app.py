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
        scraped = '\n'.join(parts)[:5000]
        print(f"\n--- Scraped content for {url} ---\n{scraped[:500]}...\n--- End ---\n")  # print first 500 chars
        return scraped
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
You are a professional analyst.

Summarize the following company for a senior business audience:
- Provide a concise overview of its mission and goals.
- Describe its strategic outlook and the types of financial services it provides (look for: 401k, RIA, RR, insurance, retirement, tax services, investment strategy).
- If visible, highlight competitive advantages or industry trends.
- Write in clear, professional language, suitable for a client-facing report. Use paragraphs and bullet points if appropriate.
- DO NOT output any code, markdown, or JSON. Write for humans.

COMPANY NAME: {company_name}

TEXT TO ANALYZE:
{combined_text}
"""
    print("\n--- Prompt Sent to GPT ---")
    print(prompt[:1000])  # print the first 1000 chars for brevity
    print("--- End Prompt ---\n")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        summary = response.choices[0].message.content.strip()
        print("\n--- GPT Response ---")
        print(summary)
        print("--- End GPT Response ---\n")
        return summary
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        return f"Error from OpenAI: {e}"

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

def parse_summary(summary):
    # Just returns the summary as plain text.
    return summary

@app.route('/run', methods=['POST'])
@limiter.limit("5 per minute")
def run_agent():
    try:
        data = request.get_json(force=True)
        print(f"\n--- Incoming Request ---\n{data}\n--- End Request ---")

        company = data.get('companyName')
        website = data.get('website')

        if not company or not website:
            print("Error: Missing companyName or website.")
            return jsonify({"error": "Missing companyName or website"}), 400

        is_valid, error = validate_inputs(company, website)
        if not is_valid:
            print(f"Input validation failed: {error}")
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

        print("\n--- Combined Text to GPT ---\n")
        print(combined_text[:2000])  # print the first 2000 chars for brevity
        print("\n--- End Combined Text ---\n")

        # Always call GPT, even if combined_text is thin
        if not combined_text.strip():
            print("WARNING: No meaningful content found after scraping.")
            combined_text = "No meaningful content was scraped from the provided URLs."

        summary = summarize_with_gpt(company, combined_text)
        human_summary = parse_summary(summary)

        result = {
            "companyName": company,
            "website": website,
            "urlsUsed": urls,
            "summary": human_summary
        }

        print("\n--- API Response to Salesforce ---\n")
        print(json.dumps(result, indent=2))
        print("\n--- End API Response ---\n")

        return jsonify(result)

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
