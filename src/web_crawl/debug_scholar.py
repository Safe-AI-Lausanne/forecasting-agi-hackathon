import requests
from bs4 import BeautifulSoup
import re

url = "https://scholar.google.com/scholar?q=TruthfulQA%3A+Measuring+How+Models+Mimic+Human+Falsehoods+Lin+Hilton+Evans"

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
}

print("Fetching URL...")
response = requests.get(url, headers=headers)
print(f"Status Code: {response.status_code}\n")

soup = BeautifulSoup(response.text, 'html.parser')

# Save HTML for inspection
with open('debug_page.html', 'w', encoding='utf-8') as f:
    f.write(soup.prettify())
print("Saved HTML to debug_page.html\n")

# Look for different result containers
print("=== Looking for result containers ===")
print(f"gs_r gs_or gs_scl: {len(soup.find_all('div', class_='gs_r gs_or gs_scl'))}")
print(f"gs_r: {len(soup.find_all('div', class_='gs_r'))}")
print(f"gs_ri: {len(soup.find_all('div', class_='gs_ri'))}")

# Try to find any div with 'gs_r' in class
gs_results = soup.find_all('div', class_=re.compile(r'gs_r'))
print(f"Any div with gs_r in class: {len(gs_results)}")

if gs_results:
    print("\n=== First result structure ===")
    first_result = gs_results[0]
    print(first_result.prettify()[:2000])

    # Look for cited by
    cited_by = first_result.find_all('a', string=re.compile(r'Cited by|cited by', re.IGNORECASE))
    print(f"\n=== Cited by links found: {len(cited_by)} ===")
    for link in cited_by:
        print(f"Text: {link.text}")
        print(f"Href: {link.get('href')}")

# Also search entire page for "Cited by"
print("\n=== All 'Cited by' text on page ===")
all_cited = soup.find_all(string=re.compile(r'Cited by', re.IGNORECASE))
print(f"Found {len(all_cited)} instances")
for i, text in enumerate(all_cited[:5]):
    print(f"{i+1}. {text}")
    if text.parent:
        print(f"   Parent tag: {text.parent.name}")
        print(f"   Parent: {text.parent}")
