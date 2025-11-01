import pandas as pd
import requests
from tqdm import tqdm
import time

def check_link(url, headers):
    """Check if a URL is accessible and returns citations."""
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            # Check if page contains "Cited by" or "Zitiert von"
            if 'Cited by' in response.text or 'Zitiert von' in response.text or 'cited by' in response.text.lower():
                return 'Valid - Has citations'
            else:
                return 'Valid - No "Cited by" found'
        elif response.status_code == 429:
            return 'Rate limited (429)'
        else:
            return f'HTTP {response.status_code}'
    except Exception as e:
        return f'Error: {str(e)[:50]}'

# Read CSV
df = pd.read_csv('benchmarkv2.csv')

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

print("Checking all Google Scholar links...\n")
print("=" * 100)

results = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking links"):
    benchmark_name = row['Benchmark Name']
    link = row['Google Scholar Link']

    status = check_link(link, headers)
    results.append({
        'Benchmark': benchmark_name,
        'Link': link[:80] + '...' if len(link) > 80 else link,
        'Status': status
    })

    # Small delay to avoid rate limiting
    time.sleep(1)

print("\n" + "=" * 100)
print("\nRESULTS:\n")

# Group by status
valid_with_citations = [r for r in results if 'Has citations' in r['Status']]
valid_no_citations = [r for r in results if 'No "Cited by"' in r['Status']]
errors = [r for r in results if not r['Status'].startswith('Valid')]

print(f"✅ Valid with citations: {len(valid_with_citations)}")
print(f"⚠️  Valid but no 'Cited by' found: {len(valid_no_citations)}")
print(f"❌ Errors/Issues: {len(errors)}\n")

if valid_with_citations:
    print("\n" + "=" * 100)
    print("VALID LINKS WITH CITATIONS:")
    print("=" * 100)
    for r in valid_with_citations:
        print(f"✅ {r['Benchmark']}")
        print(f"   {r['Link']}")
        print()

if valid_no_citations:
    print("\n" + "=" * 100)
    print("VALID LINKS BUT NO 'CITED BY' FOUND:")
    print("=" * 100)
    for r in valid_no_citations:
        print(f"⚠️  {r['Benchmark']}")
        print(f"   {r['Link']}")
        print()

if errors:
    print("\n" + "=" * 100)
    print("ERRORS/ISSUES:")
    print("=" * 100)
    for r in errors:
        print(f"❌ {r['Benchmark']}")
        print(f"   Status: {r['Status']}")
        print(f"   {r['Link']}")
        print()

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('link_check_results.csv', index=False)
print("=" * 100)
print(f"\nResults saved to link_check_results.csv")
