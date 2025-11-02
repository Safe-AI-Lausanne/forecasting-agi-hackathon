import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import csv
import json
import re
from urllib.parse import urljoin, urlparse, parse_qs
import sys
import os
from pathlib import Path
from tqdm import tqdm

class GoogleScholarCitationScraper:
    def __init__(self, csv_file='benchmarkv2.csv', output_file='citations.csv', delay=3):
        """
        Initialize the scraper.

        Args:
            csv_file: Path to the CSV file with benchmarks
            output_file: Path to save the citation data
            delay: Delay between requests in seconds (default 3)
        """
        self.csv_file = csv_file
        self.output_file = output_file
        self.delay = delay
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_page(self, url, retry=3):
        """Get a page with retry logic."""
        for attempt in range(retry):
            try:
                print(f"  Fetching: {url[:80]}...")
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    print(f"  Rate limited. Waiting 30 seconds...")
                    time.sleep(30)
                else:
                    print(f"  Error {response.status_code}, retrying...")
                    time.sleep(self.delay * 2)
            except Exception as e:
                print(f"  Request failed: {e}, retrying...")
                time.sleep(self.delay)
        return None

    def extract_citation_link(self, scholar_url):
        """Extract the 'Cited by' link from a Google Scholar search page."""
        # Add hl=en to force English language
        if 'hl=' not in scholar_url:
            separator = '&' if '?' in scholar_url else '?'
            scholar_url = f"{scholar_url}{separator}hl=en"

        response = self.get_page(scholar_url)
        if not response:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Look for "Cited by" link in various languages
        # English: "Cited by", German: "Zitiert von", French: "Cité par", Spanish: "Citado por"
        cited_by_pattern = re.compile(r'(Cited by|Zitiert von|Cité par|Citado por)[:\s]*(\d+)', re.IGNORECASE)

        # Find all links
        all_links = soup.find_all('a', string=cited_by_pattern)

        if all_links:
            cited_by_link = all_links[0]
            citation_url = urljoin('https://scholar.google.com', cited_by_link['href'])

            # Extract citation count
            cited_match = cited_by_pattern.search(cited_by_link.text)
            count = int(cited_match.group(2)) if cited_match else 0

            # Add hl=en to citation URL too
            if 'hl=' not in citation_url:
                separator = '&' if '?' in citation_url else '?'
                citation_url = f"{citation_url}{separator}hl=en"

            print(f"  Found {count} citations")
            return citation_url, count

        print("  No citations found")
        return None, 0

    def parse_citation_page(self, url, max_pages=5):
        """Parse a citation page and extract citing papers."""
        citations = []
        page_num = 0
        current_url = url

        while current_url and page_num < max_pages:
            print(f"  Parsing citation page {page_num + 1}...")
            response = self.get_page(current_url)
            if not response:
                break

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all search results
            results = soup.find_all('div', class_='gs_ri')

            for result in results:
                try:
                    citation = {}

                    # Extract PDF URL - look in title area first
                    pdf_url = ''
                    title_tag = result.find('h3', class_='gs_rt')

                    # Check for PDF link in title (before we remove spans)
                    if title_tag:
                        # Look for [PDF] link
                        pdf_link = title_tag.find('span', class_='gs_ctg2')
                        if pdf_link and pdf_link.find('a'):
                            pdf_url = pdf_link.find('a').get('href', '')

                    # If no PDF in title, check the gs_or_ggsm area (related articles/versions)
                    if not pdf_url:
                        # Look for any link with [PDF] text in the entire result
                        gs_ggs = result.find_next_sibling('div', class_='gs_or_ggsm')
                        if gs_ggs:
                            all_links = gs_ggs.find_all('a')
                            for link in all_links:
                                href = link.get('href', '')
                                # Check if it's a PDF link
                                if '.pdf' in href.lower() or 'pdf' in link.get_text().lower():
                                    pdf_url = href
                                    break

                    citation['pdf_url'] = pdf_url

                    # Title
                    if title_tag:
                        # Remove citation markers like [PDF], [HTML]
                        for span in title_tag.find_all('span', class_='gs_ct1'):
                            span.decompose()
                        for span in title_tag.find_all('span', class_='gs_ct2'):
                            span.decompose()

                        link_tag = title_tag.find('a')
                        if link_tag:
                            citation['title'] = link_tag.get_text(strip=True)
                            citation['url'] = link_tag.get('href', '')
                        else:
                            citation['title'] = title_tag.get_text(strip=True)
                            citation['url'] = ''

                    # Authors and publication info
                    author_tag = result.find('div', class_='gs_a')
                    if author_tag:
                        author_text = author_tag.get_text(strip=True)
                        parts = author_text.split(' - ')
                        if len(parts) >= 1:
                            citation['authors'] = parts[0].strip()
                        if len(parts) >= 2:
                            citation['publication'] = parts[1].strip()
                        if len(parts) >= 3:
                            citation['year'] = parts[2].strip()

                    # Abstract/snippet
                    abstract_tag = result.find('div', class_='gs_rs')
                    if abstract_tag:
                        citation['abstract'] = abstract_tag.get_text(strip=True)

                    # Cited by count (handle multiple languages)
                    cited_by_pattern = re.compile(r'(Cited by|Zitiert von|Cité par|Citado por)[:\s]*(\d+)', re.IGNORECASE)
                    cited_by = result.find('a', string=cited_by_pattern)
                    if cited_by:
                        cited_match = cited_by_pattern.search(cited_by.text)
                        if cited_match:
                            citation['cited_by_count'] = int(cited_match.group(2))

                    if citation.get('title'):
                        citations.append(citation)

                except Exception as e:
                    print(f"  Error parsing result: {e}")
                    continue

            # Find next page link
            next_button = soup.find('a', string=re.compile(r'Next'))
            if next_button and page_num < max_pages - 1:
                current_url = urljoin('https://scholar.google.com', next_button['href'])
                page_num += 1
                time.sleep(self.delay)  # Delay between pages
            else:
                current_url = None

        return citations

    def scrape_all_citations(self, max_citations_per_benchmark=100, limit_benchmarks=None, resume=True):
        """Scrape citations for all benchmarks in the CSV."""
        # Read the benchmark CSV
        df = pd.read_csv(self.csv_file)

        # Limit number of benchmarks if specified
        if limit_benchmarks:
            df = df.head(limit_benchmarks)
            print(f"Limiting to first {limit_benchmarks} benchmarks for testing\n")

        all_citations = []

        # Check if resuming from existing file
        already_scraped = set()
        if resume and os.path.exists(self.output_file):
            try:
                existing_df = pd.read_csv(self.output_file)
                if not existing_df.empty:
                    already_scraped = set(existing_df['benchmark_name'].unique())
                    all_citations = existing_df.to_dict('records')
                    print(f"📋 Resuming: Found {len(all_citations)} existing citations from {len(already_scraped)} benchmarks")
                    print(f"   Already scraped: {', '.join(sorted(already_scraped))}\n")
            except Exception as e:
                print(f"⚠️  Could not load existing citations: {e}\n")

        # Use tqdm for progress bar
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing benchmarks", unit="benchmark"):
            benchmark_name = row['Benchmark Name']
            paper_title = row['Paper Title']
            scholar_link = row['Google Scholar Link']

            # Skip if already scraped
            if benchmark_name in already_scraped:
                tqdm.write(f"\n[{idx + 1}/{len(df)}] ⏭️  Skipping {benchmark_name} (already scraped)")
                continue

            tqdm.write(f"\n[{idx + 1}/{len(df)}] Processing: {benchmark_name}")
            tqdm.write(f"  Paper: {paper_title}")

            # Get the citation link
            citation_data = self.extract_citation_link(scholar_link)
            if not citation_data:
                tqdm.write(f"  Skipping - could not find citation link")
                time.sleep(self.delay)
                continue

            citation_url, citation_count = citation_data

            if citation_count == 0:
                tqdm.write(f"  Skipping - no citations found")
                time.sleep(self.delay)
                continue

            # Calculate how many pages to fetch (each page has ~10 results)
            max_pages = min(20, (max_citations_per_benchmark + 9) // 10)

            time.sleep(self.delay)  # Delay before fetching citations

            # Get citing papers
            citations = self.parse_citation_page(citation_url, max_pages=max_pages)

            # Add benchmark info to each citation
            for citation in citations:
                citation['benchmark_name'] = benchmark_name
                citation['benchmark_paper'] = paper_title
                all_citations.append(citation)

            tqdm.write(f"  Collected {len(citations)} citations")

            # Save incrementally to avoid data loss
            self.save_citations(all_citations)

            # Delay between benchmarks
            time.sleep(self.delay)

        return all_citations

    def save_citations(self, citations):
        """Save citations to CSV file."""
        if not citations:
            return

        # Define columns - added pdf_url
        columns = ['benchmark_name', 'benchmark_paper', 'title', 'authors',
                   'publication', 'year', 'url', 'pdf_url', 'abstract', 'cited_by_count']

        # Write to CSV
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            for citation in citations:
                writer.writerow(citation)

        print(f"\n💾 Saved {len(citations)} citations to {self.output_file}")

        # Count PDFs found
        pdf_count = sum(1 for c in citations if c.get('pdf_url'))
        print(f"📄 Found {pdf_count} PDF links out of {len(citations)} papers ({pdf_count*100//len(citations) if citations else 0}%)")

        # Also save as JSON for better data structure
        json_file = self.output_file.replace('.csv', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(citations, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved to {json_file}")


def main():
    print("=" * 80)
    print("Google Scholar Citation Scraper")
    print("=" * 80)
    print("\nThis script will scrape papers that cited the benchmarks in benchmarkv2.csv")
    print("Note: Google Scholar has rate limiting. The script uses delays to avoid blocking.")
    print("The process may take a while depending on the number of citations.")
    print("\n" + "=" * 80 + "\n")

    # Check for limit argument
    limit_benchmarks = None
    if len(sys.argv) > 1:
        try:
            limit_benchmarks = int(sys.argv[1])
            print(f"🔧 Running in TEST mode - limiting to first {limit_benchmarks} benchmarks\n")
        except ValueError:
            print("Usage: python scrape_citations.py [limit_benchmarks]")
            sys.exit(1)

    scraper = GoogleScholarCitationScraper(
        csv_file='benchmarkv2.csv',
        output_file='citations.csv',
        delay=3  # 3 seconds between requests
    )

    try:
        citations = scraper.scrape_all_citations(
            max_citations_per_benchmark=200,
            limit_benchmarks=limit_benchmarks
        )
        print("\n" + "=" * 80)
        print(f"✅ Scraping complete! Total citations collected: {len(citations)}")
        print("=" * 80)
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrupted by user.")
        print("Partial data has been saved to citations.csv and citations.json")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
