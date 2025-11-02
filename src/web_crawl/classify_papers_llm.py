import pandas as pd
import json
import time
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def classify_paper_with_llm(title, abstract):
    """
    Use LLM to classify a paper based on its title and abstract.
    Returns: 'Survey', 'Benchmark', 'Methodology', or 'Other'
    """
    # Handle NaN values
    if pd.isna(abstract):
        abstract = "No abstract available"
    if pd.isna(title):
        title = "No title available"

    prompt = f"""Classify this research paper into one of these categories:
- Survey: Literature reviews, surveys, overviews of a research area
- Benchmark: Papers introducing new datasets, benchmarks, evaluation frameworks, or test suites
- Methodology: Papers introducing new methods, models, techniques, or algorithms
- Other: Papers that don't fit the above categories (applications, case studies, etc.)

Paper Title: {title}

Paper Abstract: {abstract}

Respond with ONLY ONE WORD: Survey, Benchmark, Methodology, or Other"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using cheaper model for classification
            messages=[
                {"role": "system", "content": "You are a research paper classifier. Respond with only one word."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )

        classification = response.choices[0].message.content.strip()

        # Validate response
        valid_categories = ['Survey', 'Benchmark', 'Methodology', 'Other']
        if classification not in valid_categories:
            # Try to match partial response
            for category in valid_categories:
                if category.lower() in classification.lower():
                    return category
            return 'Other'

        return classification

    except Exception as e:
        print(f"Error classifying paper: {e}")
        return 'Error'

def main():
    # Read citations CSV
    df = pd.read_csv('citations.csv')

    print(f"Total papers: {len(df)}")
    print("\nClassifying papers using LLM...")
    print("This may take a while...\n")

    # Check if we have a partially classified file to resume from
    output_file = 'citations_classified.csv'
    if os.path.exists(output_file):
        print(f"Found existing classified file. Resuming...")
        df_existing = pd.read_csv(output_file)
        if 'paper_type' in df_existing.columns:
            df = df_existing
            already_classified = df['paper_type'].notna().sum()
            print(f"Already classified: {already_classified}/{len(df)} papers\n")
    else:
        df['paper_type'] = None

    # Classify papers that haven't been classified yet
    for idx, row in df.iterrows():
        if pd.notna(row.get('paper_type')):
            continue  # Skip already classified papers

        print(f"[{idx+1}/{len(df)}] Classifying: {row['title'][:60]}...")

        classification = classify_paper_with_llm(row['title'], row['abstract'])
        df.at[idx, 'paper_type'] = classification

        # Save progress every 10 papers
        if (idx + 1) % 10 == 0:
            df.to_csv(output_file, index=False)
            print(f"  💾 Progress saved ({idx+1}/{len(df)} papers)")

        # Rate limiting - sleep to avoid API limits
        time.sleep(0.5)

    # Final save
    df.to_csv(output_file, index=False)

    # Print statistics
    print("\n" + "="*60)
    print("Classification Results:")
    print("="*60)

    type_counts = df['paper_type'].value_counts()
    for paper_type, count in type_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{paper_type:12s}: {count:4d} ({percentage:5.1f}%)")

    print(f"{'Total':12s}: {len(df):4d} (100.0%)")

    print(f"\n✅ Saved classified data to {output_file}")

    # Create filtered files
    surveys_df = df[df['paper_type'] == 'Survey']
    benchmarks_df = df[df['paper_type'] == 'Benchmark']
    methodology_df = df[df['paper_type'] == 'Methodology']

    surveys_df.to_csv('citations_surveys.csv', index=False)
    benchmarks_df.to_csv('citations_benchmarks.csv', index=False)
    methodology_df.to_csv('citations_methodology.csv', index=False)

    print(f"\n✅ Saved {len(surveys_df)} surveys to citations_surveys.csv")
    print(f"✅ Saved {len(benchmarks_df)} benchmarks to citations_benchmarks.csv")
    print(f"✅ Saved {len(methodology_df)} methodology papers to citations_methodology.csv")

    # Show some examples
    print("\n" + "="*60)
    print("Survey Examples (first 5):")
    print("="*60)
    for idx, row in surveys_df.head(5).iterrows():
        print(f"\n• {row['title']}")
        print(f"  Source: {row['benchmark_name']}")

    print("\n" + "="*60)
    print("Benchmark Examples (first 5):")
    print("="*60)
    for idx, row in benchmarks_df.head(5).iterrows():
        print(f"\n• {row['title']}")
        print(f"  Source: {row['benchmark_name']}")

    print("\n" + "="*60)
    print("Methodology Examples (first 5):")
    print("="*60)
    for idx, row in methodology_df.head(5).iterrows():
        print(f"\n• {row['title']}")
        print(f"  Source: {row['benchmark_name']}")

if __name__ == '__main__':
    main()
