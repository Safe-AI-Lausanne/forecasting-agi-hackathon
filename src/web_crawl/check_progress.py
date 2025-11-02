import pandas as pd

df = pd.read_csv('citations_classified.csv')
classified = df['paper_type'].notna().sum()
total = len(df)
pct = (classified/total)*100

print(f'Progress: {classified}/{total} papers classified ({pct:.1f}%)')
print(f'Remaining: {total - classified} papers')
print()

counts = df['paper_type'].value_counts()
print('Current classifications:')
for cat, count in counts.items():
    print(f'  {cat}: {count}')
