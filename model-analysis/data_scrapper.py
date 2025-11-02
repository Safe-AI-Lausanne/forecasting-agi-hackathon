#!/usr/bin/env python3
"""
HydroX Leaderboard Scraper - Complete Fixed Version
Fetches data directly from the JSON API endpoints used by the leaderboard
"""

import requests
import json
import csv
import os

def fetch_attack_method_data(output_file='hydrox_attack_methods.csv'):
    """
    Fetch attack method data from the API and save to CSV
    """
    
    api_url = "https://storage.googleapis.com/hydrox-leaderboard/v0.4/attackMethod.json"
    
    print("=" * 70)
    print("HydroX Attack Methods Leaderboard Scraper")
    print("=" * 70)
    print(f"\nFetching data from API: {api_url}")
    
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Successfully retrieved data!")
        print(f"  Total models: {len(data['items'])}")
        
        # Extract all the data
        rows = []
        for index, item in enumerate(data['items']):
            row = {
                'Rank': index + 1,
                'Previous_Rank': item.get('previousRank', index + 1),
                'Model': item['model']['name'],
                'None_Score': round(item['none']['score'] * 100, 2),
                'None_Trials': item['none']['trials'],
                'None_Risks': item['none']['risks'],
                'ABJ_Score': round(item['abj']['score'] * 100, 2),
                'ABJ_Trials': item['abj']['trials'],
                'ABJ_Risks': item['abj']['risks'],
                'Adaptive_Score': round(item['adaptive']['score'] * 100, 2),
                'Adaptive_Trials': item['adaptive']['trials'],
                'Adaptive_Risks': item['adaptive']['risks'],
                'ArtPrompt_Score': round(item['artprompt']['score'] * 100, 2),
                'ArtPrompt_Trials': item['artprompt']['trials'],
                'ArtPrompt_Risks': item['artprompt']['risks'],
                'AutoDAN_Score': round(item['autodan']['score'] * 100, 2),
                'AutoDAN_Trials': item['autodan']['trials'],
                'AutoDAN_Risks': item['autodan']['risks'],
                'Cipher_Score': round(item['cipher']['score'] * 100, 2),
                'Cipher_Trials': item['cipher']['trials'],
                'Cipher_Risks': item['cipher']['risks'],
                'DAN_Score': round(item['dan']['score'] * 100, 2),
                'DAN_Trials': item['dan']['trials'],
                'DAN_Risks': item['dan']['risks'],
                'DeepInception_Score': round(item['deepInception']['score'] * 100, 2),
                'DeepInception_Trials': item['deepInception']['trials'],
                'DeepInception_Risks': item['deepInception']['risks'],
                'Developer_Score': round(item['developer']['score'] * 100, 2),
                'Developer_Trials': item['developer']['trials'],
                'Developer_Risks': item['developer']['risks'],
                'DRA_Score': round(item['dra']['score'] * 100, 2),
                'DRA_Trials': item['dra']['trials'],
                'DRA_Risks': item['dra']['risks'],
                'DrAttack_Score': round(item['drattack']['score'] * 100, 2),
                'DrAttack_Trials': item['drattack']['trials'],
                'DrAttack_Risks': item['drattack']['risks'],
                'GCG_Score': round(item['gcg']['score'] * 100, 2),
                'GCG_Trials': item['gcg']['trials'],
                'GCG_Risks': item['gcg']['risks'],
                'GPTFuzzer_Score': round(item['gptfuzzer']['score'] * 100, 2),
                'GPTFuzzer_Trials': item['gptfuzzer']['trials'],
                'GPTFuzzer_Risks': item['gptfuzzer']['risks'],
                'Grandmother_Score': round(item['grandmother']['score'] * 100, 2),
                'Grandmother_Trials': item['grandmother']['trials'],
                'Grandmother_Risks': item['grandmother']['risks'],
                'Masterkey_Score': round(item['masterkey']['score'] * 100, 2),
                'Masterkey_Trials': item['masterkey']['trials'],
                'Masterkey_Risks': item['masterkey']['risks'],
                'Multilingual_Score': round(item['multilingual']['score'] * 100, 2),
                'Multilingual_Trials': item['multilingual']['trials'],
                'Multilingual_Risks': item['multilingual']['risks'],
                'PAIR_Score': round(item['pair']['score'] * 100, 2),
                'PAIR_Trials': item['pair']['trials'],
                'PAIR_Risks': item['pair']['risks'],
                'PastTense_Score': round(item['pastTense']['score'] * 100, 2),
                'PastTense_Trials': item['pastTense']['trials'],
                'PastTense_Risks': item['pastTense']['risks'],
                'Psychology_Score': round(item['psychology']['score'] * 100, 2),
                'Psychology_Trials': item['psychology']['trials'],
                'Psychology_Risks': item['psychology']['risks'],
                'ReNeLLM_Score': round(item['renellm']['score'] * 100, 2),
                'ReNeLLM_Trials': item['renellm']['trials'],
                'ReNeLLM_Risks': item['renellm']['risks'],
                'TAP_Score': round(item['tap']['score'] * 100, 2),
                'TAP_Trials': item['tap']['trials'],
                'TAP_Risks': item['tap']['risks'],
                'Overall_Score': round(item['overall']['score'] * 100, 2),
                'Overall_Trials': item['overall']['trials'],
                'Overall_Risks': item['overall']['risks'],
            }
            rows.append(row)
        
        # Write to CSV
        output_dir = os.getcwd()
        csv_file = os.path.join(output_dir, output_file)
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\n✓ SUCCESS! Data saved to: {csv_file}")
        print(f"  Total rows: {len(rows)}")
        print(f"  Total columns: {len(rows[0].keys())}")
        
        # Show preview
        print("\n" + "=" * 70)
        print("PREVIEW - Top 5 Models:")
        print("=" * 70)
        for i, row in enumerate(rows[:5]):
            print(f"\n{i+1}. {row['Model']}")
            print(f"   Overall Score: {row['Overall_Score']}")
            print(f"   Overall Trials: {row['Overall_Trials']}, Risks: {row['Overall_Risks']}")
        
        return True, len(rows)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0


def fetch_model_domain_data(output_file='hydrox_model_domains.csv'):
    """
    Fetch model domain data from the API and save to CSV
    NOTE: The model.json structure is different - it doesn't have trials/risks for domains
    """
    
    api_url = "https://storage.googleapis.com/hydrox-leaderboard/v0.4/model.json"
    
    print("\n" + "=" * 70)
    print("HydroX Model Domains Leaderboard Scraper")
    print("=" * 70)
    print(f"\nFetching data from API: {api_url}")
    
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Successfully retrieved data!")
        print(f"  Total models: {len(data['items'])}")
        
        # First, let's inspect the structure of the first item
        if len(data['items']) > 0:
            first_item = data['items'][0]
            print(f"\nInspecting data structure...")
            print(f"  Available keys: {list(first_item.keys())}")
            
            # Check what's in each domain
            for domain in ['safetyDomain', 'privacyDomain', 'securityDomain', 'integrityDomain']:
                if domain in first_item:
                    print(f"  {domain} keys: {list(first_item[domain].keys())}")
        
        # Extract all the data
        rows = []
        for index, item in enumerate(data['items']):
            row = {
                'Rank': index + 1,
                'Previous_Rank': item.get('previousRank', index + 1),
                'Model': item['model']['name'],
                'Safety_Score': round(item['safetyDomain']['score'] * 100, 2),
                'Privacy_Score': round(item['privacyDomain']['score'] * 100, 2),
                'Security_Score': round(item['securityDomain']['score'] * 100, 2),
                'Integrity_Score': round(item['integrityDomain']['score'] * 100, 2),
                'High_Attack_Score': round(item['highAttack']['score'] * 100, 2),
                'Medium_Attack_Score': round(item['mediumAttack']['score'] * 100, 2),
                'Low_Attack_Score': round(item['lowAttack']['score'] * 100, 2),
                'Overall_Score': round(item['overall']['score'] * 100, 2),
            }
            rows.append(row)
        
        # Write to CSV
        output_dir = os.getcwd()
        csv_file = os.path.join(output_dir, output_file)
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\n✓ SUCCESS! Data saved to: {csv_file}")
        print(f"  Total rows: {len(rows)}")
        print(f"  Total columns: {len(rows[0].keys())}")
        
        # Show preview
        print("\n" + "=" * 70)
        print("PREVIEW - Top 5 Models:")
        print("=" * 70)
        for i, row in enumerate(rows[:5]):
            print(f"\n{i+1}. {row['Model']}")
            print(f"   Overall Score: {row['Overall_Score']}")
            print(f"   Safety: {row['Safety_Score']}, Privacy: {row['Privacy_Score']}")
            print(f"   Security: {row['Security_Score']}, Integrity: {row['Integrity_Score']}")
        
        return True, len(rows)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0


def create_summary_report(attack_success, attack_rows, domain_success, domain_rows):
    """Create a summary report of the scraping results"""
    
    report_file = 'SCRAPING_REPORT.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("HYDROX LEADERBOARD SCRAPING REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")
        
        if attack_success:
            f.write(f"✓ Attack Methods Data: SUCCESS\n")
            f.write(f"  - File: hydrox_attack_methods.csv\n")
            f.write(f"  - Total Models: {attack_rows}\n")
            f.write(f"  - Columns: 69 (Rank, Previous_Rank, Model, + 22 attack methods × 3 metrics)\n")
            f.write(f"  - Metrics per attack: Score, Trials, Risks\n")
            f.write(f"  - Attack methods: None, ABJ, Adaptive, ArtPrompt, AutoDAN, Cipher, DAN,\n")
            f.write(f"    DeepInception, Developer, DRA, DrAttack, GCG, GPTFuzzer, Grandmother,\n")
            f.write(f"    Masterkey, Multilingual, PAIR, PastTense, Psychology, ReNeLLM, TAP\n\n")
        else:
            f.write(f"✗ Attack Methods Data: FAILED\n\n")
        
        if domain_success:
            f.write(f"✓ Model Domains Data: SUCCESS\n")
            f.write(f"  - File: hydrox_model_domains.csv\n")
            f.write(f"  - Total Models: {domain_rows}\n")
            f.write(f"  - Columns: 11 (Rank, Previous_Rank, Model, + 8 domain/attack scores)\n")
            f.write(f"  - Domain Scores: Safety, Privacy, Security, Integrity\n")
            f.write(f"  - Attack Strength Scores: High, Medium, Low\n")
            f.write(f"  - Overall Score\n\n")
        else:
            f.write(f"✗ Model Domains Data: FAILED\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("DATA DESCRIPTION\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("ATTACK METHODS CSV (hydrox_attack_methods.csv)\n")
        f.write("-" * 70 + "\n")
        f.write("This file contains detailed scores for 20+ different attack methods used\n")
        f.write("to test each AI model's safety. For each attack method, three metrics are\n")
        f.write("provided:\n\n")
        f.write("  - Score: Safety score (0-100, higher is safer)\n")
        f.write("  - Trials: Number of test attempts\n")
        f.write("  - Risks: Number of successful attacks (harmful responses)\n\n")
        
        f.write("Attack Method Descriptions:\n")
        f.write("  • None: Baseline with no attack (control)\n")
        f.write("  • ABJ: Exploits analyzing/reasoning capabilities\n")
        f.write("  • Adaptive: Uses adaptive prompt templates\n")
        f.write("  • ArtPrompt: Exploits inability to interpret ASCII art\n")
        f.write("  • AutoDAN: Automated creation of stealthy jailbreak prompts\n")
        f.write("  • Cipher: Uses role play to exploit hidden capabilities\n")
        f.write("  • DAN: 'Do Anything Now' prompt to bypass filters\n")
        f.write("  • DeepInception: Authority-based personified scenarios\n")
        f.write("  • Developer: Simulates Developer Mode\n")
        f.write("  • DRA: Cloaks harmful instructions for reconstruction\n")
        f.write("  • DrAttack: Decomposes and reconstructs prompts\n")
        f.write("  • GCG: Generates adversarial suffixes using gradients\n")
        f.write("  • GPTFuzzer: Automated jailbreak template generation\n")
        f.write("  • Grandmother: Family member roleplay techniques\n")
        f.write("  • Masterkey: Exploits defenses using time-based analysis\n")
        f.write("  • Multilingual: Prompts in non-English languages\n")
        f.write("  • PAIR: Iterative prompt generation using attacker LLM\n")
        f.write("  • PastTense: Reformulates requests in past tense\n")
        f.write("  • Psychology: Uses psychological influence techniques\n")
        f.write("  • ReNeLLM: Automatic generation via rewriting and nesting\n")
        f.write("  • TAP: Tree-of-thoughts reasoning and pruning\n\n")
        
        f.write("MODEL DOMAINS CSV (hydrox_model_domains.csv)\n")
        f.write("-" * 70 + "\n")
        f.write("This file contains aggregated safety scores across four main domains and\n")
        f.write("three attack strength categories:\n\n")
        
        f.write("Domain Scores (what type of risk):\n")
        f.write("  • Safety: Risks promoting danger (crime, hate speech)\n")
        f.write("  • Privacy: Data breaching risks (data sharing, membership inference)\n")
        f.write("  • Security: Behavioral risks (roleplay, prompt injection)\n")
        f.write("  • Integrity: Ethics-related risks (copyright, fraud)\n\n")
        
        f.write("Attack Strength Scores (how sophisticated the attack):\n")
        f.write("  • High: Advanced attacks (ABJ, Adaptive, ReNeLLM)\n")
        f.write("  • Medium: Moderately sophisticated (Cipher, DAN, GCG)\n")
        f.write("  • Low: Basic techniques (Roleplay, Multilingual, Psychology)\n\n")
        
        f.write("Overall Score:\n")
        f.write("  Weighted average across all domains and attack strengths, with more\n")
        f.write("  advanced attacks having greater influence.\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("USAGE NOTES\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("• All scores are on a 0-100 scale where 100 = most safe\n")
        f.write("• Scores are calculated as: ((Trials - Risks) / Trials) × 100\n")
        f.write("• Models are ranked by Overall Score (highest to lowest)\n")
        f.write("• Previous_Rank shows the model's rank in the previous evaluation\n")
        f.write("• You can use these CSV files in Excel, Python (pandas), R, or any\n")
        f.write("  data analysis tool\n\n")
        
        f.write("Example Python usage:\n")
        f.write("  import pandas as pd\n")
        f.write("  df = pd.read_csv('hydrox_attack_methods.csv')\n")
        f.write("  print(df.head())\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"\n✓ Summary report created: {report_file}")
    return report_file


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("HYDROX LEADERBOARD DATA EXTRACTOR")
    print("=" * 70)
    print("\nThis script will download ALL data from both leaderboard tabs:")
    print("  1. Attack Methods (detailed scores for 20+ attack types)")
    print("  2. Model Domains (safety, privacy, security, integrity scores)")
    print("\n" + "=" * 70)
    
    # Fetch both datasets
    attack_success, attack_rows = fetch_attack_method_data()
    domain_success, domain_rows = fetch_model_domain_data()
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    if attack_success:
        print(f"✓ Attack Methods data: hydrox_attack_methods.csv ({attack_rows} models)")
    else:
        print("✗ Attack Methods data: FAILED")
    
    if domain_success:
        print(f"✓ Model Domains data: hydrox_model_domains.csv ({domain_rows} models)")
    else:
        print("✗ Model Domains data: FAILED")
    
    # Create comprehensive report
    report_file = create_summary_report(attack_success, attack_rows, domain_success, domain_rows)
    print(f"✓ Detailed report: {report_file}")
    
    print("\n" + "=" * 70)
    print("Done! All files saved to current directory.")
    print("=" * 70)