import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_GT_CSV = '/Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-detection/main notebooks/ground_truth.csv'
DEFAULT_RAGA_DB = '/Users/saumyamishra/Desktop/intern/summer25/RagaDetection/raga-detection/main notebooks/raga_list_final.csv'

def load_raga_masks(db_path):
    """Load raga names from database."""
    df = pd.read_csv(db_path)
    raga_names = set()
    
    for _, row in df.iterrows():
        raw_name = row.get('raga') or row.get('names')
        if not isinstance(raw_name, str): continue
        
        # Clean up the string - remove all quote variations
        s = raw_name.strip()
        # Remove outer quotes and brackets
        s = s.strip('"').strip("'").strip('"').strip('"')
        if s.startswith('[') and s.endswith(']'): 
            s = s[1:-1]
        # Remove any remaining quotes
        s = s.strip('"').strip("'").strip('"').strip('"')
        
        # Split by comma
        parts = s.split(',')
        for p in parts:
            clean = p.strip().strip('"').strip("'").strip('"').strip('"')
            if clean:
                raga_names.add(clean)
                raga_names.add(clean.lower())
    
    return raga_names

def fuzzy_match_raga(gt_name, raga_names):
    """
    Try to find a matching raga name using fuzzy matching.
    Returns: (match_type, matched_name)
    - 'exact': Direct match found
    - 'case': Case-insensitive match found
    - 'spelling': Spelling variant match (i/ee, ti/hoti)
    - 'suffix': Suffix removal match (e.g., "Darbari Kanada" -> "Darbari")
    - 'substring': Substring match
    - None: No match found
    """
    # Direct match
    if gt_name in raga_names:
        return ('exact', gt_name)
    
    # Case insensitive
    if gt_name.lower() in raga_names:
        # Find the actual DB name
        for name in raga_names:
            if name.lower() == gt_name.lower() and name != gt_name:
                return ('case', name)
    
    # Common spelling variants
    variants = []
    
    # i/ee variations: Shri/Shree, Shruti/Shreeti
    if 'i' in gt_name:
        variant = gt_name.replace('i', 'ee')
        if variant in raga_names:
            return ('spelling_i_to_ee', variant)
        variants.append(variant)
    
    if 'ee' in gt_name:
        variant = gt_name.replace('ee', 'i')
        if variant in raga_names:
            return ('spelling_ee_to_i', variant)
        variants.append(variant)
    
    # ti/hoti variations: Jhinjoti/Jhinjhoti
    if gt_name.endswith('ti'):
        variant = gt_name[:-2] + 'hoti'
        if variant in raga_names:
            return ('spelling_ti_to_hoti', variant)
        variants.append(variant)
    
    if gt_name.endswith('hoti'):
        variant = gt_name[:-4] + 'ti'
        if variant in raga_names:
            return ('spelling_hoti_to_ti', variant)
        variants.append(variant)
    
    # Check case-insensitive variants
    for variant in variants:
        if variant.lower() in raga_names:
            for name in raga_names:
                if name.lower() == variant.lower():
                    return ('spelling_case', name)
    
    # Try removing common suffixes
    for suffix in [' Kanada', ' Kalyan', ' Sarang', ' Malhar', ' Bahar', ' Kafi']:
        if gt_name.endswith(suffix):
            base = gt_name[:-len(suffix)].strip()
            if base in raga_names:
                return ('suffix_removal', base)
            if base.lower() in raga_names:
                for name in raga_names:
                    if name.lower() == base.lower():
                        return ('suffix_removal_case', name)
    
    # Try partial matching
    gt_lower = gt_name.lower()
    for db_name in raga_names:
        db_lower = db_name.lower()
        # Only match if DB name is substring AND it's not just a partial word match
        # E.g., "Todi" should NOT match "Gurjari Todi" if "Gurjari Todi" exists in DB
        if db_lower in gt_lower and len(db_lower) >= 4 and db_lower != gt_lower:
            # Check if the GT name itself is in the database (exact match should have been caught earlier)
            # This means we should not do substring matching
            continue
    
    return (None, None)

def main():
    print("Analyzing Raga Name Mismatches...")
    print("="*70)
    
    # Load data
    gt_df = pd.read_csv(DEFAULT_GT_CSV)
    raga_names = load_raga_masks(DEFAULT_RAGA_DB)
    
    print(f"Ground Truth CSV: {len(gt_df)} entries")
    print(f"Database: {len(raga_names)} raga name variants")
    print()
    
    # Analyze matches
    match_stats = defaultdict(list)
    unique_gt_ragas = gt_df['Raga'].str.strip().unique()
    
    for gt_raga in sorted(unique_gt_ragas):
        match_type, matched_name = fuzzy_match_raga(gt_raga, raga_names)
        match_stats[match_type].append((gt_raga, matched_name))
    
    # Print results
    print("="*70)
    print("EXACT MATCHES")
    print("="*70)
    if 'exact' in match_stats:
        print(f"{len(match_stats['exact'])} ragas have exact matches")
    print()
    
    # Non-exact matches requiring fuzzy matching
    print("="*70)
    print("RAGAS REQUIRING FUZZY MATCHING (RENAME THESE IN GROUND TRUTH)")
    print("="*70)
    print()
    
    fuzzy_types = [
        ('case', 'Case-Insensitive Matches'),
        ('spelling_i_to_ee', 'Spelling: i → ee'),
        ('spelling_ee_to_i', 'Spelling: ee → i'),
        ('spelling_ti_to_hoti', 'Spelling: ti → hoti'),
        ('spelling_hoti_to_ti', 'Spelling: hoti → ti'),
        ('spelling_case', 'Spelling + Case'),
        ('suffix_removal', 'Suffix Removal'),
        ('suffix_removal_case', 'Suffix Removal + Case'),
        ('substring', 'Substring Match'),
    ]
    
    rename_suggestions = []
    
    for match_type, description in fuzzy_types:
        if match_type in match_stats and match_stats[match_type]:
            print(f"\n{description}:")
            print("-" * 70)
            for gt_name, db_name in sorted(match_stats[match_type]):
                count = len(gt_df[gt_df['Raga'].str.strip() == gt_name])
                print(f"  '{gt_name}' → '{db_name}' ({count} occurrences)")
                rename_suggestions.append({
                    'old_name': gt_name,
                    'new_name': db_name,
                    'count': count,
                    'match_type': description
                })
    
    # Not found
    if None in match_stats and match_stats[None]:
        print("\n" + "="*70)
        print("NOT FOUND IN DATABASE (ADD TO DATABASE OR CHECK SPELLING)")
        print("="*70)
        for gt_name, _ in sorted(match_stats[None]):
            count = len(gt_df[gt_df['Raga'].str.strip() == gt_name])
            print(f"  '{gt_name}' ({count} occurrences)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    total_exact = len(match_stats.get('exact', []))
    total_fuzzy = sum(len(v) for k, v in match_stats.items() if k not in ['exact', None])
    total_not_found = len(match_stats.get(None, []))
    
    print(f"Exact matches:     {total_exact}")
    print(f"Fuzzy matches:     {total_fuzzy} (SHOULD BE RENAMED)")
    print(f"Not found:         {total_not_found}")
    print(f"Total unique ragas: {len(unique_gt_ragas)}")
    
    # Save CSV with rename suggestions
    if rename_suggestions:
        rename_df = pd.DataFrame(rename_suggestions)
        output_csv = 'raga_rename_suggestions.csv'
        rename_df.to_csv(output_csv, index=False)
        print(f"\nSaved rename suggestions to: {output_csv}")
        print("\nYou can use this CSV to batch rename ragas in your ground truth file.")

if __name__ == '__main__':
    main()
