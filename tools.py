import pandas as pd
import numpy as np
from crewai.tools import tool

# Expanded dictionary of semantic synonyms for pet/e-commerce products
SYNONYM_MAP = {
    "cat collar": [
        "cat collar", "kitten collar", "feline neckband", "kitty strap", 
        "breakaway collar", "small cat collar", "cat collars", "kitten collar with bell",
        "cat collar breakaway", "feline collar", "kitty collar", "breakaway cat collar"
    ],
    "kitten collar": [
        "kitten collar", "cat collar", "small pet collar", "kitten neckband", 
        "safety cat collar", "kitten collars", "reflective kitten collar"
    ]
}

def clean_numeric_column(series):
    """Converts formatted string numbers like '116,156', '$15.00', '>40,000' into numeric floats."""
    if series is None:
        return pd.Series(dtype=float)
    return (
        series.astype(str)
        .str.replace(r'[$,%>\s]', '', regex=True)
        .str.replace(',', '', regex=False)
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0)
    )

@tool("Dataset Quantitative Profiler Engine")
def profile_csv_dataset(csv_filepath: str) -> str:
    """Calculates quantitative metrics, total keyword sales, search volume, and key dataset indicators."""
    df = pd.read_csv(csv_filepath)
    
    kw_col = next((c for c in df.columns if 'keyword' in c.lower() and 'phrase' in c.lower()), None)
    sales_col = next((c for c in df.columns if 'sales' in c.lower()), None)
    sv_col = next((c for c in df.columns if 'search volume' in c.lower() and 'trend' not in c.lower()), None)
    
    sales_num = clean_numeric_column(df[sales_col]) if sales_col else pd.Series([0]*len(df))
    sv_num = clean_numeric_column(df[sv_col]) if sv_col else pd.Series([0]*len(df))
    
    total_sales = sales_num.sum()
    total_sv = sv_num.sum()
    avg_sv = sv_num.mean()
    
    summary = (
        f"Dataset Total Rows: {len(df):,}\n"
        f"Total Keyword Sales Captured: ${total_sales:,.2f}\n"
        f"Total Aggregate Search Volume: {total_sv:,.0f}\n"
        f"Average Search Volume per Keyword: {avg_sv:,.1f}\n"
        f"Detected Columns ({len(df.columns)}): {list(df.columns)}\n"
    )
    
    if kw_col and sales_col:
        df['temp_sales'] = sales_num
        top_kw = df.sort_values(by='temp_sales', ascending=False).head(5)
        summary += "\nTop 5 Keywords by Sales:\n"
        for _, row in top_kw.iterrows():
            summary += f"- {row[kw_col]}: ${row['temp_sales']:,.2f} sales\n"
            
    return summary

@tool("Keyword and Synonym Filter Engine")
def filter_top_keywords(csv_filepath: str, target_keyword: str) -> str:
    """Expands target keywords using semantic synonyms and ranks top keywords by Search Volume and Sales."""
    df = pd.read_csv(csv_filepath)
    
    kw_col = next((c for c in df.columns if 'keyword' in c.lower()), df.columns[0])
    sv_col = next((c for c in df.columns if 'search volume' in c.lower() and 'trend' not in c.lower()), None)
    sales_col = next((c for c in df.columns if 'sales' in c.lower()), None)
    
    base_kw = target_keyword.lower().strip()
    synonyms = SYNONYM_MAP.get(base_kw, [base_kw])
    all_terms = list(set([base_kw] + synonyms))
    
    pattern = '|'.join([t.replace(' ', r'\s+') for t in all_terms])
    filtered_df = df[df[kw_col].astype(str).str.contains(pattern, case=False, na=False)].copy()
    
    if filtered_df.empty:
        words = base_kw.split()
        pattern = '|'.join(words)
        filtered_df = df[df[kw_col].astype(str).str.contains(pattern, case=False, na=False)].copy()

    if sv_col:
        filtered_df['sv_clean'] = clean_numeric_column(filtered_df[sv_col])
        filtered_df = filtered_df.sort_values(by='sv_clean', ascending=False)
        
    result = f"Found {len(filtered_df)} keywords matching target '{target_keyword}' or synonyms ({synonyms}):\n\n"
    
    display_cols = [c for c in [kw_col, sv_col, sales_col, 'Search Volume Trend'] if c in filtered_df.columns]
    result += filtered_df[display_cols].head(25).to_string(index=False)
    return result
