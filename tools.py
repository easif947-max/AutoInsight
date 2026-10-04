import pandas as pd
import numpy as np
from crewai.tools import tool

SYNONYM_MAP = {
    "cat collar": ["kitten collar", "feline neckband", "kitty strap", "breakaway collar", "small cat collar"],
    "kitten collar": ["cat collar", "small pet collar", "kitten neckband", "safety cat collar"]
}

@tool("Dataset Quantitative Profiler Engine")
def profile_csv_dataset(csv_filepath: str) -> str:
    """Calculates total revenue, mean units sold, average price, and key dataset metrics."""
    df = pd.read_csv(csv_filepath)
    
    # Calculate revenue if present
    if 'Units_Sold' in df.columns and 'Unit_Price' in df.columns:
        df['Total_Revenue'] = df['Units_Sold'] * df['Unit_Price']
        total_revenue = df['Total_Revenue'].sum()
    else:
        total_revenue = 0.0

    total_units = df['Units_Sold'].sum() if 'Units_Sold' in df.columns else 0
    avg_price = df['Unit_Price'].mean() if 'Unit_Price' in df.columns else 0.0
    
    summary = (
        f"Total Calculated Revenue: ${total_revenue:,.2f}\n"
        f"Total Units Sold: {total_units:,}\n"
        f"Average Unit Price: ${avg_price:.2f}\n"
        f"Total Dataset Rows: {len(df)}\n"
        f"Columns Detected: {list(df.columns)}"
    )
    return summary

@tool("Keyword and Synonym Filter Engine")
def filter_top_keywords(csv_filepath: str, target_keyword: str) -> str:
    """Expands target keywords using semantic synonyms and ranks top keywords by Search Volume."""
    df = pd.read_csv(csv_filepath)
    
    base_kw = target_keyword.lower().strip()
    synonyms = SYNONYM_MAP.get(base_kw, [base_kw])
    all_terms = list(set([base_kw] + synonyms))
    
    pattern = '|'.join(all_terms)
    
    if 'Keywords_Driven_Sales' in df.columns:
        filtered_df = df[df['Keywords_Driven_Sales'].astype(str).str.contains(pattern, case=False, na=False)]
    else:
        filtered_df = df
        
    sorted_df = filtered_df.sort_values(by='Search_Volume', ascending=False) if 'Search_Volume' in df.columns else filtered_df
    
    result = f"Matched Synonyms: {synonyms}\n\n"
    cols = [c for c in ['Product', 'Keywords_Driven_Sales', 'Search_Volume', 'Units_Sold'] if c in sorted_df.columns]
    result += sorted_df[cols].to_string(index=False)
    return result
