import pandas as pd
import re

# Read the exported CSV
df = pd.read_csv('/Users/jerry/hidden-connections/1/data/real_responses.csv')

# Print columns to understand structure
print("Original columns:", df.columns.tolist())
print(f"Number of rows: {len(df)}")

# The columns are the full question text, rename them
df.columns = ['timestamp', 'q1_safe_place', 'q2_stress', 'q3_understood', 'q4_free_day', 'q5_one_word']

# Generate IDs
df['id'] = [f'p_{str(i+1).zfill(3)}' for i in range(len(df))]

# Extract one-word self-description for nickname
def extract_nickname(text):
    if pd.isna(text) or text.strip() == '':
        return ''
    # Try to extract the first word/phrase before explanation
    text = str(text).strip()
    # Handle formats like "Word - explanation" or ""Word" - explanation" or just "Word"
    match = re.match(r'^["\']?([A-Za-z\u4e00-\u9fff]+)["\']?', text)
    if match:
        word = match.group(1).lower()
        # Clean up and use as nickname
        return word.replace(' ', '_')[:20]
    return ''

df['nickname'] = df['q5_one_word'].apply(extract_nickname)

# Set default demographic values (since real data doesn't have these)
df['q6_decision_style'] = 'Depends'
df['q7_social_energy'] = 'Depends'  
df['q8_region'] = 'Unknown'

# Reorder columns to match expected format
output_df = df[['id', 'q1_safe_place', 'q2_stress', 'q3_understood', 'q4_free_day', 'q5_one_word', 
                'q6_decision_style', 'q7_social_energy', 'q8_region', 'nickname']]

# Fill any NaN values with empty strings
output_df = output_df.fillna('')

# Save to CSV
output_df.to_csv('/Users/jerry/hidden-connections/1/data/responses_projective.csv', index=False)

print(f"\nConverted {len(output_df)} responses")
print("\nFirst few rows:")
print(output_df[['id', 'nickname', 'q5_one_word']].head(10).to_string())
