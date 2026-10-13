import pandas as pd


def process_csv(input_path, output_path):
    df = pd.read_csv(input_path)

    # Truncate 'question' at the first '?', keeping the '?'; leave it alone if absent.
    df['question'] = df['question'].apply(
        lambda x: x.split('?')[0] + '?' if isinstance(x, str) and '?' in x else x
    )

    # Strip a leading '(A) ' / '(B) ' option prefix.
    if 'A' in df.columns:
        df['A'] = df['A'].str.replace(r'^\([A-Za-z]\)\s*', '', regex=True)
    
    if 'B' in df.columns:
        df['B'] = df['B'].str.replace(r'^\([A-Za-z]\)\s*', '', regex=True)
        
    df.to_csv(output_path, index=False)
    print(f"Processed CSV saved to: {output_path}")

# Example usage:
process_csv('/home/yosef/ws/thesis/data/persona-1/corrigible-less/test_infer.csv', '/home/yosef/ws/thesis/data/persona-1/corrigible-less/test_infer.csv')
