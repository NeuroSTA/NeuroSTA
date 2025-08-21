import pandas as pd

def merge_results():
    ling_df = pd.read_excel('../results/linguistics_results.xlsx')
    graph_df = pd.read_excel('../results/graph_results.xlsx')

    merged_df = pd.merge(ling_df, graph_df, on='Filename', how='outer')
    merged_df.to_excel('../results/combined_analysis.xlsx', index=False)
    print("Results merged. Combined analysis saved to ../results/combined_analysis.xlsx")

if __name__ == "__main__":
    merge_results()
