#!/usr/bin/env python3
import pickle
import pandas as pd
import argparse
import os

def read_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def create_table(data):
    rows = []
    # Each key in data is a state dimension (nx)
    for nx, filters in data.items():
        row = {'nx': nx}
        for filter_key, timing in filters.items():
            row[filter_key] = timing['mean_time']
        rows.append(row)
    df = pd.DataFrame(rows)
    df.sort_values('nx', inplace=True)
    df.set_index('nx', inplace=True)
    # Transpose so that filters become rows and nx values become columns
    df = df.transpose()
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Read scalability results and create a transposed table with only mean values."
    )
    parser.add_argument('--file', type=str, default="./results/estimator6/scalability_results_dim_normal_normal.pkl",
                        help="Path to the pickle file containing the scalability results.")
    parser.add_argument('--output', type=str, default="scalability_results_mean_table_transposed.csv",
                        help="Name of the output CSV file.")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist.")
        return

    data = read_data(args.file)
    df = create_table(data)
    print("Transposed Scalability Results Table (Mean Values Only):")
    print(df.to_string())
    df.to_csv(args.output)
    print(f"\nTable saved to {args.output}")

if __name__ == "__main__":
    main()
