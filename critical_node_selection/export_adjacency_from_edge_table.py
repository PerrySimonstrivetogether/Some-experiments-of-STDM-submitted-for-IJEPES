from pathlib import Path

import numpy as np
import pandas as pd


def build_physical_adjacency(edge_table: pd.DataFrame, group_size: int = 1) -> np.ndarray:
    max_index = int(max(edge_table["from_bus"].max(), edge_table["to_bus"].max()))
    num_physical_nodes = (max_index + 1) // group_size
    adj = np.eye(num_physical_nodes, dtype=int)
    for _, row in edge_table.iterrows():
        if "closed line" in edge_table.columns and int(row["closed line"]) != 1:
            continue
        i = int(row["from_bus"]) // group_size
        j = int(row["to_bus"]) // group_size
        if i == j:
            continue
        adj[i, j] = 1
        adj[j, i] = 1
    return adj


def main():
    """
    Example usage:
      python export_adjacency_from_edge_table.py edge_param.csv IEEE37_adjacency.csv 3
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("edge_csv", type=str)
    parser.add_argument("output_csv", type=str)
    parser.add_argument("--group_size", type=int, default=1)
    args = parser.parse_args()

    edge_table = pd.read_csv(args.edge_csv)
    adj = build_physical_adjacency(edge_table, group_size=args.group_size)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(adj).to_csv(output_path, index=False, header=False)


if __name__ == "__main__":
    main()
