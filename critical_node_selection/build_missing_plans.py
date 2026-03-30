from pathlib import Path

import pandas as pd


OBSERVATION_RATES = [0.333, 0.20, 0.08]
SYSTEMS = ["IEEE33", "IEEE37"]


def main():
    base_dir = Path(__file__).resolve().parent
    for system in SYSTEMS:
        ranking = pd.read_csv(base_dir / f"{system}_ranking.csv").sort_values("rank")
        n = len(ranking)
        ranked_nodes = ranking["node"].astype(int).tolist()
        rows = []
        for rate in OBSERVATION_RATES:
            observable_nodes = max(1, int(round(rate * n)))
            missing_nodes = n - observable_nodes
            rows.append(
                {
                    "system": system,
                    "observation_rate": rate,
                    "num_nodes": n,
                    "observable_nodes": observable_nodes,
                    "missing_nodes": missing_nodes,
                    "deleted_nodes_in_order": " ".join(map(str, ranked_nodes[:missing_nodes])),
                    "kept_nodes": " ".join(map(str, ranked_nodes[missing_nodes:])),
                }
            )
        pd.DataFrame(rows).to_csv(base_dir / f"{system}_missing_plans.csv", index=False)


if __name__ == "__main__":
    main()
