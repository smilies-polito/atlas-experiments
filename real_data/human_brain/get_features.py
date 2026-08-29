import os
import pandas as pd
import numpy as np
import requests

def ensembl_to_gene_name(gene_ids, batch_size=1000):
    """Map Ensembl gene IDs to gene names using Ensembl REST API."""

    url = "https://rest.ensembl.org/lookup/id"
    annotations = {}

    for start in range(0, len(gene_ids), batch_size):
        batch = gene_ids[start:start + batch_size]
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={"ids": list(batch)},
        )

        response.raise_for_status()
        annotations.update(response.json())

    return pd.DataFrame(
        [
            {
                "ensembl_gene_id": gene_id,
                "external_gene_name": (
                    annotation.get("display_name") if annotation is not None else None
                ),
            }
            for gene_id, annotation in annotations.items()
        ]
    )


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    data_dir = os.path.join(os.getcwd(), "data", "human_brain")
    rna = pd.read_csv(os.path.join(data_dir, "GSE162170_multiome_rna_counts.tsv.gz"), sep = "\t").T
    gene_ids = rna.columns.tolist()
    print(f"NUMBER OF GENES {len(gene_ids)}")

    annotations = ensembl_to_gene_name(gene_ids)
    print(f"NUMBER OF QUERIES {len(annotations)}")

    annotations = annotations.dropna(subset=["external_gene_name"])
    print(f"NUMBER OF NON NA QUERIES {len(annotations)}")

    intersected = set(gene_ids).intersection(
        set(annotations["ensembl_gene_id"])
    )
    print(f"INSERSECTION {len(intersected)}")

    annotations.to_csv(os.path.join(data_dir,
                                    "gene_annotations.tsv"),
                        header = True,
                        index = False,
                        sep = "\t")

