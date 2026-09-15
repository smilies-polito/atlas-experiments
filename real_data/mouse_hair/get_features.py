import os
import numpy as np 
import pandas as pd
import urllib.request, gzip

if __name__=="__main__":
    seed = 42
    working_dir = os.getcwd()
    np.random.seed(seed)

    data_path = os.path.join(working_dir, "data", "mouse_hair")
    feature_path = os.path.join(data_path, "all_features.tsv")

    url = "http://ftp.ensembl.org/pub/release-102/gtf/mus_musculus/Mus_musculus.GRCm38.102.gtf.gz"
    gz = "Mus_musculus.GRCm38.102.gtf.gz"
    urllib.request.urlretrieve(url, gz)
   
    rows = []
    with gzip.open(gz, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if c[2] != "gene":
                continue
            chrom, start, end, strand, attrs = c[0], int(c[3]), int(c[4]), c[6], c[8]
            name = ""
            for field in attrs.split(";"):
                field = field.strip()
                if field.startswith("gene_name"):
                    name = field.split('"')[1]
            rows.append((name, chrom, start, end, 1 if strand == "+" else -1))

    res = pd.DataFrame(rows, columns=["external_gene_name", "chromosome_name",
                                  "start_position", "end_position", "strand"])

    res.to_csv(feature_path, sep="\t", index=False)

