# Prepare NCBI Gene data.
# $ cd ../ontology
# $ curl -L \
#     -o gene_info.gz \
#     https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz
# $ gunzip gene_info.gz

import csv
from pathlib import Path

from rdflib import RDF, RDFS, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL

HERE = Path(__file__).parent.resolve()
ONTOLOGY_DIR = HERE.parent.joinpath("ontology")
NCBI_GENE_FILE = ONTOLOGY_DIR.joinpath("gene_info")
OWL_OUTPUT_FILE = ONTOLOGY_DIR.joinpath("ncbi_gene_human.owl")

BASE = Namespace("http://purl.obolibrary.org/obo/NCBIGene_")
OBOINOWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")


def main() -> None:
    if not NCBI_GENE_FILE.exists():
        raise FileNotFoundError(f"NCBI Gene file not found: {NCBI_GENE_FILE}")

    g = Graph()
    g.bind("rdfs", RDFS)
    g.bind("oboInOwl", OBOINOWL)
    g.bind("owl", OWL)
    g.bind("ncbi_gene", BASE)

    with NCBI_GENE_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["#tax_id"] != "9606":  # Human genes only
                continue

            gene_id = row["GeneID"] if row["GeneID"] != "-" else None
            symbol = row["Symbol"] if row["Symbol"] != "-" else None
            synonyms = row["Synonyms"] if row["Synonyms"] != "-" else None

            if gene_id is None or symbol is None:
                continue

            gene_uri = URIRef(BASE[str(gene_id)])
            g.add((gene_uri, RDF.type, OWL.Class))
            g.add((gene_uri, RDFS.label, Literal(symbol)))
            if synonyms:
                for synonym in synonyms.split("|"):
                    g.add((gene_uri, OBOINOWL.hasExactSynonym, Literal(synonym)))

    g.serialize(destination=OWL_OUTPUT_FILE, format="xml")
    print(f"NCBI Gene OWL file written to: {OWL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
