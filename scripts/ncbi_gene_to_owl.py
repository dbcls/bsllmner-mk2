# Prepare NCBI Gene data.
# $ cd ../ontology
# $ curl -L \
#     -o gene_info.gz \
#     https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz
# $ gunzip gene_info.gz

import argparse
import csv
from pathlib import Path

from rdflib import RDF, RDFS, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL

HERE = Path(__file__).parent.resolve()
ONTOLOGY_DIR = HERE.parent.joinpath("ontology")
NCBI_GENE_FILE = ONTOLOGY_DIR.joinpath("gene_info")

BASE = Namespace("http://purl.obolibrary.org/obo/NCBIGene_")
OBOINOWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")
IAO_DEFINITION = URIRef("http://purl.obolibrary.org/obo/IAO_0000115")

TAXID_TO_SUFFIX = {
    "9606": "human",
    "10090": "mouse",
}


def output_path_for(taxid: str) -> Path:
    suffix = TAXID_TO_SUFFIX.get(taxid, taxid)
    return ONTOLOGY_DIR.joinpath(f"ncbi_gene_{suffix}.owl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NCBI gene_info TSV into a per-taxon OWL.")
    parser.add_argument(
        "--taxid",
        default="9606",
        help="NCBI Taxonomy ID to extract (default: 9606 = human; 10090 = mouse).",
    )
    args = parser.parse_args()
    taxid = str(args.taxid).strip()

    if not NCBI_GENE_FILE.exists():
        raise FileNotFoundError(f"NCBI Gene file not found: {NCBI_GENE_FILE}")

    g = Graph()
    g.bind("rdfs", RDFS)
    g.bind("oboInOwl", OBOINOWL)
    g.bind("owl", OWL)
    g.bind("ncbi_gene", BASE)

    # Declare annotation properties explicitly so owlready2 picks up values during OWL load.
    g.add((OBOINOWL.hasExactSynonym, RDF.type, OWL.AnnotationProperty))
    g.add((IAO_DEFINITION, RDF.type, OWL.AnnotationProperty))

    with NCBI_GENE_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["#tax_id"] != taxid:
                continue

            gene_id = row["GeneID"] if row["GeneID"] != "-" else None
            symbol = row["Symbol"] if row["Symbol"] != "-" else None
            synonyms = row["Synonyms"] if row["Synonyms"] != "-" else None
            description = row.get("description")
            if description == "-" or not description:
                description = None

            if gene_id is None or symbol is None:
                continue

            gene_uri = URIRef(BASE[str(gene_id)])
            g.add((gene_uri, RDF.type, OWL.Class))
            g.add((gene_uri, RDFS.label, Literal(symbol)))
            if synonyms:
                for synonym in synonyms.split("|"):
                    g.add((gene_uri, OBOINOWL.hasExactSynonym, Literal(synonym)))
            if description:
                g.add((gene_uri, IAO_DEFINITION, Literal(description)))

    output_path = output_path_for(taxid)
    g.serialize(destination=output_path, format="xml")
    print(f"NCBI Gene OWL file written to: {output_path}")


if __name__ == "__main__":
    main()
