"""Export the four JSON Schemas from runtime models; generated files stay out of Git."""
import argparse
from pathlib import Path

from backends.base import Report
from mapping_ir.models import Mapping
from specs.io import write_json
from specs.models import Architecture, Program

MODELS = {"arch": Architecture, "program": Program, "mapping": Mapping, "report": Report}


def schema_for(model):
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **model.model_json_schema()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "specs" / "schemas")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, model in MODELS.items():
        write_json(args.output / f"{name}.schema.json", schema_for(model))


if __name__ == "__main__":
    main()
