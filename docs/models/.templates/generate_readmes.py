"""
Run this script to generate the model-index files in `models` from the templates in `.templates/models`.
"""

import argparse
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

import modelindex


def generate_readmes(templates_path: Path, dest_path: Path):
    """Add the code snippet template to the readmes"""
    readme_templates_path = templates_path / "models"
    code_template_path = templates_path / "code_snippets.md"

    env = Environment(
        loader=FileSystemLoader([readme_templates_path, readme_templates_path.parent]),
    )

    for readme in readme_templates_path.iterdir():
        if readme.suffix == ".md":
            template = env.get_template(readme.name)

            # get the first model_name for this model family
            mi = modelindex.load(str(readme))
            model_name = mi.models[0].name

            full_content = template.render(model_name=model_name)

            # generate full_readme
            with open(dest_path / readme.name, "w") as f:
                f.write(full_content)


def main():
    parser = argparse.ArgumentParser(description="Model index generation config")
    parser.add_argument(
        "-t",
        "--templates",
        default=Path(__file__).parent / ".templates",
        type=str,
        help="Location of the markdown templates",
    )
    parser.add_argument(
        "-d",
        "--dest",
        default=Path(__file__).parent / "models",
        type=str,
        help="Destination folder that contains the generated model-index files.",
    )
    args = parser.parse_args()
    templates_path = Path(args.templates)
    dest_readmes_path = Path(args.dest)

    generate_readmes(
        templates_path,
        dest_readmes_path,
    )


if __name__ == "__main__":
    main()
