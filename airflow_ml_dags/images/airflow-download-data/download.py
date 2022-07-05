from pathlib import Path

import click
from sklearn.datasets import load_breast_cancer


@click.command("download")
@click.argument("output_dir")
def main(output_dir: str):
    features, targets = load_breast_cancer(return_X_y=True, as_frame=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "data.csv", index=False)
    targets.to_csv(output_dir / "target.csv", index=False)


if __name__ == '__main__':
    main()
