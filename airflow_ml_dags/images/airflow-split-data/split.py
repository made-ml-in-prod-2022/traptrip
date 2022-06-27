from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("split")
@click.option("--input-dir")
@click.option("--output-dir")
def split(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    features = pd.read_csv(input_dir / "data.csv")
    targets = pd.read_csv(input_dir / "target.csv")
    df = pd.concat([features, targets], axis=1)
    train_df, valid_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, shuffle=True
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    valid_df.to_csv(output_dir / "valid.csv", index=False)


if __name__ == '__main__':
    split()
 