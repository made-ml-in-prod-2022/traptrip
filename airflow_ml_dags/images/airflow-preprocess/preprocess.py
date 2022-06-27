import pickle
from pathlib import Path

import click
import pandas as pd
from sklearn.preprocessing import StandardScaler


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--save-transformer-dir")
def main(input_dir: str, output_dir: str, save_transformer_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    save_transformer_dir = Path(save_transformer_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_transformer_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()

    train_data = pd.read_csv(input_dir / "train.csv")
    valid_data = pd.read_csv(input_dir / "valid.csv")

    train_features, train_targets = train_data.drop("target", axis=1), train_data.target
    valid_features, valid_targets = valid_data.drop("target", axis=1), valid_data.target

    train_features = pd.DataFrame(scaler.fit_transform(train_features))
    valid_features = pd.DataFrame(scaler.transform(valid_features))

    train_df = pd.concat([train_features, train_targets], axis=1)
    valid_df = pd.concat([valid_features, valid_targets], axis=1)

    train_df.to_csv(output_dir / "train.csv", index=False)
    valid_df.to_csv(output_dir / "valid.csv", index=False)
    with open(save_transformer_dir / "scaler.pkl", "wb") as fout:
        pickle.dump(scaler, fout)


if __name__ == '__main__':
    main()
 