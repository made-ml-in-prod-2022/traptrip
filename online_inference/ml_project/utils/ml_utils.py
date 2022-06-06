import pickle


def load_pickle(model_path: str):
    with open(model_path, "rb") as fin:
        model = pickle.load(fin)
    return model
