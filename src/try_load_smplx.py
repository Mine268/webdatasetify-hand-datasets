import pickle as pkl

with open("models/mano/MANO_RIGHT.pkl", "rb") as f:
    obj = pkl.load(f, encoding="latin1")

print(obj)