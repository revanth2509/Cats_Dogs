from datasets import CustomeTransform

tr = CustomeTransform(batch_size=64)

data, valid = tr.loader(r"C:\Users\revan\Downloads\Cats_V1\training_set\training_set")

print(data, valid)