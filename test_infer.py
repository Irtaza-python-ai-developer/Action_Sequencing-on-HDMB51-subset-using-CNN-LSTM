from PIL import Image
from src.inference import predict_action

img = Image.open("samples/wave1.jpg")  # put any frame here
annotated, label = predict_action(img)

img2 = Image.open("samples/test1.jpg")  # put any frame here
annotated2, label2 = predict_action(img)

print(label, label2)
annotated.show()
