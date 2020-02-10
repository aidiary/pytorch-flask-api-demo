import json
import io
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms


def get_model():
    model = models.densenet121(pretrained=True)
    model.eval()
    return model


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def format_class_name(class_name):
    class_name = class_name.replace('_', ' ')
    class_name = class_name.title()
    return class_name


model = get_model()
imagenet_class_index = json.load(open('static/imagenet_class_index.json'))


def get_prediction(image_bytes):
    try:
        tensor = transform_image(image_bytes)
        outputs = model.forward(tensor)
    except Exception:
        return 0, 'error'
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]
