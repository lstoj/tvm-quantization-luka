import torch
import torchvision
# assigns model to an instance of resnet50
model = torchvision.models.resnet50(weights = None).eval()
# weight = None - assignes the 50 layers of the structure to random values
# .eval() - switches to evaluation mode rather than training; also for use of the model "setting"


x = torch.randn(1,3,224,224)
# creates an image with 3 different color types that is 224 x 224

test = model(x)
# testing out if works
print(test.shape)
