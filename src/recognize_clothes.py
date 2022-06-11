import torch, random, os
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from cnn_net import MyNet

FILE = "save/my_model.pt"
batch_size = 4

# Transform each image into tensor
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

# open image
filepath = "None"
while (os.path.exists(filepath) == False):
	filepath = input("Enter filepath test image:")
img = Image.open(filepath)

# inverse color
invert_image = ImageOps.invert(img)
# resize good format
resize = transforms.functional.resize(invert_image, size=[28, 28])
# apply gray filter 3 channels -> 1 channel
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
						transforms.Normalize([0.5], [0.5])])
to_tensor = transforms.ToTensor()
my_tensor = torch.empty(4, 1, 28, 28)
resize = transform(to_tensor(resize))
for i in range(4):
	my_tensor[i] = resize

#load model
loaded_model = MyNet()
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

# examples = iter(test_loader)
# for i in range(random.randint(2, 500)):
#     example_data, example_targets = examples.next()

# output = loaded_model(example_data)
output = loaded_model(my_tensor)
_, pred = torch.max(output.data, 1)
plt.subplot(2,3, 1)
int_pred = int(pred[0])
plt.title(int_pred)
plt.imshow(img, cmap='gray')
plt.show()