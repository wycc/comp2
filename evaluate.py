import torch
import torchvision.transforms as transforms
import json
from torchvision import datasets
import sys

def evaluate_model(model):
  # Evaluate the model
  # Load test dataset
  dataset_dir = 'dataset'
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: torch.flatten(x))
  ])
  test_dataset = datasets.ImageFolder(dataset_dir, transform=transform)
  test_dataloder = torch.utils.data.DataLoader(test_dataset, batch_size=32)

  # iterate all batches on the dataset
  correct = 0
  total = 0
  num=0
  for images, labels in test_dataloder:
    #print(images.shape)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    num += labels.size(0)
    if sys.argv[2] != 'testing':
      if num > len(test_dataset)*0.2:
        break
    correct += (predicted == labels).sum().item()


  # Return evaluation result

  ret = {"total": total, "correct": correct, "accurancy": correct / total * 100, "status": "success %g" % (correct / total * 100)}
  # return json string
  print(json.dumps(ret))

# load model from argv[1]
model = torch.load(sys.argv[1])
evaluate_model(model)
