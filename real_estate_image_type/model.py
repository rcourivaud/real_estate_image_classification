from torch import nn
import torch
from torchvision import models
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class LinearClassifier(nn.Module):
    def __init__(self, in_feature_number, class_number):
        super(LinearClassifier, self).__init__()
        self.f1 = nn.Linear(in_feature_number, 256)
        self.f2 = nn.Linear(256, class_number)
        self.dropout = nn.Dropout(p=0.1)
        self.soft_max = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.f1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.f2(x)
        output = self.soft_max(x)
        return output

def _build_last_classifier(in_feature_number, class_number):
    return LinearClassifier(in_feature_number, class_number)


def build_model(num_labels, device):
    model = models.densenet161(pretrained=True)

    # Turn off training for their parameters
    for param in model.parameters():
        param.requires_grad = False

    logger.info("DenseNet successfully loaded and make it un-trainable")

    last_classifier = _build_last_classifier(model.classifier.in_features, num_labels)
    logger.info("Successfully build final classifier")

    # Replace default classifier with new classifier
    logger.info("Replace classifier of DenseNet")

    model.classifier = last_classifier
    logger.info("Try to expose model to device {device}".format(device=device))

    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)
    logger.info("Successfully exposed to device {device}".format(device=device))

    return model


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model, path)