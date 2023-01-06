import torchvision.models as models
import torch
import torch.nn as nn


class DecisionResnetModel(nn.Module):

  def __init__(self,num_classes, resnet=50, pretrained=True, layers_to_freeze=[]):
    super(DecisionResnetModel,self).__init__()
    if resnet==18:
      self.resnet = models.resnet18(pretrained=pretrained)
    elif resnet==34:
      self.resnet = models.resnet34(pretrained=pretrained)
    elif resnet==50:
      self.resnet = models.resnet50(pretrained=pretrained)
    elif resnet==101:
      self.resnet = models.resnet101(pretrained=pretrained)
    elif resnet==152:
      self.resnet = models.resnet152(pretrained=pretrained)

    for layer in layers_to_freeze:

      if layer == 'conv1':
        for param in self.resnet.conv1.parameters():
          param.requires_grad = False
      if layer == 'bn1':
        for param in self.resnet.bn1.parameters():
          param.requires_grad = False
      if layer == 'layer1':
        for param in self.resnet.layer1.parameters():
          param.requires_grad = False
      if layer == 'layer2':
        for param in self.resnet.layer2.parameters():
          param.requires_grad = False
      if layer == 'layer3':
        for param in self.resnet.layer3.parameters():
          param.requires_grad = False
      if layer == 'layer4':
        for param in self.resnet.layer4.parameters():
          param.requires_grad = False

    num_features = self.resnet.fc.in_features
    self.resnet.fc = nn.Linear(num_features, num_classes)

  def forward(self, input, before_sigmoid=False):

    scores = self.resnet(input)
    proba = torch.sigmoid(scores)
    if before_sigmoid:
        return scores
    return proba


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class DenseNet121(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.feat_extract = models.densenet121(pretrained=pretrained)
        self.feat_extract.classifier = Identity()
        self.output_size = 1024

    def forward(self, x):
        return self.feat_extract(x)


class DecisionDensenetModel(nn.Module):

    def __init__(self, num_classes=40, pretrained=False):
        super().__init__()
        self.feat_extract = DenseNet121(pretrained=pretrained)
        self.classifier = nn.Linear(self.feat_extract.output_size, num_classes)

    def forward(self, input, before_sigmoid=False):

        feat = self.feat_extract(input)
        scores = self.classifier(feat)
        proba = torch.sigmoid(scores)
        if before_sigmoid:
            return scores
        return proba


class DecisionExplainDensenetModel(nn.Module):

    def __init__(self, num_attributes=4, num_explanations=21, dropout=0.3, pretrained=False):
        super().__init__()
        self.feat_extract = DenseNet121(pretrained=pretrained)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.feat_extract.output_size, num_attributes)
        self.explainer = nn.Linear(self.feat_extract.output_size+num_attributes, num_explanations)

    def forward(self, input, before_sigmoid=False, return_reason=False):
        feat = self.feat_extract(input)
        feat = self.dropout1(feat)
        scores = self.classifier(feat)
        proba = torch.sigmoid(scores)

        feat_explainer = torch.cat((self.dropout2(feat), scores), -1)
        scores_reason = self.explainer(feat_explainer)
        proba_reson = torch.sigmoid(scores_reason)
        if before_sigmoid:
          if return_reason:
            return scores, scores_reason
          else:
            return scores
        if return_reason:
          return proba, proba_reson
        else: 
          return proba