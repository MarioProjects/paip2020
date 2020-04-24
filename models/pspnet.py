"""
https://arxiv.org/abs/1612.01105
"""
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch import nn


class PSPNetResnetD101b(nn.Module):
    def __init__(self, num_classes, pretrained="scratch", in_size=(), aux=False):
        """
        :param num_classes: Number of classes of the problem / number channels output
        :param pretrained: Can be "scratch", "imagenet_encoder", "coco", "coco_encoder", "voc" or "voc_encoder"
        :param aux: Return middle and last output to help trainning process
        """
        super(PSPNetResnetD101b, self).__init__()

        if pretrained not in ["scratch", "imagenet_encoder", "coco", "coco_encoder", "voc", "voc_encoder"]:
            assert False, "Unknown pretrained: {}".format(pretrained)
        pretrained_weights = True if pretrained in ["coco", "coco_encoder", "voc", "voc_encoder"] else False

        # Ya sea scratch o preentrenado debemos coger la arquitectura entera
        if "coco" in pretrained:
            if "coco_encoder" in pretrained:
                self.model = ptcv_get_model("pspnet_resnetd101b_coco", pretrained=False, in_size=in_size, aux=aux)
                tmp_model = ptcv_get_model("pspnet_resnetd101b_coco", pretrained=True, in_size=in_size, aux=aux)
                self.model.backbone = tmp_model.backbone
            else:
                self.model = ptcv_get_model("pspnet_resnetd101b_coco", pretrained=pretrained_weights, in_size=in_size, aux=aux)
        else:
            if "voc_encoder" in pretrained:
                self.model = ptcv_get_model("pspnet_resnetd101b_voc", pretrained=False, in_size=in_size, aux=aux)
                tmp_model = ptcv_get_model("pspnet_resnetd101b_voc", pretrained=True, in_size=in_size, aux=aux)
                self.model.backbone = tmp_model.backbone
            else:
                self.model = ptcv_get_model("pspnet_resnetd101b_voc", pretrained=pretrained_weights, in_size=in_size, aux=aux)

        if pretrained == "imagenet_encoder":
            resnetd101b = ptcv_get_model("resnetd101b", pretrained=True, in_size=in_size)
            self.model.backbone.load_state_dict(resnetd101b.features.state_dict())

        self.model.final_block.conv2 = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        if aux:
            self.model.aux_block.conv2 = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.aux = aux

    def forward(self, x):
        return self.model(x)


def psp_model_selector(model_name, num_classes, in_size, aux):

    if model_name == "pspnet_resnetd101b_scratch":
        pretrained = "scratch"
    elif model_name == "pspnet_resnetd101b_imagenet_encoder":
        pretrained = "imagenet_encoder"
    elif model_name == "pspnet_resnetd101b_coco":
        pretrained = "coco"
    elif model_name == "pspnet_resnetd101b_coco_encoder":
        pretrained = "coco_encoder"
    elif model_name == "pspnet_resnetd101b_voc":
        pretrained = "voc"
    elif model_name == "pspnet_resnetd101b_voc_encoder":
        pretrained = "voc_encoder"
    else:
        assert False, "Uknown pretrained configuration for PSPNet: {}".format(model_name)

    model = PSPNetResnetD101b(pretrained=pretrained, num_classes=num_classes, in_size=in_size, aux=aux)

    if "encoder" in pretrained:  # Frost encoder
        for param in model.model.parameters():  # Frost model
            param.requires_grad = False
        for param in model.model.pool.parameters():  # Defrost decoder
            param.requires_grad = True
        for param in model.model.final_block.parameters():  # Defrost decoder
            param.requires_grad = True
        if aux:
            for param in model.model.aux_block():  # Defrost decoder
                param.requires_grad = True
    elif pretrained in ["coco", "voc"]:  # Frost all model
        for param in model.model.parameters():  # Frost model
            param.requires_grad = False
    else:  # Defrost all model
        for param in model.model.parameters():  # Defrost model
            param.requires_grad = True

    for param in model.model.final_block.conv2.parameters():  # Always defrost last/top layer!
        param.requires_grad = True

    return model.cuda()
