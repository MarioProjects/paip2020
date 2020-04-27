from .pspnet import *
from .resnet import *


def model_selector(model_name, num_classes=1, in_size=(), aux=False):
    """

    :param model_name:
    :param num_classes:
    :param in_size:
    :param aux:
    :return:
    """
    if "pspnet" in model_name:
        return psp_model_selector(model_name, num_classes, in_size, aux)
    if "resnet34" in model_name:
        return resnet_model_selector(model_name, num_classes)
    else:
        assert False, "Unknown model selected: {}".format(model_name)
