from model.resnet8 import ResNet8
from model.mobilenet import MobileNet


def get_model(args):

    if args.model == "resnet":
        model = ResNet8(args).to(args.device)
    elif args.model == "mobilenet":
        model = MobileNet(args).to(args.device)

    return model