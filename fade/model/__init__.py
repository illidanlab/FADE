from typing import Type
from torch.nn import Module


def import_model(model_name, sub_module) -> Type[Module]:
    # These has to be loaded for access the `if_personal_local_adaptation()`.
    import importlib
    module = importlib.import_module(f"fade.model.{sub_module}")
    return getattr(module, model_name)


def get_model(name="", dataset="", **kwargs):
    model = None
    if "Mnist" in dataset or dataset in ["comb/Digit", "comb/M2U", "comb/U2M", "comb/S2M", "USPS", "SVHN"]:
        if name == "cnn-split":
            name = "MnistCnnSplit"
        elif name == "cnn-seprep":
            name = "MnistCnnSepRep"
        model = import_model(name, "mnist")(**kwargs)
    elif dataset.startswith("Office") or dataset.startswith("comb/Office") \
            or dataset.startswith("Visda") or dataset.startswith("comb/Visda")\
            or dataset.startswith("DomainNet") or dataset.startswith("comb/DomainNet"):
        if name == "cnn-split":
            name = "OfficeCnnSplit"
        elif name == "dnn-split":
            name = "OfficeDnnSplit"
        model = import_model(name, "office")(**kwargs)
    elif dataset.startswith("Celeba"):
        if name == "cnn-split":
            name = "CelebaCnnSplit"
        model = import_model(name, "celeba")(**kwargs)
    elif dataset.startswith("Adult") or dataset.startswith("comb/Adult"):
        if name == "dnn-split":
            name = "AdultDNNSplit"
        model = import_model(name, "adult")(**kwargs)
    if model is None:
        raise NotImplementedError(f"{name}, {dataset}, thus model is {model}")
    return model
