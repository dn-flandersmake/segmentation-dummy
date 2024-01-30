from segmentation_models_pytorch import create_model

def get_model(name, model_opts):
    model = create_model(name, **model_opts)
    return model