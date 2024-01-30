from .SegmentationDataset import SegmentationDataset

def get_dataset(name, dataset_opts):
    if name == "segmentation_dataset":
        return SegmentationDataset(**dataset_opts)
    else:
        raise RuntimeError(f'Dataset {name} is not available.')