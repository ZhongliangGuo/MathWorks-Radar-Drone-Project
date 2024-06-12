IMPLEMENTED_NETS = (
    'resnet18',
    'resnet50',
    'convnext_tiny',
    'convnext_base',
    'efficientnet_v2_s',
    'efficientnet_v2_m',
    'resnext50_32x4d',
    'alexnet',
)

DRONE_CLASS_INDEX = {
    'Autel_Evo_II': 0,
    'DJI_Matrice_210': 1,
    'DJI_Mavic_3': 2,
    'DJI_Mini_2': 3,
    'Yuneec_H520E': 4,
}

SUPPORTED_TASKS = {
    'binary': 2,
    'drone-classification': len(DRONE_CLASS_INDEX),
}

NUM_TO_CLASS_BINARY = {
    0: 'non-drone',
    1: 'drone',
}

NUM_TO_CLASS_DRONES = {
    DRONE_CLASS_INDEX[key]: key for key in DRONE_CLASS_INDEX.keys()
}
