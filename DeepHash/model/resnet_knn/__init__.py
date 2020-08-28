from .resnet_knn import ResNetKNN
from .util import Dataset

def validation(database_img, query_img, config):
    model = ResNetKNN(config)
    img_database = Dataset(database_img, config.output_dim)
    img_query = Dataset(query_img, config.output_dim)
    return model.validation(img_query, img_database, config.R)
