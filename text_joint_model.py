from kg_models import DatasetKG, TextModelKG
from reviews_models import DatasetReviews, TextModelReviews


class TextData(DatasetKG, DatasetReviews):
    pass


class TextModel(TextModelReviews, TextModelKG):
    pass
