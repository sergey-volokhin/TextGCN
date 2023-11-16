from .kg_models import DatasetKG, TextModelKG
from .reviews_models import DatasetReviews, TextModelReviews


class TextData(DatasetKG, DatasetReviews):
    pass


class TextModel(TextModelReviews, TextModelKG):
    pass


class TestModel(TextModel):
    ''' evaluate the Simple models (concat text emb at inference) '''

    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.items_as_avg_reviews = dataset.items_as_avg_reviews
        self.users_as_avg_reviews = dataset.users_as_avg_reviews
        self.users_as_avg_desc = dataset.users_as_avg_desc
        self.items_as_desc = dataset.items_as_desc

        for fn in [self.representation_rev_rev,
                   self.representation_kg_kg,
                   self.representation_rev_kg,
                   self.representation_kg_rev]:
            self.representation = fn
            self.evaluate()
        exit()

    @property
    def representation_kg_kg(self):
        return self.users_as_avg_desc, self.items_as_desc

    @property
    def representation_rev_kg(self):
        return self.users_as_avg_reviews, self.items_as_desc

    @property
    def representation_kg_rev(self):
        return self.users_as_avg_desc, self.items_as_avg_reviews

    @property
    def representation_rev_rev(self):
        return self.users_as_avg_reviews, self.items_as_avg_reviews
