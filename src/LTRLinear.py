import torch
from torch import nn

from .LightGCN import LightGCNRank
from .RankingModel import RankingModel
from .LTRBaseModel import LTRBaseModel, LTRBaseWPop
from .LightGCN import LightGCNScore
from .ScoringModel import ScoringModel


class LTRLinearRank(LTRBaseModel, RankingModel):

    def _add_vars(self, *args, **kwargs):
        super()._add_vars(*args, **kwargs)
        self.foundation_class = LightGCNRank


class LTRLinearRankWPop(LTRBaseWPop, LTRLinearRank):
    ...


class LTRLinearScore(LTRBaseModel, ScoringModel):

    def _add_vars(self, *args, **kwargs):
        super()._add_vars(*args, **kwargs)
        self.foundation_class = LightGCNScore


class LTRLinearScoreWPop(LTRBaseWPop, LTRLinearScore):
    ...