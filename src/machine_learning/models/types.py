# -*- coding: utf-8 -*-
from typing import List

import timm

CLASSIFICATION_MODELS: List[str] = timm.list_models()
REGRESSION_MODELS: List[str] = timm.list_models()
