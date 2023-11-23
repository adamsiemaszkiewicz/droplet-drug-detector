# -*- coding: utf-8 -*-
from src.aml.components.sample_component.config import SampleConfig
from src.aml.components.sample_component.options import get_config
from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


def main() -> None:
    config: SampleConfig = get_config()
    _logger.info(f"Running with following config: {config}")


if __name__ == "__main__":
    main()
