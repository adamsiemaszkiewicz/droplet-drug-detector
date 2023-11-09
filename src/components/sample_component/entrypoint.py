# -*- coding: utf-8 -*-
from src.common.utils.logger import get_logger
from src.components.sample_component.config import SampleConfig
from src.components.sample_component.options import parse_args

_logger = get_logger(__name__)


def main() -> None:
    config: SampleConfig = parse_args()

    _logger.info(f"Running with following config: {config}")


if __name__ == "__main__":
    main()
