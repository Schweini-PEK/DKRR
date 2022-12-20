from argparse import ArgumentParser, Namespace
import ruamel.yaml


def parse_config(config: str = 'configs/relative.yaml'):
    parser = ArgumentParser()

    # if config is None:
    #     parser.add_argument('--config', type=str)

    args, unknown = parser.parse_known_args()
    with open(config, "r") as yml:
        train_config = ruamel.yaml.load(yml, Loader=ruamel.yaml.Loader)

    # argument precedence is config file > command line
    args = Namespace(
        **{
            **vars(args),
            **{k: v for k, v in train_config.items() if not isinstance(v, dict)},  # top level config items
            **{k: v for k, v in train_config['training'].items()},  # training configurations
            **{k: v for k, v in train_config['hparams'].items()},  # model specific hparams
        }
    )

    return args