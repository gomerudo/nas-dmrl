import math
import numpy as np
import nasgym.utl.configreader as cr
from nasgym import nas_logger
from nasgym import CONFIG_INI
from nasgym.net_ops.net_benchmark import NetBenchmarking
from nasgym.envs.factories import DatasetHandlerFactory
from nasgym.envs.factories import TrainerFactory
from nasgym.utl.miscellaneous import compute_str_hash
from nasgym.utl.miscellaneous import state_to_string


if __name__ == '__main__':


    n_epochs = 100
    dataset_handler = DatasetHandlerFactory.get_handler("meta-dataset")

    try:
        log_path = CONFIG_INI[cr.SEC_DEFAULT][cr.PROP_LOGPATH]
    except KeyError:
        log_path = "workspace"

    log_trainer_dir = "{lp}/trainer-{h}".format(lp=log_path, h="vgg19")

    batch_size, decay_steps, beta1, beta2, epsilon, fcl_units, dropout_rate, \
        split_prop = TrainerFactory._load_default_trainer_attributes()

    trainset_length = math.floor(
        dataset_handler.current_n_observations()*(1. - split_prop)
    )
    evaluator = NetBenchmarking(
        input_shape=dataset_handler.current_shape(),
        n_classes=dataset_handler.current_n_classes(),
        batch_size=batch_size,
        log_path=log_trainer_dir,
        variable_scope="cnn-{h}".format(h="vgg19"),
        n_epochs=n_epochs,
        op_beta1=0.9,
        op_beta2=0.999,
        op_epsilon=10e-08,
        dropout_rate=0.4,
        n_obs_train=trainset_length
    )

    train_features, train_labels = None, None
    val_features, val_labels = None, None

    def custom_train_input_fn():
        return dataset_handler.current_train_set()

    def custom_eval_input_fn():
        return dataset_handler.current_validation_set()

    train_input_fn = custom_train_input_fn
    eval_input_fn = custom_eval_input_fn

    nas_logger.debug(
        "Training architecture %s for %d epochs", "vgg19", n_epochs
    )
    evaluator.train(
        train_data=train_features,
        train_labels=train_labels,
        train_input_fn=train_input_fn,
        n_epochs=n_epochs  # As specified by BlockQNN
    )

    nas_logger.debug("Evaluating architecture %s", "vgg19")
    res = evaluator.evaluate(
        eval_data=val_features,
        eval_labels=val_labels,
        eval_input_fn=eval_input_fn
    )
    nas_logger.debug(
        "Train-evaluation procedure finished for architecture %s",
        "vgg19"
    )

    accuracy = res['accuracy']*100

    nas_logger.info("Final accuracy is %f", accuracy)
