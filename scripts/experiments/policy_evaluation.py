import math
import numpy as np
import nasgym.utl.configreader as cr
from nasgym import nas_logger
from nasgym import CONFIG_INI
from nasgym.net_ops.net_eval import NetEvaluation
from nasgym.envs.factories import DatasetHandlerFactory
from nasgym.envs.factories import TrainerFactory
from nasgym.utl.miscellaneous import compute_str_hash
from nasgym.utl.miscellaneous import state_to_string


if __name__ == '__main__':

    state = np.array([
        [0, 0, 0, 0, 0],  # 1
        [0, 0, 0, 0, 0],  # 2
        [0, 0, 0, 0, 0],  # 3
        [0, 0, 0, 0, 0],  # 4
        [0, 0, 0, 0, 0],  # 5
        [1, 1, 3, 0, 0],  # 6
        [2, 1, 3, 1, 0],  # 7
        [3, 1, 3, 2, 0],  # 8
        [4, 2, 2, 3, 0],  # 9
        [5, 2, 3, 4, 0],  # 10
    ])

    n_epochs = 100
    dataset_handler = DatasetHandlerFactory.get_handler("meta-dataset")

    hash_state = compute_str_hash(state_to_string(state))
    composed_id = "{d}-{h}".format(
        d=dataset_handler.current_dataset_name(), h=hash_state
    )
    try:
        log_path = CONFIG_INI[cr.SEC_DEFAULT][cr.PROP_LOGPATH]
    except KeyError:
        log_path = "workspace"

    log_trainer_dir = "{lp}/trainer-{h}".format(lp=log_path, h=composed_id)

    batch_size, decay_steps, beta1, beta2, epsilon, fcl_units, dropout_rate, \
        split_prop = TrainerFactory._load_default_trainer_attributes()

    trainset_length = math.floor(
        dataset_handler.current_n_observations()*(1. - split_prop)
    )
    evaluator = NetEvaluation(
        encoded_network=state,
        input_shape=dataset_handler.current_shape(),
        n_classes=dataset_handler.current_n_classes(),
        batch_size=batch_size,
        log_path=log_trainer_dir,
        variable_scope="cnn-{h}".format(h=hash_state),
        n_epochs=n_epochs,
        op_beta1=0.9,
        op_beta2=0.999,
        op_epsilon=10e-08,
        fcl_units=1024,
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
        "Training architecture %s for %d epochs", composed_id, n_epochs
    )
    evaluator.train(
        train_data=train_features,
        train_labels=train_labels,
        train_input_fn=train_input_fn,
        n_epochs=n_epochs  # As specified by BlockQNN
    )

    nas_logger.debug("Evaluating architecture %s", composed_id)
    res = evaluator.evaluate(
        eval_data=val_features,
        eval_labels=val_labels,
        eval_input_fn=eval_input_fn
    )
    nas_logger.debug(
        "Train-evaluation procedure finished for architecture %s",
        composed_id
    )

    accuracy = res['accuracy']*100

    nas_logger.info("Final accuracy is %f", accuracy)
