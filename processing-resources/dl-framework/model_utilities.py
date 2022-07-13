






def mi_rims_module_generate_hyperparameter_sets(run_config_dict, model_config_dict):
    lf = optimization_config_dict['loss_functions'].split(',')
    lr = [float(lr.strip()) for lr in optimization_config_dict['learning_rates'].split(',')]
    bs = [int(bs.strip()) for bs in optimization_config_dict['batch_sizes'].split(',')]
    op = [op.strip() for op in optimization_config_dict['optimizers'].split(',')]
    hdo = [float(lr.strip()) for lr in model_config_dict['bert_hidden_dropout_prob'].split(',')]
    ado = [float(lr.strip()) for lr in model_config_dict['bert_attention_probs_dropout_prob'].split(',')]
    plg = [op.strip() for op in model_config_dict['pooling_strategy'].split(',')]
    hparams = []
    for loss_function, learning_rate, batch_size, optimizer, hidden_dropout_prob, attention_probs_dropout_prob, pooling_strategy in \
            itertools.product(lf, lr, bs, op, hdo, ado, plg):
        hparams.append({
            'loss_function': loss_function,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'llm_hidden_dropout_prob': hidden_dropout_prob,
            'llm_attention_probs_dropout_prob': attention_probs_dropout_prob,
            'pooling_strategy': pooling_strategy
        })
    return hparams