# Lazy imports to avoid circular import warnings when running as __main__
def __getattr__(name):
    if name in (
        'Trainer', 'run_experiment', 'run_ddp_experiment',
        'setup_ddp', 'cleanup_ddp', 'get_ddp_dataloaders'
    ):
        from src.training import trainer
        return getattr(trainer, name)
    elif name in ('evaluate_test_set', 'evaluate_test_set_by_match', 'print_results'):
        from src.training import evaluation
        return getattr(evaluation, name)
    elif name in (
        'load_model_from_checkpoint', 'run_inference_on_csv',
        'run_inference_on_folder', 'load_temperature'
    ):
        from src.training import inference
        return getattr(inference, name)
    elif name in (
        'compute_winner_accuracy', 'compute_log_loss', 'compute_ece', 'compute_all_metrics'
    ):
        from src.training import metrics
        return getattr(metrics, name)
    elif name in (
        'find_optimal_temperature', 'calibrate_from_checkpoint',
        'calibrate_from_split_csv', 'compute_ece_with_temperature'
    ):
        from src.training import calibration
        return getattr(calibration, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'Trainer',
    'run_experiment',
    'run_ddp_experiment',
    'setup_ddp',
    'cleanup_ddp',
    'get_ddp_dataloaders',
    'evaluate_test_set',
    'evaluate_test_set_by_match',
    'print_results',
    'load_model_from_checkpoint',
    'run_inference_on_csv',
    'run_inference_on_folder',
    'load_temperature',
    'compute_winner_accuracy',
    'compute_log_loss',
    'compute_ece',
    'compute_all_metrics',
    'find_optimal_temperature',
    'calibrate_from_checkpoint',
    'calibrate_from_split_csv',
    'compute_ece_with_temperature',
]

