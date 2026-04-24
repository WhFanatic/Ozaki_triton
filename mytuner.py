import optuna
import triton
from triton.runtime.autotuner import Autotuner, Config
from triton import knobs
import builtins, time, itertools

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _expand_configs(param_space):
    """从 param_space 展开全量 configs 列表。"""
    cfg_keys = {'num_warps', 'num_stages', 'num_ctas'}
    return [Config({k: v for k, v in zip(param_space, v) if k not in cfg_keys},
                 **{k: v for k, v in zip(param_space, v) if k in cfg_keys})
            for v in itertools.product(*param_space.values())]


class OptunaAutotuner(Autotuner):

    def __init__(self, fn, arg_names, param_space, key, n_trials=100, sampler=None, **kwargs):
        self.param_space = param_space
        self.n_trials = n_trials
        self.sampler = sampler if sampler else optuna.samplers.TPESampler()

        configs = _expand_configs(param_space)
        super().__init__(fn, arg_names, configs, key, **kwargs)

        assert hasattr(self, '_bench'), "Autotuner._bench not found, check triton version"

    def _optuna_bench(self, pruned_configs, *args, **kwargs):
            keys = sorted(self.param_space)
            to_key = lambda c: tuple((k, c.all_kwargs()[k]) for k in keys)
            all_map = {to_key(c): c for c in self.configs}
            pruned_keys = {to_key(c) for c in pruned_configs}
            timings = {}

            def objective(trial):
                ck = tuple((k, trial.suggest_categorical(k, v)) for k, v in sorted(self.param_space.items()))
                if ck not in pruned_keys:
                    return float('inf')
                t = self._bench(*args, config=all_map[ck], **kwargs)
                timings[all_map[ck]] = t
                return t[0]

            study = optuna.create_study(direction="minimize", sampler=self.sampler)
            study.optimize(objective, n_trials=self.n_trials)
            if not timings:
                timings = {c: self._bench(*args, config=c, **kwargs) for c in pruned_configs}
            return timings

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
            key = [_args[key] for key in self.keys if key in _args]
            for _, arg in _args.items():
                if hasattr(arg, "dtype"):
                    key.append(str(arg.dtype))
            key = tuple(key)
            if key not in self.cache:
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)

                def benchmark():
                    bench_start = time.time()
                    # === 唯一改动 ===
                    timings = self._optuna_bench(pruned_configs, *args, **kwargs)
                    bench_end = time.time()
                    self.bench_time = bench_end - bench_start
                    self.cache[key] = builtins.min(timings, key=timings.get)
                    full_nargs = {**self.nargs, **kwargs, **self.cache[key].all_kwargs()}
                    self.pre_hook(full_nargs, reset_only=True)
                    self.configs_timings = timings

                if self.cache_results:
                    used_cached_result = self.check_disk_cache(key, pruned_configs, benchmark)
                else:
                    benchmark()

            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if knobs.autotuning.print and not used_cached_result:
            print(f"Triton autotuning for function {self.base_fn.__name__},\nwith key as {key},\n"
                  f"finished after {self.bench_time:.2f}s,\nbest config selected: {self.best_config};")
        if config.pre_hook is not None:
            full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
            config.pre_hook(full_nargs)
        ret = self.fn.run(*args, **kwargs, **config.all_kwargs())
        self.nargs = None
        return ret


def optuna_autotune(param_space, key, n_trials=100, sampler=None,
                    prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
                    warmup=None, rep=None, use_cuda_graph=False, do_bench=None, cache_results=False):
    def decorator(fn):
        return OptunaAutotuner(fn, fn.arg_names, param_space, key, n_trials=n_trials, sampler=sampler,
                               reset_to_zero=reset_to_zero, restore_value=restore_value, pre_hook=pre_hook,
                               post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                               use_cuda_graph=use_cuda_graph, do_bench=do_bench, cache_results=cache_results)
    return decorator
