# PlasticineLab: A Soft-Body Manipulation Benchmark with Differentiable Physics

## Usage
 - Install `python3 -m pip install -e .`
 - Run `python3 -m plb.algorithms.solve --algo [algo] --env_name [env_name] --path [output-dir]`. It will run algorithms `algo` for environment `env-name` and store results in `output-dir`. For example
    `python3 -m plb.algorithms.solve --algo action --env_name Move-v1 --path output` will run call an Adam optimizer to optimize an action sequence in environment `Move-v1`
