from .app import (
    CheckpointEvaluator,
    ConfigComposer,
    EmbraceCliBuilder,
    LaunchArgumentAdapter,
    OutputLayoutPlanner,
    TaskApplication,
    TRAINING_TASKS,
)

import warnings
warnings.filterwarnings("ignore")


class EmbraceLauncher:
    def __init__(self, argv=None):
        del argv
        self.config_composer = ConfigComposer()
        self.output_planner = OutputLayoutPlanner()

    def run(self, args):
        if args.command == "evaluate":
            CheckpointEvaluator(exp_dir=args.exp_dir, ckpt_name=args.ckpt_name).run()
            return

        if args.command == "rescue":
            self._run_rescue(args)
            return

        adapter = LaunchArgumentAdapter(args)
        cfg = self.config_composer.compose(adapter.build_compose_request())
        seeds = adapter.resolve_seeds(cfg)
        launch_plan = self.output_planner.build(cfg, seeds)

        for seed in seeds:
            seed_cfg, output_dir = self.config_composer.materialize_seed_cfg(cfg, launch_plan, seed)
            TaskApplication(
                cfg=seed_cfg,
                output_dir=output_dir,
            ).run()

    def _run_rescue(self, args):
        base_cfg = self.config_composer.compose_saved_seed_cfg(args.exp_dir)
        task_name = str(base_cfg.task)
        if task_name not in TRAINING_TASKS:
            raise ValueError(f"`rescue` only supports training tasks: {sorted(TRAINING_TASKS)}. Got: {task_name}")

        adapter = LaunchArgumentAdapter(args)
        cfg = self.config_composer.compose_saved_seed_cfg(
            args.exp_dir,
            direct_overrides=adapter.build_direct_overrides(task_name=task_name),
        )
        seeds = adapter.resolve_seeds(cfg)
        launch_plan = self.output_planner.build(cfg, seeds)

        for seed in seeds:
            seed_cfg, output_dir = self.config_composer.materialize_seed_cfg(cfg, launch_plan, seed)
            TaskApplication(
                cfg=seed_cfg,
                output_dir=output_dir,
            ).run()


def main(argv=None):
    parser = EmbraceCliBuilder().build_parser()
    args = parser.parse_args(argv)

    EmbraceLauncher(argv=argv).run(args)


if __name__ == "__main__":
    main()
