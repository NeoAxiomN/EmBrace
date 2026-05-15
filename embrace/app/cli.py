from .config import ConfigRepository
from .utils import parse_bool

import argparse

FINETUNE_MODELS = ("cbramod", "codebrain", "csbrain", "labram")
ETMS_MODELS = ("eegnet", "deformer", "sera", "sparc", "contra", "cnntrans", "ffcl", "sttrans")
EMBRACE_MODELS = ("embrace",)
SOURCE_MODEL_NAMES = ("cbramod", "labram", "codebrain", "csbrain")
RUNTIME_MODES = ("auto", "cpu", "single", "ddp")
RUNTIME_BACKENDS = ("auto", "nccl", "gloo")


class EmbraceCliBuilder:
    def __init__(self, repository: ConfigRepository | None = None):
        self.repository = repository or ConfigRepository()
        self.dataset_names = tuple(payload["name"] for payload in self.repository.list_dataset_payloads())

    def build_parser(self):
        parser = argparse.ArgumentParser(
            description="EmBrace CLI",
            formatter_class=argparse.HelpFormatter,
        )
        subparsers = parser.add_subparsers(dest="command", required=True)
        self._add_finetune_parser(subparsers)
        self._add_fuse_parser(subparsers)
        self._add_etms_parser(subparsers)
        self._add_select_parser(subparsers)
        self._add_rescue_parser(subparsers)
        self._add_evaluate_parser(subparsers)
        return parser

    def _add_finetune_parser(self, subparsers):
        parser = self._new_run_parser(
            subparsers,
            "finetune",
            "Run finetuning for one foundation model.",
            FINETUNE_MODELS,
        )
        self._add_training_args(parser)
        self._add_finetune_model_args(parser)

    def _add_fuse_parser(self, subparsers):
        parser = self._new_run_parser(
            subparsers,
            "fuse",
            "Run Embrace fusion training.",
            EMBRACE_MODELS,
        )
        self._add_training_args(parser)
        self._add_embrace_model_args(parser)

    def _add_etms_parser(self, subparsers):
        parser = self._new_run_parser(
            subparsers,
            "etms",
            "Run ETM baselines.",
            ETMS_MODELS,
        )
        self._add_training_args(parser)
        group = parser.add_argument_group("ETM Model")
        group.add_argument("--dropout", type=float, default=None, help="Override model.params.dropout.")

    def _add_select_parser(self, subparsers):
        parser = self._new_run_parser(
            subparsers,
            "select",
            "Run Embrace model selection.",
            EMBRACE_MODELS,
        )
        group = parser.add_argument_group("Selection")
        group.add_argument("--batch-size", type=int, default=None, help="Override model.trainer.batch_size.")
        group.add_argument("--sample-ratio", type=float, default=None, help="Override sample_ratio.")
        group.add_argument("--parc-n-dims", type=float, default=None, help="Override parc.n_dims.")
        self._add_embrace_model_args(parser)

    def _add_rescue_parser(self, subparsers):
        parser = subparsers.add_parser(
            "rescue",
            help="Restart training from a saved resolved config",
            description="Load a saved config_resolved.yaml, apply CLI overrides, and restart training from scratch.",
            formatter_class=argparse.HelpFormatter,
        )
        parser.add_argument("--exp-dir", type=str, required=True, help="Saved seed experiment directory.")
        parser.add_argument(
            "--model-folder-id",
            type=str,
            default=None,
            help="Override model.model_folder_id for the rescued run.",
        )
        self._add_run_args(parser)
        self._add_dataset_args(parser)
        self._add_runtime_args(parser)
        self._add_training_args(parser)
        self._add_rescue_task_override_args(parser)

    @staticmethod
    def _add_evaluate_parser(subparsers):
        parser = subparsers.add_parser(
            "evaluate",
            help="Evaluate a checkpoint from a saved experiment",
            description="Load a saved config_resolved.yaml plus checkpoint and run test evaluation only.",
            formatter_class=argparse.HelpFormatter,
        )
        parser.add_argument("--exp-dir", type=str, required=True, help="Seed experiment directory.")
        parser.add_argument(
            "--ckpt-name",
            type=str,
            default="best_checkpoint.pth",
            help="Checkpoint filename inside the experiment directory.",
        )

    def _new_run_parser(self, subparsers, command, description, model_choices):
        parser = subparsers.add_parser(
            command,
            help=description,
            description=description,
            formatter_class=argparse.HelpFormatter,
        )
        self._add_config_args(parser, command, model_choices)
        self._add_run_args(parser)
        self._add_dataset_args(parser)
        self._add_runtime_args(parser)
        return parser

    def _add_config_args(self, parser, command, model_choices):
        group = parser.add_argument_group("Config")
        group.add_argument(
            "--config",
            type=str,
            default=None,
            help=f"Task YAML path. Defaults to config/tasks/{command}.yaml.",
        )
        group.add_argument(
            "--dataset",
            type=str,
            choices=self.dataset_names or None,
            default=None,
            help="Dataset name.",
        )
        group.add_argument("--dataset-config", type=str, default=None, help="Dataset YAML path.")
        group.add_argument("--model", type=str, choices=model_choices, default=None, help="Model name.")
        group.add_argument("--model-folder-id", type=str, default=None, help="Override model.model_folder_id.")

    @staticmethod
    def _add_run_args(parser):
        group = parser.add_argument_group("Run")
        group.add_argument("--seed", type=int, default=None, help="Single seed override.")
        group.add_argument(
            "--seeds",
            type=str,
            nargs="+",
            default=None,
            help="Multiple seeds, e.g. --seeds 0 1 2 or --seeds 0,1,2.",
        )
        group.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda.")
        group.add_argument("--output-base-dir", type=str, default=None, help="Override output.base_dir.")

    @staticmethod
    def _add_dataset_args(parser):
        group = parser.add_argument_group("Dataset Overrides")
        group.add_argument("--dataset-path", type=str, default=None, help="Override dataset.path.")
        group.add_argument("--num-workers", type=int, default=None, help="Override dataset.num_workers.")
        group.add_argument("--pin-memory", type=parse_bool, default=None, help="Override dataset.pin_memory.")
        group.add_argument("--drop-last", type=parse_bool, default=None, help="Override dataset.drop_last for train DataLoader.")
        group.add_argument("--chans", type=int, default=None, help="Override dataset.chans.")
        group.add_argument("--patch-num", type=int, default=None, help="Override dataset.patch_num.")
        group.add_argument("--dataset-task", type=str, default=None, help="Override dataset.task.")
        group.add_argument("--num-classes", type=int, default=None, help="Override dataset.num_classes.")
        group.add_argument("--monitor-key", type=str, default=None, help="Override dataset.monitor_key.")
        group.add_argument("--scale", type=float, default=None, help="Override dataset.scale.")
        group.add_argument("--seq-len", type=int, default=None, help="Override dataset.seq_len.")

    @staticmethod
    def _add_runtime_args(parser):
        group = parser.add_argument_group("Runtime")
        group.add_argument(
            "--runtime-mode",
            type=str,
            choices=RUNTIME_MODES,
            default=None,
            help="Override runtime.mode.",
        )
        group.add_argument(
            "--runtime-backend",
            "--backend",
            dest="runtime_backend",
            type=str,
            choices=RUNTIME_BACKENDS,
            default=None,
            help="Override runtime.backend.",
        )
        group.add_argument(
            "--runtime-world-size",
            "--world-size",
            dest="runtime_world_size",
            type=int,
            default=None,
            help="Override runtime.world_size.",
        )
        group.add_argument(
            "--runtime-master-addr",
            "--master-addr",
            dest="runtime_master_addr",
            type=str,
            default=None,
            help="Override runtime.master_addr.",
        )
        group.add_argument(
            "--runtime-master-port",
            "--master-port",
            dest="runtime_master_port",
            type=int,
            default=None,
            help="Override runtime.master_port.",
        )
        group.add_argument(
            "--runtime-find-unused-parameters",
            type=parse_bool,
            default=None,
            help="Override runtime.find_unused_parameters.",
        )
        group.add_argument(
            "--runtime-timeout-seconds",
            type=int,
            default=None,
            help="Override runtime.timeout_seconds for distributed collectives.",
        )
        group.add_argument(
            "--runtime-debug",
            type=parse_bool,
            default=None,
            help="Override runtime.debug to enable rank-level distributed diagnostics.",
        )

    @staticmethod
    def _add_training_args(parser):
        group = parser.add_argument_group("Training")
        group.add_argument("--visualize", type=parse_bool, default=None, help="Override visualize.")
        group.add_argument("--batch-size", type=int, default=None, help="Override model.trainer.batch_size.")
        group.add_argument("--lr", type=float, default=None, help="Override model.trainer.lr.")
        group.add_argument("--lr-backbone", type=float, default=None, help="Override model.trainer.lr_backbone.")
        group.add_argument("--epochs", type=int, default=None, help="Override model.trainer.epochs.")
        group.add_argument("--patience", type=int, default=None, help="Override model.trainer.patience.")
        group.add_argument(
            "--clip-grad-norm",
            type=float,
            default=None,
            help="Override model.trainer.clip_grad_norm.",
        )
        group.add_argument("--weight-decay", type=float, default=None, help="Override model.trainer.weight_decay.")
        group.add_argument(
            "--label-smoothing",
            type=float,
            default=None,
            help="Override model.trainer.label_smoothing.",
        )
        group.add_argument(
            "--betas",
            type=float,
            nargs=2,
            default=None,
            metavar=("BETA1", "BETA2"),
            help="Override model.trainer.betas.",
        )
        group.add_argument("--eps", type=float, default=None, help="Override model.trainer.eps.")
        group.add_argument(
            "--save-every-epoch",
            type=parse_bool,
            default=None,
            help="Override model.trainer.save_every_epoch.",
        )

    @staticmethod
    def _add_finetune_model_args(parser):
        group = parser.add_argument_group("Finetune Model")
        group.add_argument(
            "--use-pretrained-weights",
            type=parse_bool,
            default=None,
            help="Override model.params.use_pretrained_weights.",
        )
        group.add_argument("--dropout", type=float, default=None, help="Override model.params.dropout.")
        group.add_argument("--foundation-dir", type=str, default=None, help="Override model.params.foundation_dir.")

    @staticmethod
    def _add_rescue_task_override_args(parser):
        group = parser.add_argument_group("Task-Specific Overrides")
        group.add_argument(
            "--use-pretrained-weights",
            type=parse_bool,
            default=None,
            help="Override model.params.use_pretrained_weights for finetune rescues.",
        )
        group.add_argument(
            "--dropout",
            type=float,
            default=None,
            help="Override task-specific dropout for rescued finetune, etms, or fuse runs.",
        )
        group.add_argument(
            "--foundation-dir",
            type=str,
            default=None,
            help="Override model.params.foundation_dir for finetune rescues.",
        )
        EmbraceCliBuilder._add_embrace_model_args(parser, include_dropout=False)

    @staticmethod
    def _add_embrace_model_args(parser, include_dropout=True):
        group = parser.add_argument_group("Embrace Model")
        group.add_argument(
            "--target-name",
            type=str,
            choices=SOURCE_MODEL_NAMES,
            default=None,
            help="Override model.target_name.",
        )
        group.add_argument(
            "--source-names",
            type=str,
            nargs="+",
            default=None,
            help="Override model.source_names.",
        )
        group.add_argument("--source-tag", type=str, default=None, help="Override model.source_tag.")
        group.add_argument("--proj-dim", type=int, default=None, help="Override model.proj_dim.")
        group.add_argument(
            "--use-shared-bridge",
            type=parse_bool,
            default=None,
            help="Override model.use_shared_bridge.",
        )
        group.add_argument(
            "--use-unified-bridge",
            type=parse_bool,
            default=None,
            help="Override model.use_unified_bridge.",
        )
        group.add_argument("--cbramod-mode", type=str, default=None, help="Override model.cbramod_mode.")
        group.add_argument("--csbrain-mode", type=str, default=None, help="Override model.csbrain_mode.")
        group.add_argument(
            "--coupled-weight",
            type=parse_bool,
            default=None,
            help="Override model.coupled_weight.",
        )
        group.add_argument("--fusor-mode", type=str, default=None, help="Override model.fusor_mode.")
        group.add_argument("--fusor-temp", type=float, default=None, help="Override model.fusor_temp.")
        group.add_argument("--attn-softmax", type=parse_bool, default=None, help="Override model.attn_softmax.")
        if include_dropout:
            group.add_argument("--dropout", type=float, default=None, help="Override model.dropout.")
        group.add_argument("--weight-ce", type=float, default=None, help="Override model.weight_ce.")
        group.add_argument("--weight-emb", type=float, default=None, help="Override model.weight_emb.")
        group.add_argument("--weight-attn", type=float, default=None, help="Override model.weight_attn.")
        group.add_argument(
            "--foundation-dir-cbramod",
            type=str,
            default=None,
            help="Override model.foundation_dir_dict.cbramod.",
        )
        group.add_argument(
            "--foundation-dir-labram",
            type=str,
            default=None,
            help="Override model.foundation_dir_dict.labram.",
        )
        group.add_argument(
            "--foundation-dir-codebrain",
            type=str,
            default=None,
            help="Override model.foundation_dir_dict.codebrain.",
        )
        group.add_argument(
            "--foundation-dir-csbrain",
            type=str,
            default=None,
            help="Override model.foundation_dir_dict.csbrain.",
        )
