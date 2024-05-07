# import the libraries
from kubeflow.training.api.training_client import TrainingClient
from kubeflow.storage_initializer.s3 import S3DatasetParams
from kubeflow.storage_initializer.hugging_face import (
    HuggingFaceModelParams,
    HuggingFaceTrainerParams,
    HuggingFaceDatasetParams,
)
from kubeflow.storage_initializer.constants import INIT_CONTAINER_MOUNT_PATH
from peft import LoraConfig
import transformers
from transformers import TrainingArguments
from kubeflow.training import constants
import os
# create a training client, pass config_file parameter if you want to use kubeconfig other than "~/.kube/config"
client = TrainingClient()

# mention the model, datasets and training parameters
client.train(
    name=os.environ.get('job_name', 'kubeflow finetune test'),
    num_workers=2,
    num_procs_per_worker=1,
    # specify the storage class if you don't want to use the default one for the storage-initializer PVC
    # storage_config={
    #     "size": "10Gi",
    #     "storage_class": "<your storage class>",
    # },
    model_provider_parameters=HuggingFaceModelParams(
        model_uri=os.environ.get("model_uri","hf://TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        transformer_type=transformers.AutoModelForCausalLM,
    ),
    # it is assumed for text related tasks, you have 'text' column in the dataset.
    # for more info on how dataset is loaded check load_and_preprocess_data function in sdk/python/kubeflow/trainer/hf_llm_training.py
    dataset_provider_parameters=HuggingFaceDatasetParams(repo_id=os.environ.get("dataset_repo","imdatta0/ultrachat_1k")),
    trainer_parameters=HuggingFaceTrainerParams(
        lora_config=LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        training_parameters=TrainingArguments(
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            },  # this is mandatory if checkpointng is enabled
            warmup_steps=0.02,
            learning_rate=1,
            lr_scheduler_type="cosine",
            bf16=False,
            logging_steps=0.01,
            output_dir=INIT_CONTAINER_MOUNT_PATH,
            optim=f"sgd",
            save_steps=0.01,
            save_total_limit=3,
            disable_tqdm=False,
            resume_from_checkpoint=True,
            remove_unused_columns=True,
            ddp_backend="nccl",  # change the backend to gloo if you want cpu based training and remove the gpu key in resources_per_worker
        ),
    ),
    resources_per_worker={
        "gpu": 1,
        "cpu": 2,
        "memory": "4Gi",
    },  # remove the gpu key if you don't want to attach gpus to the pods
)
# check the logs of the job
client.get_job_logs(name=os.environ.get('job_name', 'kubeflow finetune test'), job_kind=constants.PYTORCHJOB_KIND)