# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Dict, List, Optional

import onnx
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
from nemo.collections.nlp.data.domain_intent_slot_classification import (
    DomainIntentSlotClassificationDataset,
    DomainIntentSlotDataDesc,
    DomainIntentSlotInferenceDataset,
)
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common import DomainIntentTokenClassifier
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes import typecheck
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types import NeuralType
from nemo.utils import logging


class DomainIntentSlotClassificationModel(NLPModel):
    """
    domain_cls_strategy options:
    shared: uses the CLS token for both intent and domain classification
    CLS2_random: adds a [CLS2] token after [CLS], uses [CLS] for intents and [CLS2] for domains; [CLS2] is randomly initialized
    CLS2_from_CLS: similar to CLS2_random, but initializes the the CLS2 embedding with a copy of the CLS embedding

    """
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.classifier.output_types

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """ Initializes BERT Joint Intent and Slot model.
        """
        self.max_seq_length = cfg.language_model.max_seq_length

        # Setup tokenizer.
        self.setup_tokenizer(cfg.tokenizer)

        strategy = cfg.domain_cls_strategy

        if strategy == "shared":
            self.extra_cls_token = False
        elif strategy in ["CLS2_random", "CLS2_from_CLS"]:
            special_tokens = {'additional_special_tokens': ['[CLS2]']}
            self.tokenizer.add_special_tokens(special_tokens_dict=special_tokens)
            self.extra_cls_token = True
        else:
            raise Exception('domain_cls_strategy must be one of: "shared", "CLS2_random", "CLS2_from_CLS"')

        # Check the presence of data_dir.
        if not cfg.data_dir or not os.path.exists(cfg.data_dir):
            # Disable setup methods.
            DomainIntentSlotClassificationModel._set_model_restore_state(is_being_restored=True)
            # Set default values of data_desc.
            self._set_defaults_data_desc(cfg)
        else:
            self.data_dir = cfg.data_dir
            # Update configuration of data_desc.
            self._set_data_desc_to_cfg(cfg, cfg.data_dir, cfg.train_ds, cfg.validation_ds)

        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        # Enable setup methods.
        DomainIntentSlotClassificationModel._set_model_restore_state(is_being_restored=False)

        # Initialize Bert model
        self.bert_model = get_lm_model(
            pretrained_model_name=self.cfg.language_model.pretrained_model_name,
            config_file=self.cfg.language_model.config_file,
            config_dict=OmegaConf.to_container(self.cfg.language_model.config)
            if self.cfg.language_model.config
            else None,
            checkpoint_file=self.cfg.language_model.lm_checkpoint,
            vocab_file=cfg.tokenizer.vocab_file,
        )

        # Initialize the [CLS2] token embedding using the pretrained [CLS] embedding or a random embedding.
        # But only do this when starting to train a new model; not when reloading a saved model.
        if self.tokenizer.vocab_size > self.bert_model.embeddings.word_embeddings.weight.shape[0]:
            if strategy in ["CLS2_random", "CLS2_from_CLS"]:
                # resize bert model to account for extra [CLS2] token
                self.bert_model.resize_token_embeddings(self.tokenizer.vocab_size)

            if strategy == "CLS2_from_CLS":
                input_embs = self.bert_model.get_input_embeddings()
                with torch.no_grad():
                    input_embs.weight[-1] = input_embs.weight[self.tokenizer.cls_id]
                self.bert_model.set_input_embeddings(input_embs)

        # Initialize Classifier.
        self._reconfigure_classifier()

    def _set_defaults_data_desc(self, cfg):
        """
        Method makes sure that cfg.data_desc params are set.
        If not, set's them to "dummy" defaults.
        """
        if not hasattr(cfg, "data_desc"):
            OmegaConf.set_struct(cfg, False)
            cfg.data_desc = {}
            # Domains.
            cfg.data_desc.domain_labels = " "
            cfg.data_desc.domain_label_ids = {" ": 0}
            cfg.data_desc.domain_weights = [1]
            # Intents.
            cfg.data_desc.intent_labels = " "
            cfg.data_desc.intent_label_ids = {" ": 0}
            cfg.data_desc.intent_weights = [1]
            # Slots.
            cfg.data_desc.slot_labels = " "
            cfg.data_desc.slot_label_ids = {" ": 0}
            cfg.data_desc.slot_weights = [1]

            cfg.data_desc.pad_label = "O"
            OmegaConf.set_struct(cfg, True)


    def _set_data_desc_to_cfg(self, cfg, data_dir, train_ds, validation_ds):
        """ Method creates IntentSlotDataDesc and copies generated values to cfg.data_desc. """
        # Save data from data desc to config - so it can be reused later, e.g. in inference.
        data_desc = DomainIntentSlotDataDesc(data_dir=data_dir, modes=[train_ds.prefix, validation_ds.prefix])
        OmegaConf.set_struct(cfg, False)
        if not hasattr(cfg, "data_desc") or cfg.data_desc is None:
            cfg.data_desc = {}
        # Domains.
        cfg.data_desc.domain_labels = list(data_desc.domains_label_ids.keys())
        cfg.data_desc.domain_label_ids = data_desc.domains_label_ids
        cfg.data_desc.domain_weights = data_desc.domain_weights
        # Intents.
        cfg.data_desc.intent_labels = list(data_desc.intents_label_ids.keys())
        cfg.data_desc.intent_label_ids = data_desc.intents_label_ids
        cfg.data_desc.intent_weights = data_desc.intent_weights
        # Slots.
        cfg.data_desc.slot_labels = list(data_desc.slots_label_ids.keys())
        cfg.data_desc.slot_label_ids = data_desc.slots_label_ids
        cfg.data_desc.slot_weights = data_desc.slot_weights

        cfg.data_desc.pad_label = data_desc.pad_label

        # for older(pre - 1.0.0.b3) configs compatibility
        if not hasattr(cfg, "class_labels") or cfg.class_labels is None:
            cfg.class_labels = {}
            cfg.class_labels = OmegaConf.create(
                {'domain_labels_file': 'domain_labels.csv', 'intent_labels_file': 'intent_labels.csv', 'slot_labels_file': 'slot_labels.csv'}
            )

        slot_labels_file = os.path.join(data_dir, cfg.class_labels.slot_labels_file)
        intent_labels_file = os.path.join(data_dir, cfg.class_labels.intent_labels_file)
        domain_labels_file = os.path.join(data_dir, cfg.class_labels.domain_labels_file)
        self._save_label_ids(data_desc.slots_label_ids, slot_labels_file)
        self._save_label_ids(data_desc.intents_label_ids, intent_labels_file)
        self._save_label_ids(data_desc.domains_label_ids, domain_labels_file)

        self.register_artifact(cfg.class_labels.domain_labels_file, domain_labels_file)
        self.register_artifact(cfg.class_labels.intent_labels_file, intent_labels_file)
        self.register_artifact(cfg.class_labels.slot_labels_file, slot_labels_file)
        OmegaConf.set_struct(cfg, True)

    def _save_label_ids(self, label_ids: Dict[str, int], filename: str) -> None:
        """ Saves label ids map to a file """
        with open(filename, 'w') as out:
            labels, _ = zip(*sorted(label_ids.items(), key=lambda x: x[1]))
            out.write('\n'.join(labels))
            logging.info(f'Labels: {label_ids}')
            logging.info(f'Labels mapping saved to : {out.name}')

    def _reconfigure_classifier(self):
        """ Method reconfigures the classifier depending on the settings of model cfg.data_desc """

        self.classifier = DomainIntentTokenClassifier(
            hidden_size=self.bert_model.config.hidden_size,
            num_domains=len(self.cfg.data_desc.domain_labels),
            num_intents=len(self.cfg.data_desc.intent_labels),
            num_slots=len(self.cfg.data_desc.slot_labels),
            dropout=self.cfg.head.fc_dropout,
            num_layers=self.cfg.head.num_output_layers,
            log_softmax=False,
            extra_cls_token=self.extra_cls_token
        )

        # define losses
        if self.cfg.class_balancing == 'weighted_loss':
            # You may need to increase the number of epochs for convergence when using weighted_loss
            self.domain_loss = CrossEntropyLoss(logits_ndim=2, weight=self.cfg.data_desc.domain_weights)
            self.intent_loss = CrossEntropyLoss(logits_ndim=2, weight=self.cfg.data_desc.intent_weights)
            self.slot_loss = CrossEntropyLoss(logits_ndim=3, weight=self.cfg.data_desc.slot_weights)
        else:
            self.domain_loss = CrossEntropyLoss(logits_ndim=2)
            self.intent_loss = CrossEntropyLoss(logits_ndim=2)
            self.slot_loss = CrossEntropyLoss(logits_ndim=3)

        self.total_loss = AggregatorLoss(
            num_inputs=3, weights=[self.cfg.domain_loss_weight, self.cfg.intent_loss_weight,
                                   1.0 - (self.cfg.domain_loss_weight + self.cfg.intent_loss_weight)]
        )

        # setup to track metrics
        self.domain_classification_report = ClassificationReport(
            num_classes=len(self.cfg.data_desc.domain_labels),
            label_ids=self.cfg.data_desc.domain_label_ids,
            dist_sync_on_step=True,
            mode='micro',
        )
        self.intent_classification_report = ClassificationReport(
            num_classes=len(self.cfg.data_desc.intent_labels),
            label_ids=self.cfg.data_desc.intent_label_ids,
            dist_sync_on_step=True,
            mode='micro',
        )
        self.slot_classification_report = ClassificationReport(
            num_classes=len(self.cfg.data_desc.slot_labels),
            label_ids=self.cfg.data_desc.slot_label_ids,
            dist_sync_on_step=True,
            mode='micro',
        )

    def update_data_dir_for_training(self, data_dir: str, train_ds, validation_ds) -> None:
        """
        Update data directory and get data stats with Data Descriptor.
        Also, reconfigures the classifier - to cope with data with e.g. different number of slots.

        Args:
            data_dir: path to data directory
        """
        logging.info(f'Setting data_dir to {data_dir}.')
        self.data_dir = data_dir
        # Update configuration with new data.
        self._set_data_desc_to_cfg(self.cfg, data_dir, train_ds, validation_ds)
        # Reconfigure the classifier for different settings (number of intents, slots etc.).
        self._reconfigure_classifier()

    def update_data_dir_for_testing(self, data_dir) -> None:
        """
        Update data directory.

        Args:
            data_dir: path to data directory
        """
        logging.info(f'Setting data_dir to {data_dir}.')
        self.data_dir = data_dir

    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        domain_logits, intent_logits, slot_logits = self.classifier(hidden_states=hidden_states)
        return domain_logits, intent_logits, slot_logits

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, domain_labels, intent_labels, slot_labels = batch
        domain_logits, intent_logits, slot_logits = self(
            input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask
        )

        # calculate combined loss for intents and slots
        domain_loss = self.domain_loss(logits=domain_logits, labels=domain_labels)
        intent_loss = self.intent_loss(logits=intent_logits, labels=intent_labels)
        slot_loss = self.slot_loss(logits=slot_logits, labels=slot_labels, loss_mask=loss_mask)
        train_loss = self.total_loss(loss_1=domain_loss, loss_2=intent_loss, loss_3=slot_loss)
        lr = self._optimizer.param_groups[0]['lr']

        self.log('train_loss', train_loss)
        self.log('lr', lr, prog_bar=True)

        return {
            'loss': train_loss,
            'lr': lr,
        }

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, domain_labels, intent_labels, slot_labels = batch
        domain_logits, intent_logits, slot_logits = self(
            input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask
        )

        # calculate combined loss for intents and slots
        domain_loss = self.domain_loss(logits=domain_logits, labels=domain_labels)
        intent_loss = self.intent_loss(logits=intent_logits, labels=intent_labels)
        slot_loss = self.slot_loss(logits=slot_logits, labels=slot_labels, loss_mask=loss_mask)
        val_loss = self.total_loss(loss_1=domain_loss, loss_2=intent_loss, loss_3=slot_loss)

        # calculate accuracy metrics for intents and slot reporting
        # domains
        preds = torch.argmax(domain_logits, axis=-1)
        self.domain_classification_report.update(preds, domain_labels)
        # intents
        preds = torch.argmax(intent_logits, axis=-1)
        self.intent_classification_report.update(preds, intent_labels)
        # slots
        subtokens_mask = subtokens_mask > 0.5
        preds = torch.argmax(slot_logits, axis=-1)[subtokens_mask]
        slot_labels = slot_labels[subtokens_mask]
        self.slot_classification_report.update(preds, slot_labels)

        return {
            'val_loss': val_loss,
            'domain_tp': self.domain_classification_report.tp,
            'domain_fn': self.domain_classification_report.fn,
            'domain_fp': self.domain_classification_report.fp,
            'intent_tp': self.intent_classification_report.tp,
            'intent_fn': self.intent_classification_report.fn,
            'intent_fp': self.intent_classification_report.fp,
            'slot_tp': self.slot_classification_report.tp,
            'slot_fn': self.slot_classification_report.fn,
            'slot_fp': self.slot_classification_report.fp,
        }

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # calculate metrics and log classification report (separately for intents and slots)
        domain_precision, domain_recall, domain_f1, domain_report = self.domain_classification_report.compute()
        logging.info(f'Domain report: {domain_report}')

        intent_precision, intent_recall, intent_f1, intent_report = self.intent_classification_report.compute()
        logging.info(f'Intent report: {intent_report}')

        slot_precision, slot_recall, slot_f1, slot_report = self.slot_classification_report.compute()
        logging.info(f'Slot report: {slot_report}')

        self.log('val_loss', avg_loss)
        self.log('domain_precision', domain_precision)
        self.log('domain_recall', domain_recall)
        self.log('domain_f1', domain_f1)
        self.log('intent_precision', intent_precision)
        self.log('intent_recall', intent_recall)
        self.log('intent_f1', intent_f1)
        self.log('slot_precision', slot_precision)
        self.log('slot_recall', slot_recall)
        self.log('slot_f1', slot_f1)

        self.intent_classification_report.reset()
        self.slot_classification_report.reset()
        self.domain_classification_report.reset()

        return {
            'val_loss': avg_loss,
            'domain_precision': domain_precision,
            'domain_recall': domain_recall,
            'domain_f1': domain_f1,
            'intent_precision': intent_precision,
            'intent_recall': intent_recall,
            'intent_f1': intent_f1,
            'slot_precision': slot_precision,
            'slot_recall': slot_recall,
            'slot_f1': slot_f1,
        }

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        """
        Called at the end of test to aggregate outputs.
        :param outputs: list of individual outputs of each test step.
        """
        return self.validation_epoch_end(outputs)

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        input_file = f'{self.data_dir}/{cfg.prefix}.tsv'
        slot_file = f'{self.data_dir}/{cfg.prefix}_slots.tsv'

        if not (os.path.exists(input_file) and os.path.exists(slot_file)):
            raise FileNotFoundError(
                f'{input_file} or {slot_file} not found. Please refer to the documentation for the right format \
                 of Intents and Slots files.'
            )

        dataset = DomainIntentSlotClassificationDataset(
            input_file=input_file,
            slot_file=slot_file,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            num_samples=cfg.num_samples,
            pad_label=self.cfg.data_desc.pad_label,
            ignore_extra_tokens=self.cfg.ignore_extra_tokens,
            ignore_start_end=self.cfg.ignore_start_end,
            extra_cls_token=self.extra_cls_token
        )

        return DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last,
            collate_fn=dataset.collate_fn,
        )

    def _setup_infer_dataloader(self, queries: List[str], test_ds) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a infer data loader.
        Args:
            queries: text
            batch_size: batch size to use during inference
        Returns:
            A pytorch DataLoader.
        """

        dataset = DomainIntentSlotInferenceDataset(
            tokenizer=self.tokenizer, queries=queries, max_seq_length=-1, do_lower_case=False, extra_cls_token=self.extra_cls_token
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=test_ds.batch_size,
            shuffle=test_ds.shuffle,
            num_workers=test_ds.num_workers,
            pin_memory=test_ds.pin_memory,
            drop_last=test_ds.drop_last,
        )

    def predict_from_examples(self, queries: List[str], test_ds) -> List[List[str]]:
        """
        Get prediction for the queries (intent and slots)
        Args:
            queries: text sequences
            test_ds: Dataset configuration section.
        Returns:
            predicted_domains, predicted_intents, predicted_slots: model intent and slot predictions
        """
        predicted_domains = []
        predicted_intents = []
        predicted_slots = []
        mode = self.training

        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Retrieve intent and slot vocabularies from configuration.
            domain_labels = self.cfg.data_desc.domain_labels
            intent_labels = self.cfg.data_desc.intent_labels
            slot_labels = self.cfg.data_desc.slot_labels

            # Initialize tokenizer.
            # if not hasattr(self, "tokenizer"):
            #    self._setup_tokenizer(self.cfg.tokenizer)
            # Initialize modules.
            # self._reconfigure_classifier()

            # Switch model to evaluation mode
            self.eval()
            self.to(device)

            # Dataset.
            infer_datalayer = self._setup_infer_dataloader(queries, test_ds)

            for batch in infer_datalayer:
                input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask = batch


                domain_logits, intent_logits, slot_logits = self.forward(
                    input_ids=input_ids.to(device),
                    token_type_ids=input_type_ids.to(device),
                    attention_mask=input_mask.to(device),
                )
                # predict intents and slots for these examples
                # domains
                domain_preds = tensor2list(torch.argmax(domain_logits, axis=-1))
                for domain_num in domain_preds:
                    if domain_num < len(domain_labels):
                        predicted_domains.append(domain_labels[int(domain_num)])
                    else:
                        # should not happen
                        predicted_domains.append("Unknown Domain")

                # intents
                intent_preds = tensor2list(torch.argmax(intent_logits, axis=-1))

                # convert numerical outputs to Intent and Slot labels from the dictionaries
                for intent_num in intent_preds:
                    if intent_num < len(intent_labels):
                        predicted_intents.append(intent_labels[int(intent_num)])
                    else:
                        # should not happen
                        predicted_intents.append("Unknown Intent")

                # slots
                slot_preds = torch.argmax(slot_logits, axis=-1)

                for slot_preds_query, mask_query in zip(slot_preds, subtokens_mask):
                    query_slots = ''
                    for slot, mask in zip(slot_preds_query, mask_query):
                        if mask == 1:
                            if slot < len(slot_labels):
                                query_slots += slot_labels[int(slot)] + ' '
                            else:
                                query_slots += 'Unknown_slot '
                    predicted_slots.append(query_slots.strip())

        finally:
            # set mode back to its original value
            self.train(mode=mode)

        return predicted_domains, predicted_intents, predicted_slots


    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method must be implemented, but there are no available pretrained models, so it returns None.

        Returns:
            List of available pre-trained models.
        """

        return None


if __name__ == "__main__":
    import os
    import time

    from nemo.collections import nlp as nemo_nlp
    from nemo.utils.exp_manager import exp_manager
    from nemo.utils import logging

    import torch
    from omegaconf import OmegaConf
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.loggers import WandbLogger
    import wandb

    '''
    ############## SWEEP ##################################################
    wandb.login()

    sweep_config = {
        'method': 'grid',
        'metric': {
            "name": 'train_loss',
            'goal': "minimize"
        },
        'parameters': {
            "domain_loss_weight": {
                "values": [0.05, 0.1, 0.2, 0.3]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="nvcc", entity="carola")

    def train():
        HOME_DIR = "/home/carola/Documents"

        # directory with data converted to nemo format
        data_dir = os.path.join(HOME_DIR, "data/domain_merging/combined_domains/merged_with_domain")

        # config
        config_file = os.path.join(HOME_DIR,
                                   "configs/joint_domain_intent_slot/domain_intent_slot_classification_new_hyperparams.yaml")
        config = OmegaConf.load(config_file)
        config.trainer.max_epochs = 2
        config.model.data_dir = data_dir
        config.model.validation_ds.prefix = "dev"
        config.model.test_ds.prefix = "dev"
        # config.model.intent_loss_weight = 0.5
        # config.model.domain_loss_weight = 0.05
        config.model.class_balancing = "weighted_loss"
        config.trainer.val_check_interval = 100

        cuda = 1 if torch.cuda.is_available() else 0
        config.trainer.gpus = cuda
        config.trainer.precision = 16 if torch.cuda.is_available() else 32

        # for mixed precision training, uncomment the line below (precision should be set to 16 and amp_level to O1):
        # config.trainer.amp_level = O1

        # remove distributed training flags
        config.trainer.accelerator = None
        
        early_stop_callback = EarlyStopping(monitor='intent_f1', min_delta=1e-1, patience=10, verbose=True, mode='max')
        trainer = pl.Trainer(callbacks=[early_stop_callback], **config.trainer)
        config.exp_manager.create_wandb_logger = True
        config.exp_manager.wandb_logger_kwargs = {"name": "sweeps test",
                                                  "project": "nvcc", "entity": "carola"}
        config.exp_manager.version = time.strftime('%Y-%m-%d_%H-%M-%S')
        config.exp_manager.exp_dir = os.path.join(HOME_DIR, "output/")
        exp_dir = exp_manager(trainer, config.get("exp_manager", None))
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                run = logger.experiment
                break
        wandb_config = run.config
        domain_loss_weight = wandb_config.domain_loss_weight
        config.model.domain_loss_weight = domain_loss_weight
        config.model.intent_loss_weight = 0.6-domain_loss_weight
        config.model.domain_cls_strategy = 'CLS2_from_CLS'
        config.model.language_model.pretrained_model_name = "distilbert-base-uncased"
        model = nemo_nlp.models.DomainIntentSlotClassificationModel(config.model, trainer=trainer)
        trainer.fit(model)
        wandb.finish()


    wandb.agent(sweep_id, train, count=4)

    '''
    ###################### END OF SWEEP ########################################################

    HOME_DIR = "/home/carola/Documents"

    # directory with data converted to nemo format
    data_dir = os.path.join(HOME_DIR, "data/domain_merging/combined_domains/merged_with_domain")

    # config
    config_file = os.path.join(HOME_DIR,
                               "configs/joint_domain_intent_slot/domain_intent_slot_classification_new_hyperparams.yaml")
    config = OmegaConf.load(config_file)
    config.trainer.max_epochs = 1
    config.model.data_dir = data_dir
    config.model.validation_ds.prefix = "dev"
    config.model.test_ds.prefix = "dev"
    config.model.intent_loss_weight = 0.5
    config.model.domain_loss_weight = 0.05
    config.model.class_balancing = "weighted_loss"
    # config.trainer.val_check_interval = 100

    cuda = 1 if torch.cuda.is_available() else 0
    config.trainer.gpus = cuda
    config.trainer.precision = 16 if torch.cuda.is_available() else 32

    # for mixed precision training, uncomment the line below (precision should be set to 16 and amp_level to O1):
    # config.trainer.amp_level = O1

    # remove distributed training flags
    config.trainer.accelerator = None

    # trainer = pl.Trainer(callbacks=[early_stop_callback], **config.trainer)
    trainer = pl.Trainer(**config.trainer)

    config.exp_manager.create_wandb_logger = True
    config.exp_manager.wandb_logger_kwargs = {"name": "domain_intent_slot_test_4",
                                              "project": "nvcc", "entity": "carola"}
    config.exp_manager.version = time.strftime('%Y-%m-%d_%H-%M-%S')
    config.exp_manager.exp_dir = os.path.join(HOME_DIR, "output/")
    exp_dir = exp_manager(trainer, config.get("exp_manager", None))

    # config.model.domain_cls_strategy = wandb_config.domain_cls_strategy  # options: shared, CLS2_random, CLS2_from_CLS
    config.model.domain_cls_strategy = 'CLS2_from_CLS'
    config.model.language_model.pretrained_model_name = "distilbert-base-uncased"

    model = nemo_nlp.models.DomainIntentSlotClassificationModel(config.model, trainer=trainer)

    trainer.fit(model)

    wandb.finish()

    # c = "/home/carola/Documents/output/IntentSlot/2021-05-18_11-48-36/checkpoints/IntentSlot--intent_f1=94.00-epoch=1.ckpt"  # new, distilbert, cls2 from cls; 98.25
    # infer_model = nemo_nlp.models.DomainIntentSlotClassificationModel.load_from_checkpoint(checkpoint_path=c)
