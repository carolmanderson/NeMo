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

from typing import Dict, Optional, List

from nemo.collections.common.parts import MultiLayerPerceptron
from nemo.collections.nlp.modules.common.classifier import Classifier
from nemo.core.classes import typecheck
from nemo.core.neural_types import LogitsType, NeuralType

__all__ = ['MultiDomainSequenceTokenClassifier']


class MultiDomainSequenceTokenClassifier(Classifier):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "intent_logits": [NeuralType(('B', 'D'), LogitsType())],
            "slot_logits": [NeuralType(('B', 'T', 'D'), LogitsType())],
        }

    def __init__(
        self,
        hidden_size: int,
        num_domains: int,
        num_intents: List[int],
        num_slots: List[int],
        num_layers: int = 2,
        activation: str = 'relu',
        log_softmax: bool = False,
        dropout: float = 0.0,
        use_transformer_init: bool = True,
    ):
        """
        Initializes the SequenceTokenClassifier module, could be used for tasks that train sequence and
        token classifiers jointly, for example, for intent detection and slot tagging task.
        Args:
            hidden_size: hidden size of the mlp head on the top of the encoder
            num_domains: number of the domains to predict
            num_intents: list of the number of the intents to predict per domain (should have length equal to the number of domains)
            num_slots: list of number of the slots to predict per domain (should have length equal to the number of domains)
            num_layers: number of the linear layers of the mlp head on the top of the encoder
            activation: type of activations between layers of the mlp head
            log_softmax: applies the log softmax on the output
            dropout: the dropout used for the mlp head
            use_transformer_init: initializes the weights with the same approach used in Transformer
        """
        super().__init__(hidden_size=hidden_size, dropout=dropout)

        if not len(num_slots) == num_domains:
            raise Exception("Number of domains should be the same as the length of the list of number of slots per domain "
            "passed to MultiDomainSequenceTokenClassifier")

        if not len(num_intents) == num_domains:
            raise Exception("Number of domains should be the same as the length of the list of number of intents per domain "
            "passed to MultiDomainSequenceTokenClassifier")

        self.intent_mlps = []
        for i in range(num_domains):
            intent_mlp = MultiLayerPerceptron(
                hidden_size=hidden_size,
                num_classes=num_intents[i],
                num_layers=num_layers,
                activation=activation,
                log_softmax=log_softmax,
            )
            self.intent_mlps.append(intent_mlp)

        self.slot_mlps = []
        for i in range(num_domains):
            slot_mlp = MultiLayerPerceptron(
                hidden_size=hidden_size,
                num_classes=num_slots[i],
                num_layers=num_layers,
                activation=activation,
                log_softmax=log_softmax,
            )
            self.slot_mlps.append(slot_mlp)

        self.post_init(use_transformer_init=use_transformer_init)

    @typecheck()
    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        intent_logits = [intent_mlp(hidden_states[:, 0]) for intent_mlp in self.intent_mlps]
        slot_logits = [slot_mlp(hidden_states) for slot_mlp in self.slot_mlps]
        return intent_logits, slot_logits

