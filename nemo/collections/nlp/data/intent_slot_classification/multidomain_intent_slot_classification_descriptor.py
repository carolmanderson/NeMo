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

import itertools
from typing import List

from nemo.collections.nlp.data.data_utils.data_preprocessing import (
    fill_class_weights,
    get_freq_weights,
    get_label_stats,
    if_exist,
)
from nemo.utils import logging


class MultiDomainIntentSlotDataDesc:
    """ Convert the raw data to the standard format supported by
    IntentSlotDataDesc.

    By default, the None label for slots is 'O'.

    IntentSlotDataDesc requires two files:

        input_file: file to sequence + label.
            the first line is header (sentence [tab] label)
            each line should be [sentence][tab][label]

        slot_file: file to slot labels, each line corresponding to
            slot labels for a sentence in input_file. No header.

    To keep the mapping from label index to label consistent during
    training and inferencing we require the following files:
        dicts.intents.csv: each line is an intent. The first line
            corresponding to the 0 intent label, the second line
            corresponding to the 1 intent label, and so on.

        dicts.slots.csv: each line is a slot. The first line
            corresponding to the 0 slot label, the second line
            corresponding to the 1 slot label, and so on.

    Args:
        data_dir: the directory of the dataset
        modes: ['train', 'test', 'dev'],
        none_slot_label: the label for slots that aren't identified defaulted to 'O'
        pad_label: the int used for padding. If set to -1, it'll be set to the whatever the None label is.
    """

    def __init__(
        self,
        data_dir: str,
        modes: List[str] = ['train', 'test', 'dev'],
        none_slot_label: str = 'O',
        pad_label: int = -1,
    ):
        if not if_exist(data_dir, ['dict.intents.csv', 'dict.slots.csv', 'dict.domains.csv']):
            raise FileNotFoundError(
                "Make sure that your data follows the standard format "
                "supported by JointIntentSlotDataset. Your data must "
            "contain dict.domains.csv, dict.intents.csv, and dict.slots.csv." )

        self.data_dir = data_dir
        self.intent_dict_file = self.data_dir + '/dict.intents.csv'
        self.slot_dict_file = self.data_dir + '/dict.slots.csv'
        self.domain_dict_file = self.data_dir + '/dict.domains.csv'

        # get intents and slots for each domain.
        self.domains_label_ids = MultiDomainIntentSlotDataDesc.label2idx(self.domain_dict_file)

        num_domains = len(self.domains_label_ids)
        self.num_domains = num_domains
        self.domain_labels = list(range(num_domains))
        intent_map, slot_map = MultiDomainIntentSlotDataDesc.intent_slot_dicts_by_domain(self.data_dir, num_domains)
        self.intents_label_ids = []
        for i in range(num_domains):
            self.intents_label_ids.append(intent_map[i])
        self.num_intents = [len(intent_list) for intent_list in self.intents_label_ids]
        self.slots_label_ids = []
        for i in range(num_domains):
            self.slots_label_ids.append(slot_map[i])
        self.num_slots = [len(slot_list) for slot_list in self.slots_label_ids]

        infold = self.data_dir
        for mode in modes:
            if not if_exist(self.data_dir, [f'{mode}.tsv']):
                logging.info(f' Stats calculation for {mode} mode' f' is skipped as {mode}.tsv was not found.')
                continue
            logging.info(f' Stats calculating for {mode} mode...')
            slot_file = f'{self.data_dir}/{mode}_slots.tsv'
            with open(slot_file, 'r') as f:
                slot_lines = f.readlines()

            input_file = f'{self.data_dir}/{mode}.tsv'
            with open(input_file, 'r') as f:
                input_lines = f.readlines()[1:]  # Skipping headers at index 0

            if len(slot_lines) != len(input_lines):
                raise ValueError(
                    "Make sure that the number of slot lines match the "
                    "number of intent lines. There should be a 1-1 "
                    "correspondence between every slot and intent lines."
                )

            dataset = list(zip(slot_lines, input_lines))

            raw_slots, raw_intents, raw_domains = [], [], []
            slots_by_domain, intents_by_domain = {i: [] for i in range(num_domains)}, {i: [] for i in range(num_domains)}
            for slot_line, input_line in dataset:
                slot_list = [int(slot) for slot in slot_line.strip().split()]
                raw_slots.append(slot_list)
                parts = input_line.strip().split()
                domain = int(parts[-1])
                intent = int(parts[-2])
                raw_intents.append(intent)
                raw_domains.append(domain)
                slots_by_domain[domain].append(slot_list)
                intents_by_domain[domain].append(intent)

            slots_label_freqs = []
            slots_total = []
            for i in range(num_domains):
                logging.info(f'For domain {i}, three most popular slots in {mode} mode:')
                slots_to_count = itertools.chain.from_iterable(slots_by_domain[i])
                num_slots, slot_label_freq, max_id = get_label_stats(
                    slots_to_count, infold + f'/domain_{i}_{mode}_slot_stats.tsv'
                )
                slots_label_freqs.append(slot_label_freq)
                slots_total.append(num_slots)

            intent_label_freqs = []
            total_intents = []
            for i in range(num_domains):
                logging.info(f'For domain {i}, three most popular intents in {mode} mode:')
                num_intents, intent_label_freq, max_id = get_label_stats(
                    intents_by_domain[i], infold + f'/domain_{i}_{mode}_intent_stats.tsv'
                )
                intent_label_freqs.append(intent_label_freq)
                total_intents.append(num_intents)

            logging.info(f'Total Number of Intents: {total_intents}')
            logging.info(f'Intent Label Frequencies: {intent_label_freqs}')
            logging.info(f'Total Number of Slots: {slots_total}')
            logging.info(f'Slots Label Frequencies: {slots_label_freqs}')

            if mode == 'train':
                intent_weights_dicts = [get_freq_weights(intent_label_freq) for intent_label_freq in intent_label_freqs]
                logging.info(f'Intent Weights: {intent_weights_dicts}')
                slot_weights_dicts = [get_freq_weights(slots_label_freq) for slots_label_freq in slots_label_freqs]
                logging.info(f'Slot Weights: {slot_weights_dicts}')

        self.intent_weights = [fill_class_weights(intent_weights_dict, self.num_intents[i] - 1) for i, intent_weights_dict in enumerate(intent_weights_dicts)]
        self.slot_weights = [fill_class_weights(slot_weights_dict, self.num_slots[i] - 1) for i, slot_weights_dict in enumerate(slot_weights_dicts)]

        if pad_label != -1:
            self.pad_label = pad_label
        else:
            # TODO: assuming pad label is the same for all domains here; add check or change so it can be different for each domain
            for label_dict in self.slots_label_ids:
                if none_slot_label not in label_dict.keys():
                    raise ValueError(f'none_slot_label {none_slot_label} not ' f'found in {self.slot_dict_file}.')
            self.pad_label = self.slots_label_ids[0][none_slot_label]

    @staticmethod
    def label2idx(file):
        lines = open(file, 'r').readlines()
        lines = [line.strip() for line in lines if line.strip()]
        labels = {lines[i]: i for i in range(len(lines))}
        return labels

    @staticmethod
    def intent_slot_dicts_by_domain(data_dir, num_domains):
        '''
        Return Intent and slot dictionaries
        '''
        intents_by_domain = {i: {} for i in range(num_domains)}
        slots_by_domain = {i: {} for i in range(num_domains)}

        intent_dict_file = data_dir + '/dict.intents.csv'
        slot_dict_file = data_dir + '/dict.slots.csv'

        with open(slot_dict_file, "r") as slot_file:
            for line in slot_file.readlines()[1:]:
                domain_name, domain_index, slot_name, slot_index = line.strip().split()
                slots_by_domain[int(domain_index)][slot_name] = int(slot_index)

        with open(intent_dict_file, "r") as intent_file:
            for line in intent_file.readlines()[1:]:
                domain_name, domain_index, intent_name, intent_index = line.strip().split()
                intents_by_domain[int(domain_index)][intent_name] = int(intent_index)

        return intents_by_domain, slots_by_domain
