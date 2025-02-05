from data.nuscenes_pred_split import get_nuscenes_pred_split
import random, copy

from .preprocessor import preprocess
from .ethucy_split import get_ethucy_split
from lib.utils import print_log


class data_generator(object):
    # Constructor
    def __init__(self, parser, log, split='train', phase='training'):
        self.past_frames     = parser.past_frames
        self.min_past_frames = parser.min_past_frames
        self.frame_skip      = parser.get('frame_skip', 1)
        self.phase           = phase
        self.split           = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'
        # Check the name of the dataset
        if parser.dataset == 'nuscenes_pred':
            data_root = parser.data_root_nuscenes_pred
            seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
            self.init_frame = 0
        elif parser.dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:
            data_root = parser.data_root_ethucy
            seq_train, seq_val, seq_test = get_ethucy_split(parser.dataset)
            self.init_frame = 0
        elif parser.dataset == 'city_center':
            data_root = parser.data_root_city_center
            seq_train = []
            seq_val = []
            seq_test = ['gt_meters']
            self.init_frame = 0
        else:
            raise ValueError('Unknown dataset!')

        process_func   = preprocess
        self.data_root = data_root

        print_log("\n-------------------------- Loading %s data --------------------------" % split, log=log)
        if self.split == 'train':  self.sequence_to_load = seq_train
        elif self.split == 'val':  self.sequence_to_load = seq_val
        elif self.split == 'test': self.sequence_to_load = seq_test
        else:                      assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list   = []
        self.sequence          = []

        # Iterate over the sequences
        for seq_name in self.sequence_to_load:
            print_log("Loading sequence {} ...".format(seq_name), log=log)
            # Applying the preprocessing function
            preprocessed     = process_func(data_root, seq_name, parser, log, self.split, self.phase)
            num_seq_samples  = preprocessed.num_fr - (parser.min_past_frames + parser.min_future_frames - 1) * self.frame_skip
            self.num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessed)
        # Sample list is a list of consecutive indices
        self.sample_list = list(range(self.num_total_samples))
        self.index = 0
        print_log(f'Total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ Done --------------------------------\n", log=log)

    # Method to apply shuffling
    def shuffle(self):
        random.shuffle(self.sample_list)

    # Method to get the sequence index and frame index
    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    # Method to check if the epoch has ended
    def is_epoch_end(self):
        if self.index >= self.num_total_samples:
            self.index = 0      # reset
            return True
        else:
            return False

    # Method to get the next sample
    def next_sample(self):
        sample_index     = self.sample_list[self.index]
        # Get the sequence index and frame
        seq_index, frame = self.get_seq_and_frame(sample_index)
        # Get the sequence from self.sequence (which is an array of preprocessed sequences)
        seq              = self.sequence[seq_index]
        self.index      += 1
        # Get the data
        data = seq(frame)
        return data

    def __call__(self):
        return self.next_sample()
