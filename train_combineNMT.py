import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn

from seq2seq import models, utils
from seq2seq.data.dictionary import Dictionary
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY

def get_args():
    """ Defines training-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', action='store_true', help='Use a GPU')

    # Add data arguments
    parser.add_argument('--data', default='indomain/preprocessed_data/', help='path to data directory')
    parser.add_argument('--source-lang', default='fr', help='source language')
    parser.add_argument('--target-lang', default='en', help='target language')
    parser.add_argument('--max-tokens', default=None, type=int, help='maximum number of tokens in a batch')
    parser.add_argument('--batch-size', default=500, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--train-on-tiny', action='store_true', help='train model on a tiny dataset')

    # autoencoding loss weight
    parser.add_argument('--ae-loss-weight', default=0.5, type=float, help='Weight for the autoencoding loss')

    # Add model arguments
    parser.add_argument('--arch', default='lstm', choices=ARCH_MODEL_REGISTRY.keys(), help='model architecture')

    # Add optimization arguments
    parser.add_argument('--max-epoch', default=10000, type=int, help='force stop training at specified epoch')
    parser.add_argument('--clip-norm', default=4.0, type=float, help='clip threshold of gradients')
    parser.add_argument('--lr', default=0.003, type=float, help='learning rate')
    parser.add_argument('--patience', default=5, type=int,
                        help='number of epochs without improvement on validation set before early stopping')

    # Add checkpoint arguments
    parser.add_argument('--log-file', default=None, help='path to save logs')
    parser.add_argument('--save-dir', default='checkpoints_asg4', help='path to save checkpoints')
    parser.add_argument('--restore-file', default='checkpoint_last.pt', help='filename to load checkpoint')
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    parser.add_argument('--epoch-checkpoints', action='store_true', help='store all epoch checkpoints')

    # Parse twice as model arguments are not known the first time
    args, _ = parser.parse_known_args()
    model_parser = parser.add_argument_group(argument_default=argparse.SUPPRESS)
    ARCH_MODEL_REGISTRY[args.arch].add_args(model_parser)
    args = parser.parse_args()
    ARCH_CONFIG_REGISTRY[args.arch](args)
    return args


def main(args):
    """ Main training function. Trains the translation model over the course of several epochs, including dynamic
    learning rate adjustment and gradient clipping. """

    logging.info('Commencing training!')
    torch.manual_seed(42)

    utils.init_logging(args)

    # Load dictionaries
    src_dict = Dictionary.load(os.path.join(args.data, 'dict.{:s}'.format(args.source_lang)))
    logging.info('Loaded a source dictionary ({:s}) with {:d} words'.format(args.source_lang, len(src_dict)))
    tgt_dict = Dictionary.load(os.path.join(args.data, 'dict.{:s}'.format(args.target_lang)))
    logging.info('Loaded a target dictionary ({:s}) with {:d} words'.format(args.target_lang, len(tgt_dict)))

    # Load datasets
    def load_data(split):
        return Seq2SeqDataset(
            src_file=os.path.join(args.data, '{:s}.{:s}'.format(split, args.source_lang)),
            tgt_file=os.path.join(args.data, '{:s}.{:s}'.format(split, args.target_lang)),
            src_dict=src_dict, tgt_dict=tgt_dict)

    train_dataset = load_data(split='train') if not args.train_on_tiny else load_data(split='tiny_train')
    valid_dataset = load_data(split='valid')

    # Build model and optimization criterion
    model = models.build_model(args, src_dict, tgt_dict)
    logging.info('Built a model with {:d} parameters'.format(sum(p.numel() for p in model.parameters())))
    #criterion = nn.CrossEntropyLoss(ignore_index=src_dict.pad_idx, reduction='sum')

    # Define two separate loss functions: one for translation and one for autoencoding
    translation_criterion = nn.CrossEntropyLoss(ignore_index=src_dict.pad_idx, reduction='sum')
    autoencoding_criterion = nn.CrossEntropyLoss(ignore_index=tgt_dict.pad_idx, reduction='sum')

    # to validate translation loss
    criterion = translation_criterion

    if args.cuda:
        model = model.cuda()
        translation_criterion = translation_criterion.cuda()
        autoencoding_criterion = autoencoding_criterion.cuda()

    # Instantiate optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    #print("optimizer1", optimizer)

    # Load last checkpoint if one exists
    state_dict = utils.load_checkpoint(args, model, optimizer)  # lr_scheduler
    last_epoch = state_dict['last_epoch'] if state_dict is not None else -1
    #print("optimizer2", optimizer)

    # Initialize ReduceLROnPlateau learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True, threshold=0.0001) # threshold

    # print("args", args)
    # Arguments: {'cuda': False, 'data': 'data/en-fr/prepared', 'source_lang': 'fr', 'target_lang': 'en', 'max_tokens': None, 'batch_size': 500, 'train_on_tiny': False, 'ae_loss_weight': 0.5, 'arch': 'lstm', 'max_epoch': 10000, 'clip_norm': 4.0, 'lr': 0.003, 'patience': 3, 'log_file': None, 'save_dir': 'assignments/03/baseline/checkpoints_combineNMT_lr0.003', 'restore_file': 'checkpoint_last.pt', 'save_interval': 1, 'no_save': False, 'epoch_checkpoints': False, 'encoder_embed_dim': 64, 'encoder_embed_path': None, 'encoder_hidden_size': 64, 'encoder_num_layers': 1, 'encoder_bidirectional': 'True', 'encoder_dropout_in': 0.25, 'encoder_dropout_out': 0.25, 'decoder_embed_dim': 64, 'decoder_embed_path': None, 'decoder_hidden_size': 128, 'decoder_num_layers': 1, 'decoder_dropout_in': 0.25, 'decoder_dropout_out': 0.25, 'decoder_use_attention': 'True', 'decoder_use_lexical_model': 'False', 'device_id': 0}
    # for param_group in optimizer.param_groups: # reset lr to the assigned parameter
    #     param_group['lr'] = args.lr
        # param_group['ae-loss-weight'] = args.ae_loss_weight * (0.99 ** epoch)
    # print(state_dict)
   # print("optimizer3", optimizer)

    # Track validation performance for early stopping
    bad_epochs = 0
    best_validate = float('inf')

    for epoch in range(last_epoch + 1, args.max_epoch):
        # Descend ae_loss_weight
        current_ae_loss_weight = args.ae_loss_weight * (0.99 ** epoch)
        param_group = optimizer.param_groups[0]
        param_group['ae-loss-weight'] = current_ae_loss_weight

        train_loader = \
            torch.utils.data.DataLoader(train_dataset, num_workers=1, collate_fn=train_dataset.collater,
                                        batch_sampler=BatchSampler(train_dataset, args.max_tokens, args.batch_size, 1,
                                                                   0, shuffle=False, seed=42))
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        # print(f"Epoch {epoch}, learning rate: {current_lr}")

        stats = OrderedDict()
        stats['loss'] = 0
        stats['ae_loss'] = 0  # Track autoencoding loss
        stats['total_loss'] = 0  # Track total loss
        stats['lr'] = 0
        stats['num_tokens'] = 0
        stats['batch_size'] = 0
        stats['grad_norm'] = 0
        stats['clip'] = 0

        # Display progress
        progress_bar = tqdm(train_loader, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=False)

        # Iterate over the training set
        for i, sample in enumerate(progress_bar):
            if args.cuda:
                sample = utils.move_to_cuda(sample)
            if len(sample) == 0:
                continue
            model.train()

            output, _ = model(sample['src_tokens'], sample['src_lengths'], sample['tgt_inputs'])
            #loss = criterion(output.view(-1, output.size(-1)), sample['tgt_tokens'].view(-1)) / len(sample['src_lengths'])

            # Compute translation loss
            translation_loss = translation_criterion(output.view(-1, output.size(-1)),
                                                     sample['tgt_tokens'].view(-1)) / len(sample['src_lengths'])

            # Prepare inputs for autoencoding task
            ae_inputs = sample['tgt_tokens']  # Target tokens as input for the autoencoder
            ae_output, _ = model(ae_inputs, sample['tgt_lengths'], ae_inputs)  # Autoencoding output

            # Compute autoencoding loss
            ae_loss = autoencoding_criterion(ae_output.view(-1, ae_output.size(-1)), ae_inputs.view(-1)) / len(
                sample['src_lengths'])

            # Combine losses
            loss = translation_loss + args.ae_loss_weight * ae_loss

            if torch.isnan(loss).any():
                logging.warning('Loss is NAN!')
                print(src_dict.string(sample['src_tokens'].tolist()[0]), '---', tgt_dict.string(sample['tgt_tokens'].tolist()[0]))
                # print()
                # import pdb;pdb.set_trace()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            optimizer.zero_grad()

            # Update statistics for progress bar
            #total_loss, num_tokens, batch_size = loss.item(), sample['num_tokens'], len(sample['src_tokens'])
            total_loss, ae_loss_value, num_tokens, batch_size = loss.item(), ae_loss.item(), sample['num_tokens'], len(
                sample['src_tokens'])

            stats['loss'] += total_loss * len(sample['src_lengths']) / sample['num_tokens']
            stats['ae_loss'] += ae_loss_value * len(sample['src_lengths']) / sample['num_tokens']
            stats['total_loss'] += (total_loss + args.ae_loss_weight * ae_loss_value) * len(sample['src_lengths']) / \
                                   sample['num_tokens']
            stats['lr'] += optimizer.param_groups[0]['lr']
            # print("param_groups", optimizer.param_groups[0]['lr'])
            stats['num_tokens'] += num_tokens / len(sample['src_tokens'])
            stats['batch_size'] += batch_size
            stats['grad_norm'] += grad_norm
            stats['clip'] += 1 if grad_norm > args.clip_norm else 0
            progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()},
                                     refresh=True)

        logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.4g}'.format(
            value / len(progress_bar)) for key, value in stats.items())))

        # Calculate validation loss
        valid_perplexity = validate(args, model, criterion, valid_dataset, epoch)
        model.train()

        scheduler.step(valid_perplexity)
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch}, learning rate: {param_group['lr']}")

        # Save checkpoints
        if epoch % args.save_interval == 0:
            utils.save_checkpoint(args, model, optimizer, epoch, valid_perplexity)  # lr_scheduler

        # Check whether to terminate training
        if valid_perplexity < best_validate:
            best_validate = valid_perplexity
            bad_epochs = 0
        else:
            bad_epochs += 1
            print("bad_epochs", bad_epochs)
            print("args.patience", args.patience)
        if bad_epochs >= args.patience:
            logging.info('No validation set improvements observed for {:d} epochs. Early stop!'.format(args.patience))
            break


def validate(args, model, criterion, valid_dataset, epoch):
    """ Validates model performance on a held-out development set. """
    valid_loader = \
        torch.utils.data.DataLoader(valid_dataset, num_workers=1, collate_fn=valid_dataset.collater,
                                    batch_sampler=BatchSampler(valid_dataset, args.max_tokens, args.batch_size, 1, 0,
                                                               shuffle=False, seed=42))
    model.eval()
    stats = OrderedDict()
    stats['valid_loss'] = 0
    stats['num_tokens'] = 0
    stats['batch_size'] = 0

    # Iterate over the validation set
    for i, sample in enumerate(valid_loader):
        if args.cuda:
            sample = utils.move_to_cuda(sample)
        if len(sample) == 0:
            continue
        with torch.no_grad():
            # Compute loss
            output, attn_scores = model(sample['src_tokens'], sample['src_lengths'], sample['tgt_inputs'])
            loss = criterion(output.view(-1, output.size(-1)), sample['tgt_tokens'].view(-1))
        # Update tracked statistics
        stats['valid_loss'] += loss.item()
        stats['num_tokens'] += sample['num_tokens']
        stats['batch_size'] += len(sample['src_tokens'])

    # Calculate validation perplexity
    stats['valid_loss'] = stats['valid_loss'] / stats['num_tokens']
    perplexity = np.exp(stats['valid_loss'])
    stats['num_tokens'] = stats['num_tokens'] / stats['batch_size']

    logging.info(
        'Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.3g}'.format(value) for key, value in stats.items())) +
        ' | valid_perplexity {:.3g}'.format(perplexity))

    return perplexity


if __name__ == '__main__':
    args = get_args()
    args.device_id = 0

    # Set up logging to file
    logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    if args.log_file is not None:
        # Logging to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

    main(args)
