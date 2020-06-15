import argparse
import os

parser = argparse.ArgumentParser()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Model Parameters
model_arg = parser.add_argument_group('Model')
# general model config
model_arg.add_argument('--complex_mask', type=str, default=True, help="phase masking or not")
model_arg.add_argument('--loss', type=str, default="wsdr", help="loss type (l1, l2, sdr, wsdr, spec)")
model_arg.add_argument('--padding', type=str, default="SAME", help="padding for Conv")
# for Hybrid Network
model_arg.add_argument('--network', type=str, default='unet', help="network type (complex/unet/lstm/dilated)")
model_arg.add_argument('--model', type=str, default='model10', help="network size (model10/model20/model10_tight)")
model_arg.add_argument('--network2', type=str, default='none', help="network2 type (none for not using)")
model_arg.add_argument('--model2', type=str, default='model10', help="network2 size")
model_arg.add_argument('--hybrid', type=str, default=False, help="hybrid or not")
model_arg.add_argument('--loss_seq', type=str, default='second', help="hybrid type (mid/ end/ both/ two/ first/ second)")
model_arg.add_argument('--loss_thereshold', type=int, default=-1, help="loss scheduling threshold")
# for DFT
model_arg.add_argument('--nfft', type=int, default=512, help="fft size (window size of front-end 1d conv filter)")
model_arg.add_argument('--stride', type=int, default=256, help="hop size (stride of front-end 1d conv filter)")
model_arg.add_argument('--ortho', type=str2bool, default=True, help="orthogonal stft or not")
model_arg.add_argument('--hanning', type=str2bool, default=True, help="hanning or not")
# for inference
model_arg.add_argument('--restore', type=str2bool, default=False, help="restore model or not")
model_arg.add_argument('--which_model', type=str, default='train', help="which model to restore or save (folder name)")
model_arg.add_argument('--model_number', type=int, default=300000, help="model number to restore")
model_arg.add_argument('--moving_rate', type=int, default=4, help="moving rate")
model_arg.add_argument('--pad_type', type=str, default='reflect', help="padding type for inference")


# Data Parameters
data_arg = parser.add_argument_group('Data')
# for inference while training
data_arg.add_argument('--train_root', type=str, default="", help="train dataset directory")
data_arg.add_argument('--test_sample_root', type=str, default="./TestAudioFile", help="test audio while training")
data_arg.add_argument('--simul_sample_root', type=str, default="./SimulAudioFile", help="test audio while training")
data_arg.add_argument('--simul_sample_root2', type=str, default="./SimulAudioFile2", help="test audio while training")
# data hyperparemeters
data_arg.add_argument('--is_multi', type=str2bool, default=False, help="multi channel or not")
data_arg.add_argument('--source', type=str, default='vocal', help="which source")
data_arg.add_argument('--sampling_rate', type=int, default=16000, help="sampling rate")
data_arg.add_argument('--audio_size', type=int, default=1024*16, help="input & output audio size")
data_arg.add_argument('--batch_size', type=int, default=16, help="batch-size while training")
data_arg.add_argument('--eps', type=float, default=1e-7, help="epsilon")
data_arg.add_argument('--is_log', type=str2bool, default=True, help="log magnitude for U-Net")

# Training Parameters
train_arg = parser.add_argument_group('Training')
train_arg.add_argument('--train', type=str2bool, default=True, help="Train session(True)")
# Optimizer
train_arg.add_argument('--trainer_type', type=str, default='ADAM', help="which trainer")
train_arg.add_argument('--Beta1', type=float, default=0.5, help="Beta1")
train_arg.add_argument('--Beta2', type=float, default=0.9, help="Beta2")
train_arg.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
train_arg.add_argument('--lr_decay', type=str2bool, default=True, help="Learning rate decay")
train_arg.add_argument('--lr_decay_rate', type=float, default=1.4, help="lr_decay_rate")
train_arg.add_argument('--lr_decay_iter', type=int, default=50000, help="lr_decay_iteration")
# for training
train_arg.add_argument('--iter', type=int, default=300000, help="iterations")
train_arg.add_argument('--model_save_iter', type=int, default=300000, help="iterations")
train_arg.add_argument('--sdr_lambda', type=int, default=1, help="hyperparameter for sdr loss")
train_arg.add_argument('--spec_lambda', type=int, default=1, help="hyperparameter for spec loss")
train_arg.add_argument('--comment', type=str, default='', help="comment section")

# Test Parameters
test_arg = parser.add_argument_group('Test')
test_arg.add_argument('--test_root', type=str, default="./", help="train audio dataset directory")
test_arg.add_argument('--test_npys', type=str, default="./train/Test/*.npy", help="test npys")
test_arg.add_argument('--use_npys', type=str2bool, default=False, help="if true: use npys preprocess files, else: mp3")
test_arg.add_argument('--use_mask', type=str2bool, default=True, help="if true: use hanning")
test_arg.add_argument('--evaluation', type=str2bool, default=True, help="if true: do sdr evaluatino")
test_arg.add_argument('--framewise_sdr', type=str2bool, default=True, help="if true: do framewise evaluatino")


config, unparsed = parser.parse_known_args()

print("\n-----------------------------------configuration----------------------------------------")
print(config, "\n") # print all the arguments that are defined in this file


if __name__ == "__main__":
    print(config.train_root)
    print(os.listdir(config.train_root))
