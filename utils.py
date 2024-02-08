from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import torch
import numpy as np

class MetricLogger(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressLogger(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def metric_AUROC(target, output, nb_classes=14):
    outAUROC = []

    target = target.cpu().numpy()
    output = output.cpu().numpy()

    for i in range(nb_classes):
        outAUROC.append(roc_auc_score(target[:, i], output[:, i]))

    return outAUROC


def vararg_callback_bool(option, opt_str, value, parser):
    assert value is None

    arg = parser.rargs[0]
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        value = True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        value = False

    del parser.rargs[:1]
    setattr(parser.values, option.dest, value)


def vararg_callback_int(option, opt_str, value, parser):
    assert value is None
    value = []

    def intable(str):
        try:
            int(str)
            return True
        except ValueError:
            return False

    for arg in parser.rargs:
        # stop on --foo like options
        if arg[:2] == "--" and len(arg) > 2:
            break
        # stop on -a, but not on -3 or -3.0
        if arg[:1] == "-" and len(arg) > 1 and not intable(arg):
            break
        value.append(int(arg))

    del parser.rargs[:len(value)]
    setattr(parser.values, option.dest, value)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

def step_decay(step, lr, epochs):

    progress = (step - 20) / float(epochs - 20)
    progress = np.clip(progress, 0.0, 1.0)
    lr = lr * 0.5 * (1. + np.cos(np.pi * progress))

    lr = lr * np.minimum(1., step / 20)

    return lr

def cosine_anneal_schedule(t,epochs,learning_rate):
    T=epochs
    M=1
    alpha_zero = learning_rate

    cos_inner = np.pi * (t % (T // M))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    return float(alpha_zero / 2 * cos_out)

def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1 > 0.5).astype(np.bool)
    im2 = np.asarray(im2 > 0.5).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def mean_dice_coef(y_true,y_pred):
    sum=0
    for i in range (y_true.shape[0]):
        sum += dice(y_true[i,:,:,:],y_pred[i,:,:,:])
    return sum/y_true.shape[0]

def print_metrics(y_test, pred, dataset):
    if dataset == "COVIDx":
        mapping = {
            'normal': 0,
            'pneumonia': 1,
            'COVID-19': 2
        }
    else:
        mapping = {
            'Normal': 0,
            'No Lung Opacity/Not Normal': 1,
            'Lung Opacity': 2
        }
    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')
    print(matrix)

    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]

    print('Sens', ', '.join('{}: {:.3f}'.format(cls.capitalize(), class_acc[i]) for cls, i in mapping.items()))
    print('PPV', ', '.join('{}: {:.3f}'.format(cls.capitalize(), ppvs[i]) for cls, i in mapping.items()))


def load_swin_pretrained(ckpt, model):
    state_dict = ckpt
    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            print(f"Error in loading {k}, passing......", file=writter)
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized


    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    del ckpt
    torch.cuda.empty_cache()