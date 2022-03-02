import os, sys, json, glob
import pickle as pkl
import shutil
import argparse
from utils import *
from models import *
from trainer import Trainer
from dataset import *
from torch.utils.data import DataLoader
from metrics import get_all_EERs_my

def initParams():
    parser = argparse.ArgumentParser(description="SASV2022_ProductRule.")

    parser.add_argument('--seed', type=int, help="random number seed", default=688)

    parser.add_argument('-m', '--model_name', help='Model arch', default='pr',
                        choices=['baseline1', 'baseline2',
                                 'pr_l_u', 'pr_s_u', 'pr_c_u', 'pr_l_t', 'pr_s_t',
                                 'baseline1_l_u', 'baseline1_s_u'])

    # Output folder prepare
    parser.add_argument(
        "-o", "--output_dir", dest="output_dir", type=str,
        help="output directory for results", required=True,
        default="./exp_result",
    )
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")

    # Dataset prepare
    parser.add_argument("--embedding_dir", type=str, default="./embeddings/",
                        help="folder for the pretrained ASV and CM embeddings")
    parser.add_argument("--spk_meta_dir", type=str, default="./spk_meta/",
                        help="folder for the speaker meta information")
    parser.add_argument("--sasv_dev_trial", type=str,
                        default="protocols/ASVspoof2019.LA.asv.dev.gi.trl.txt")
    parser.add_argument("--sasv_eval_trial", type=str,
                        default="protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt")
    parser.add_argument("--cm_trn_list", type=str,
                        default="protocols/ASVspoof2019.LA.cm.train.trn.txt")
    parser.add_argument("--cm_dev_list", type=str,
                        default="protocols/ASVspoof2019.LA.cm.dev.trl.txt")
    parser.add_argument("--cm_eval_list", type=str,
                        default="protocols/ASVspoof2019.LA.cm.eval.trl.txt")

    ## Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=1024,
                        help="Mini batch size for training")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--clip_norm', type=int, default=None, help="clip norm for training")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for computation")
    parser.add_argument('--cudnn_deterministic_toggle', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--cudnn_benchmark_toggle', type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--test_only', action='store_true',
                        help="test the trained model in case the test crash sometimes or another test method")

    args = parser.parse_args()

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    set_seed(args)

    # generate speaker-utterance meta information
    if not (
            os.path.exists(args.spk_meta_dir + "spk_meta_trn.pk")
            and os.path.exists(args.spk_meta_dir + "spk_meta_dev.pk")
            and os.path.exists(args.spk_meta_dir + "spk_meta_eval.pk")
    ):
        generate_spk_meta(args)

    if args.test_only:
        pass
    else:
        # Path for output data
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        else:
            shutil.rmtree(args.output_dir)
            os.mkdir(args.output_dir)

        # Folder for intermediate results
        if not os.path.exists(os.path.join(args.output_dir, 'checkpoints')):
            os.makedirs(os.path.join(args.output_dir, 'checkpoints'))
        else:
            shutil.rmtree(os.path.join(args.output_dir, 'checkpoints'))
            os.mkdir(os.path.join(args.output_dir, 'checkpoints'))

        # Save training arguments
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as file:
            file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def train(args):
    if "pr" in args.model_name:
        mapping_dict = {"l": "linear", "s": "sigmoid", "c": None}
        trainable_dict = {"u": False, "t": True}
        if "c" in args.model_name:
            f = open("./calibrator/sv_sigmoid.pk", 'rb')
            calibrator = pkl.load(f)
        else:
            calibrator = None
        model = Parallel_PR(trainable=trainable_dict[args.model_name[-1]], calibrator=calibrator,
                            map_function=mapping_dict[args.model_name[-3]])
    elif "baseline1_" in args.model_name:
        mapping_dict = {"l": "linear", "s": "sigmoid"}
        model = Baseline1_improved(map_function=mapping_dict[args.model_name[-3]])
    elif args.model_name == "baseline1":
        model = Baseline1()
    elif args.model_name == "baseline2":
        model = Baseline2(num_nodes=[256, 128, 64])
    else:
        raise ValueError("Which model do you want to use?")
    set_init_weights(model)
    trainer = Trainer(args, model)
    trainer.run_train()

    return trainer

def evaluate_one_iter(args, model, data_minibatch):
    asv1, asv2, cm2, ans, key = data_minibatch
    if torch.cuda.is_available():
        asv1 = asv1.to(args.device)
        asv2 = asv2.to(args.device)
        cm2 = cm2.to(args.device)

    pred = model(asv1, asv2, cm2)

    return {"pred": pred, "key": key}

def evaluate_on_set(args, model, set):
    model.eval()
    evaluation_set = SASV_Dataset(args, set)
    eval_loader = DataLoader(evaluation_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False, pin_memory=True)
    preds, keys = [], []
    with torch.no_grad():
        for num, data_slice in enumerate(eval_loader):
            output = evaluate_one_iter(args, model, data_slice)
            preds.append(output["pred"])
            keys.extend(list(output["key"]))
        if args.model_name == "baseline" or args.model_name == "baseline2":
            preds = torch.cat(preds, dim=0)[:, 1].detach().cpu().numpy()
        else:
            preds = torch.cat(preds, dim=0).detach().cpu().numpy()

        with open(os.path.join(args.output_dir, set + '_preds.pkl'), 'wb') as handle:
            pkl.dump(preds, handle, protocol=pkl.HIGHEST_PROTOCOL)
        with open(os.path.join(args.output_dir, set + '_keys.pkl'), 'wb') as handle:
            pkl.dump(keys, handle, protocol=pkl.HIGHEST_PROTOCOL)

        sasv_eer, sv_eer, spf_eer = get_all_EERs_my(preds=preds, keys=keys)
        print("sasv_eer_" + set + ": %0.3f, sv_eer_" % (100 * sasv_eer) + set +
              ": %0.3f, spf_eer_" % (100 * sv_eer) + set + ": %0.3f" % (100 * spf_eer))

def evaluate(args, model):
    print("\nFinal evaluation for the best epoch:")
    evaluate_on_set(args, model, "dev")
    evaluate_on_set(args, model, "eval")


if __name__ == '__main__':
    args = initParams()
    if not args.test_only:
        train(args)
    model = torch.load(glob.glob(os.path.join(args.output_dir, "*_best.pt"))[-1])
    evaluate(args, model)

