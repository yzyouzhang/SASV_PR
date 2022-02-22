import random
import pickle as pk
from torch.utils.data import Dataset


class SASV_Dataset(Dataset):
    def __init__(self, args, partition):
        self.part = partition
        self.embedding_dir = args.embedding_dir
        if self.part == "trn":
            self.spk_meta_dir = args.spk_meta_dir
            self.load_meta_information()
        else:
            sasv_trial = getattr(args, 'sasv_' + self.part + '_trial')
            with open(sasv_trial, "r") as f:
                self.utt_list = f.readlines()
        self.load_embeddings()

    def load_meta_information(self):
        with open(self.spk_meta_dir + "spk_meta_trn.pk", "rb") as f:
            self.spk_meta = pk.load(f)

    def load_embeddings(self):
        # load saved countermeasures(CM) related preparations
        with open(self.embedding_dir + "cm_embd_" + self.part + ".pk", "rb") as f:
            self.cm_embd = pk.load(f)
        # load saved automatic speaker verification(ASV) related preparations
        with open(self.embedding_dir + "asv_embd_" + self.part + ".pk", "rb") as f:
            self.asv_embd = pk.load(f)
        if self.part in ["dev", "eval"]:
            # load speaker models for development and evaluation sets
            with open(self.embedding_dir + "spk_model_" + self.part + ".pk", "rb") as f:
                self.spk_model = pk.load(f)
            # load speaker CM models for development and evaluation sets
            with open(self.embedding_dir + "spk_cm_model_" + self.part + ".pk", "rb") as f:
                self.spk_cm_model = pk.load(f)

    def __len__(self):
        if self.part == "trn":
            return len(self.cm_embd.keys())
        elif self.part in ["dev", "eval"]:
            return len(self.utt_list)

    def __getitem__(self, idx):
        return getattr(self, 'getitem_'+self.part)(idx)

    def getitem_trn(self, index):
        ans_type = random.randint(0, 1)
        if ans_type == 1:  # target
            spk = random.choice(list(self.spk_meta.keys()))
            enr, tst = random.sample(self.spk_meta[spk]["bonafide"], 2)
            nontarget_type = 0
        elif ans_type == 0:  # nontarget
            nontarget_type = random.randint(1, 2)

            if nontarget_type == 1:  # zero-effort nontarget
                spk, ze_spk = random.sample(self.spk_meta.keys(), 2)
                enr = random.choice(self.spk_meta[spk]["bonafide"])
                tst = random.choice(self.spk_meta[ze_spk]["bonafide"])

            if nontarget_type == 2:  # spoof nontarget
                spk = random.choice(list(self.spk_meta.keys()))
                if len(self.spk_meta[spk]["spoof"]) == 0:
                    while True:
                        spk = random.choice(list(self.spk_meta.keys()))
                        if len(self.spk_meta[spk]["spoof"]) != 0:
                            break
                enr = random.choice(self.spk_meta[spk]["bonafide"])
                tst = random.choice(self.spk_meta[spk]["spoof"])
        else:
            raise ValueError

        return self.asv_embd[enr], self.asv_embd[tst], \
               self.cm_embd[enr], self.cm_embd[tst], \
               ans_type, nontarget_type

    def getitem_dev(self, index):
        line = self.utt_list[index]
        spkmd, key, _, ans = line.strip().split(" ")
        ans_type = int(ans == "target")
        nontype_dict = {"target": 0, "nontarget": 1, "spoof": 2}

        return self.spk_model[spkmd], self.asv_embd[key], \
               self.spk_cm_model[spkmd], self.cm_embd[key], \
               ans_type, nontype_dict[ans]

    def getitem_eval(self, index):
        line = self.utt_list[index]
        spkmd, key, _, ans = line.strip().split(" ")
        ans_type = int(ans == "target")
        nontype_dict = {"target": 0, "nontarget": 1, "spoof": 2}

        return self.spk_model[spkmd], self.asv_embd[key], \
               self.spk_cm_model[spkmd], self.cm_embd[key], \
               ans_type, nontype_dict[ans]

