import numpy as np
import glob, os
from tqdm import tqdm
import pickle
from scipy.spatial.distance import cdist
from metrics import get_all_EERs
from sklearn.calibration import CalibratedClassifierCV

def utt_to_meta(spk_meta):
    utt_meta_dict = {}
    for spkid in spk_meta:
        realList = spk_meta[spkid]['bonafide']
        fakeList = spk_meta[spkid]['spoof']

        for realFile in realList:
            utt_meta_dict[realFile] = ['bonafide', spkid]

        for fakeFile in fakeList:
            utt_meta_dict[fakeFile] = ['spoof', spkid]

    return utt_meta_dict

def create_calibration_data(cohortUttEmbeds, cohortSpkEmbeds,
                            cohortUttMeta):
    utt_bonafide = []
    for cohortUtt in tqdm(cohortUttEmbeds):
        if cohortUttMeta[cohortUtt][0] == 'spoof':
            continue
        utt_bonafide.append(cohortUtt)
    scores = []
    labels = []
    for spk in tqdm(cohortSpkEmbeds):
        spkEmb = cohortSpkEmbeds[spk]
        for utt in utt_bonafide:
            cos_sim = 1 - cdist([spkEmb], [cohortUttEmbeds[utt]], 'cosine')[0]
            scores.append(cos_sim)
            if cohortUttMeta[utt][1] == spk:
                labels.append(1)
            else:
                labels.append(0)
    return scores, labels

if __name__ == '__main__':
    # cohort data loading
    coh_utt_emb_file = './embeddings/asv_embd_dev.pk'
    coh_spk_emb_file = './embeddings/spk_model_dev.pk'
    coh_spk_meta_file = './spk_meta/spk_meta_dev.pk'

    f = open(coh_spk_meta_file, 'rb')  # 'r' for reading; can be omitted
    coh_spk_meta = pickle.load(f)  # load file content as mydict

    f = open(coh_utt_emb_file, 'rb')  # 'r' for reading; can be omitted
    coh_utt_emb = pickle.load(f)  # load file content as mydict

    f = open(coh_spk_emb_file, 'rb')  # 'r' for reading; can be omitted
    coh_spk_emb = pickle.load(f)  # load file content as mydict

    coh_utt_meta_dict = utt_to_meta(coh_spk_meta)

    scores, labels = create_calibration_data(coh_utt_emb, coh_spk_emb, coh_utt_meta_dict)

    sv_isotonic = CalibratedClassifierCV(method="isotonic")
    sv_isotonic.fit(scores, labels)

    with open("./calibrator/" + "sv_isotonic.pk", "wb") as f:
        pickle.dump(sv_isotonic, f)

    sv_sigmoid = CalibratedClassifierCV(method="sigmoid")
    sv_sigmoid.fit(scores, labels)

    with open("./calibrator/" + "sv_sigmoid.pk", "wb") as f:
        pickle.dump(sv_sigmoid, f)

