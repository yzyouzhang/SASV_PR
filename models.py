import torch
from torch import nn


class SimpleProbModel_woTrn(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SimpleProb_woTrn"
        self.fc_cm = nn.Linear(160, 1)
        self.coss = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.sigmoid = nn.Sigmoid()
        aasist = torch.load("./aasist/models/weights/AASIST.pth")
        self.fc_cm.weight = nn.Parameter(aasist["out_layer.weight"][1, :].unsqueeze(0), requires_grad=False)
        self.fc_cm.bias = nn.Parameter(aasist["out_layer.bias"][1], requires_grad=False)

    def forward_SV_prob(self, embd_asv_enr, embd_asv_tst):
        asv_cos = self.coss(embd_asv_enr, embd_asv_tst).unsqueeze(1)
        asv_score = asv_cos
        p_sv = self.sigmoid(asv_score)
        # p_sv = asv_score
        return p_sv

    def forward_CM_prob(self, embd_cm_tst):
        cm_score = self.fc_cm(embd_cm_tst)
        p_cm = self.sigmoid(cm_score)
        # p_cm = cm_score
        return p_cm

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_tst):
        p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        p_cm = self.forward_CM_prob(embd_cm_tst)
        x = p_sv * p_cm
        return x

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_tst, labels):
        return 0


class Baseline1(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Baseline1"
        self.fc_cm = nn.Linear(160, 2)
        self.coss = nn.CosineSimilarity(dim=1, eps=1e-8)
        aasist = torch.load("./aasist/models/weights/AASIST.pth")
        self.fc_cm.weight = nn.Parameter(aasist["out_layer.weight"], requires_grad=False)
        self.fc_cm.bias = nn.Parameter(aasist["out_layer.bias"], requires_grad=False)

    def forward_SV_prob(self, embd_asv_enr, embd_asv_tst):
        asv_cos = self.coss(embd_asv_enr, embd_asv_tst).unsqueeze(1)
        asv_score = asv_cos
        p_sv = asv_score
        return p_sv

    def forward_CM_prob(self, embd_cm_tst):
        cm_score = self.fc_cm(embd_cm_tst)
        p_cm = cm_score[:, 1].unsqueeze(1)
        return p_cm

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_tst):
        p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        p_cm = self.forward_CM_prob(embd_cm_tst)
        x = p_sv + p_cm
        return x

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_tst, labels):
        return 0


class Baseline2(nn.Module):
    def __init__(self, num_nodes=[256, 128, 64]):
        super().__init__()
        self.name = "Baseline2"
        self.enh_DNN = self._make_layers(544, num_nodes)
        self.fc_out = torch.nn.Linear(num_nodes[-1], 2, bias = False)
        self.loss = torch.nn.CrossEntropyLoss(
            weight=torch.FloatTensor([0.1, 0.9])
        )

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_tst):
        asv_enr = torch.squeeze(embd_asv_enr, 1) # shape: (bs, 192)
        asv_tst = torch.squeeze(embd_asv_tst, 1) # shape: (bs, 192)
        cm_tst = torch.squeeze(embd_cm_tst, 1) # shape: (bs, 160)

        x = self.enh_DNN(torch.cat([asv_enr, asv_tst, cm_tst], dim = 1)) # shape: (bs, 32)
        x = self.fc_out(x)  # (bs, 2)

        return x

    def _make_layers(self, in_dim, l_nodes):
        l_fc = []
        for idx in range(len(l_nodes)):
            if idx == 0:
                l_fc.append(torch.nn.Linear(in_features = in_dim,
                    out_features = l_nodes[idx]))
            else:
                l_fc.append(torch.nn.Linear(in_features = l_nodes[idx-1],
                    out_features = l_nodes[idx]))
            l_fc.append(torch.nn.LeakyReLU(negative_slope = 0.3))
        return torch.nn.Sequential(*l_fc)

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_tst, labels):
        asv_enr = torch.squeeze(embd_asv_enr, 1)  # shape: (bs, 192)
        asv_tst = torch.squeeze(embd_asv_tst, 1)  # shape: (bs, 192)
        cm_tst = torch.squeeze(embd_cm_tst, 1)  # shape: (bs, 160)

        x = self.enh_DNN(torch.cat([asv_enr, asv_tst, cm_tst], dim=1))  # shape: (bs, 32)
        x = self.fc_out(x)  # (bs, 2)
        return self.loss(x, labels)


class Parallel_PR(nn.Module):
    def __init__(self, trainable=True, calibrator=None):
        super().__init__()
        self.name = "ProductRule"
        self.trainable = trainable
        self.calibrator = calibrator
        self.fc_cm = nn.Linear(160, 1)
        if not self.trainable:
            aasist = torch.load("./aasist/models/weights/AASIST.pth")
            self.fc_cm.weight = nn.Parameter(aasist["out_layer.weight"][1, :].unsqueeze(0), requires_grad=False)
            self.fc_cm.bias = nn.Parameter(aasist["out_layer.bias"][1], requires_grad=False)
        self.coss = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.sigmoid = nn.Sigmoid()
        self.loss_sasv = nn.BCELoss(weight=torch.FloatTensor([0.1]))

    def forward_SV_prob(self, embd_asv_enr, embd_asv_tst):
        asv_cos = self.coss(embd_asv_enr, embd_asv_tst).unsqueeze(1)
        asv_score = asv_cos
        p_sv = self.sigmoid(asv_score)
        return p_sv

    def forward_CM_prob(self, embd_cm_tst):
        cm_score = self.fc_cm(embd_cm_tst)
        p_cm = self.sigmoid(cm_score)
        return p_cm

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_tst):
        p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        if not self.trainable and self.calibrator:
            asv_cos = self.coss(embd_asv_enr, embd_asv_tst).unsqueeze(1)
            p_sv = self.calibrator.predict_proba(asv_cos.cpu().numpy())[:, 1]
            p_sv = torch.from_numpy(p_sv).to(embd_asv_enr.device).unsqueeze(1)
        p_cm = self.forward_CM_prob(embd_cm_tst)
        x = p_sv * p_cm
        return x

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_tst, labels):
        if not self.trainable:
            return 0
        sasv_score = self.forward(embd_asv_enr, embd_asv_tst, embd_cm_tst)
        loss = self.loss_sasv(sasv_score, labels.unsqueeze(1).float())
        return loss


class Parallel_SR(nn.Module):
    def __init__(self, trainable=True):
        super().__init__()
        self.name = "SumRule"
        self.trainable = trainable
        self.fc_cm = nn.Linear(160, 1)
        if not self.trainable:
            aasist = torch.load("./aasist/models/weights/AASIST.pth")
            self.fc_cm.weight = nn.Parameter(aasist["out_layer.weight"][1, :].unsqueeze(0), requires_grad=False)
            self.fc_cm.bias = nn.Parameter(aasist["out_layer.bias"][1], requires_grad=False)
        self.coss = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.sigmoid = nn.Sigmoid()
        self.loss_sasv = nn.BCELoss(weight=torch.FloatTensor([0.1]))

    def forward_SV_prob(self, embd_asv_enr, embd_asv_tst):
        asv_cos = self.coss(embd_asv_enr, embd_asv_tst).unsqueeze(1)
        asv_score = asv_cos
        p_sv = self.sigmoid(asv_score)
        return p_sv

    def forward_CM_prob(self, embd_cm_tst):
        cm_score = self.fc_cm(embd_cm_tst)
        p_cm = self.sigmoid(cm_score)
        return p_cm

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_tst):
        p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        p_cm = self.forward_CM_prob(embd_cm_tst)
        x = p_sv + p_cm
        return x

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_tst, labels):
        if not self.trainable:
            return 0
        sasv_score = self.forward(embd_asv_enr, embd_asv_tst, embd_cm_tst)
        loss = self.loss_sasv(torch.clamp(sasv_score, max=1), labels.unsqueeze(1).float())
        return loss
