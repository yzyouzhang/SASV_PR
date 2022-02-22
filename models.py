import torch
from torch import nn

class BaselineModel(nn.Module):
    def __init__(self, num_nodes=[512, 256, 128, 64]):
        super().__init__()
        self.name = "Baseline"
        self.enh_DNN = self._make_layers(704, num_nodes)
        self.fc_out = torch.nn.Linear(num_nodes[-1], 2, bias = False)
        self.loss = torch.nn.CrossEntropyLoss(
            weight=torch.FloatTensor([0.1, 0.9])
        )

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst):

        asv_enr = torch.squeeze(embd_asv_enr, 1) # shape: (bs, 192)
        asv_tst = torch.squeeze(embd_asv_tst, 1) # shape: (bs, 192)
        cm_enr = torch.squeeze(embd_cm_enr, 1)  # shape: (bs, 160)
        cm_tst = torch.squeeze(embd_cm_tst, 1) # shape: (bs, 160)

        x = self.enh_DNN(torch.cat([asv_enr, asv_tst, cm_enr, cm_tst], dim = 1)) # shape: (bs, 32)
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

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst, labels, keys):
        asv_enr = torch.squeeze(embd_asv_enr, 1)  # shape: (bs, 192)
        asv_tst = torch.squeeze(embd_asv_tst, 1)  # shape: (bs, 192)
        cm_enr = torch.squeeze(embd_cm_enr, 1)  # shape: (bs, 160)
        cm_tst = torch.squeeze(embd_cm_tst, 1)  # shape: (bs, 160)

        x = self.enh_DNN(torch.cat([asv_enr, asv_tst, cm_enr, cm_tst], dim=1))  # shape: (bs, 32)
        x = self.fc_out(x)  # (bs, 2)
        return self.loss(x, labels)



class OCSoftmax(nn.Module):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()

        scores[labels == 0] = self.r_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.r_fake

        loss = self.softplus(self.alpha * scores).mean()

        # print(output_scores.squeeze(1).shape)

        return loss, -output_scores.squeeze(1)


class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)

    def forward(self, input):
        return nn.functional.linear(input, self.log_weight.exp())


class SimpleProbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SimpleProb"
        self.fc_sv = PositiveLinear(1, 1)
        self.fc_cm = nn.Linear(160, 1)
        self.coss = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.sigmoid = nn.Sigmoid()
        self.loss_sv = nn.BCELoss(weight=torch.FloatTensor([0.1]))
        self.loss_cm = nn.BCELoss(weight=torch.FloatTensor([0.1]))

    def forward_SV_prob(self, embd_asv_enr, embd_asv_tst):
        asv_cos = self.coss(embd_asv_enr, embd_asv_tst).unsqueeze(1)
        asv_score = self.fc_sv(asv_cos)
        p_sv = self.sigmoid(asv_score)
        # p_sv = asv_score
        return p_sv

    def forward_CM_prob(self, embd_cm_tst):
        cm_score = self.fc_cm(embd_cm_tst)
        p_cm = self.sigmoid(cm_score)
        # p_cm = cm_score
        return p_cm

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst):
        p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        p_cm = self.forward_CM_prob(embd_cm_tst)
        x = p_sv * p_cm
        return x

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst, labels, keys):
        p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        loss_sv = self.loss_sv(p_sv, labels.unsqueeze(1).float())
        labels_cm = 2 - keys - labels
        p_cm = self.forward_CM_prob(embd_cm_tst)
        loss_cm = self.loss_cm(p_cm, labels_cm.unsqueeze(1).float())
        return loss_cm + loss_sv


class MTProbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "MultitaskProb"
        self.fc_sv = PositiveLinear(1, 1)
        self.fc_cm = nn.Linear(160, 1)
        self.coss = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.sigmoid = nn.Sigmoid()
        self.loss_sv = nn.BCELoss(weight=torch.FloatTensor([0.1]))
        self.loss_cm = nn.BCELoss(weight=torch.FloatTensor([0.1]))
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward_SV_prob(self, embd_asv_enr, embd_asv_tst):
        asv_cos = self.coss(embd_asv_enr, embd_asv_tst).unsqueeze(1)
        asv_score = self.fc_sv(asv_cos)
        p_sv = self.sigmoid(asv_score)
        return p_sv

    def forward_CM_prob(self, embd_cm_tst):
        cm_score = self.fc_cm(embd_cm_tst)
        p_cm = self.sigmoid(cm_score)
        return p_cm

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst):
        p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        p_cm = self.forward_CM_prob(embd_cm_tst)
        x = p_sv * p_cm
        return x

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst, labels, keys):
        p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        loss_sv = self.loss_sv(p_sv, labels.unsqueeze(1).float())
        labels_cm = 2 - keys - labels
        p_cm = self.forward_CM_prob(embd_cm_tst)
        loss_cm = self.loss_cm(p_cm, labels_cm.unsqueeze(1).float())
        prec = torch.exp(-self.log_vars)
        loss = prec[0] * loss_cm + prec[1] * loss_sv + self.log_vars.sum()
        return loss


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

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst):
        p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        p_cm = self.forward_CM_prob(embd_cm_tst)
        x = p_sv * p_cm
        return x

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst, labels, keys):
        return 0


class Baseline1(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Baseline1"
        self.fc_cm = nn.Linear(160, 2)
        self.coss = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        aasist = torch.load("./aasist/models/weights/AASIST.pth")
        self.fc_cm.weight = nn.Parameter(aasist["out_layer.weight"], requires_grad=False)
        self.fc_cm.bias = nn.Parameter(aasist["out_layer.bias"], requires_grad=False)

    def forward_SV_prob(self, embd_asv_enr, embd_asv_tst):
        asv_cos = self.coss(embd_asv_enr, embd_asv_tst).unsqueeze(1)
        asv_score = asv_cos
        # p_sv = self.sigmoid(asv_score)
        p_sv = asv_score
        return p_sv

    def forward_CM_prob(self, embd_cm_tst):
        cm_score = self.fc_cm(embd_cm_tst)
        # cm_score = self.softmax(cm_score)
        # p_cm = self.sigmoid(cm_score)
        p_cm = cm_score[:, 1].unsqueeze(1)
        return p_cm

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst):
        p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        p_cm = self.forward_CM_prob(embd_cm_tst)
        x = p_sv + p_cm
        return x

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst, labels, keys):
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

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst):

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

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst, labels, keys):
        asv_enr = torch.squeeze(embd_asv_enr, 1)  # shape: (bs, 192)
        asv_tst = torch.squeeze(embd_asv_tst, 1)  # shape: (bs, 192)
        cm_tst = torch.squeeze(embd_cm_tst, 1)  # shape: (bs, 160)

        x = self.enh_DNN(torch.cat([asv_enr, asv_tst, cm_tst], dim=1))  # shape: (bs, 32)
        x = self.fc_out(x)  # (bs, 2)
        return self.loss(x, labels)


class JointProbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "JointProb"
        self.fc_cm = nn.Linear(160, 1)
        self.coss = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.sigmoid = nn.Sigmoid()
        self.loss_sv = nn.BCELoss(weight=torch.FloatTensor([0.1]))

    def forward_SV_prob(self, embd_asv_enr, embd_asv_tst):
        asv_cos = self.coss(embd_asv_enr, embd_asv_tst).unsqueeze(1)
        # asv_score = self.fc_sv(asv_cos)
        asv_score = asv_cos
        p_sv = self.sigmoid(asv_score)
        # p_sv = asv_score
        return p_sv

    def forward_CM_prob(self, embd_cm_tst):
        cm_score = self.fc_cm(embd_cm_tst)
        p_cm = self.sigmoid(cm_score)
        # p_cm = cm_score
        return p_cm

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst):
        p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        p_cm = self.forward_CM_prob(embd_cm_tst)
        x = p_sv * p_cm
        return x

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst, labels, keys):
        p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        p_cm = self.forward_CM_prob(embd_cm_tst)
        loss = self.loss_sv(p_sv * p_cm, labels.unsqueeze(1).float())
        return loss


class ToyProbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "ToyProb"
        # self.fc_sv = PositiveLinear(1, 1)
        # self.fc_spoofprint = PositiveLinear(1, 1)
        self.fc_cm = nn.Linear(160, 1)
        self.fc_asvcm = nn.Linear(352, 1)
        self.coss = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.sigmoid = nn.Sigmoid()
        self.loss_sv = nn.BCELoss(weight=torch.FloatTensor([0.1]))
        self.loss_cm = nn.BCELoss(weight=torch.FloatTensor([0.1]))
        self.loss_spoofprint = nn.BCELoss(weight=torch.FloatTensor([0.1]))
        # aasist = torch.load("./aasist/models/weights/AASIST.pth")
        # self.fc_cm.weight = nn.Parameter(aasist["out_layer.weight"][1, :].unsqueeze(0), requires_grad=False)
        # self.fc_cm.bias = nn.Parameter(aasist["out_layer.bias"][1], requires_grad=False)


    def forward_SV_prob(self, embd_asv_enr, embd_asv_tst):
        asv_cos = self.coss(embd_asv_enr, embd_asv_tst).unsqueeze(1)
        # asv_score = self.fc_sv(asv_cos)
        asv_score = asv_cos
        p_sv = self.sigmoid(asv_score)
        return p_sv

    def forward_CM_prob(self, embd_cm_tst):
        cm_score = self.fc_cm(embd_cm_tst)
        p_cm = self.sigmoid(cm_score)
        return p_cm

    def forward_ASVCM_prob(self, embd_asv_tst, embd_cm_tst):
        embd_tst = torch.cat((embd_asv_tst, embd_cm_tst), dim=1)
        asvcm_score = self.fc_asvcm(embd_tst)
        p_asvcm = self.sigmoid(asvcm_score)
        return p_asvcm

    def forward_Spoofprint_prob(self, embd_cm_enr, embd_cm_tst):
        cm_cos = self.coss(embd_cm_enr, embd_cm_tst).unsqueeze(1)
        # spoofprint_score = self.fc_spoofprint(cm_cos)
        spoofprint_score = cm_cos
        p_spoofprint = self.sigmoid(spoofprint_score)
        return p_spoofprint

    def forward_SpoofASVprint_prob(self, embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst):
        embd_enr = torch.cat((embd_asv_enr, embd_cm_enr), dim=1)
        embd_tst = torch.cat((embd_asv_tst, embd_cm_tst), dim=1)
        embd_cos = self.coss(embd_enr, embd_tst).unsqueeze(1)
        spoofprint_score = embd_cos
        p_spoofprint = self.sigmoid(spoofprint_score)
        return p_spoofprint

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst):
        p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        p_cm = self.forward_CM_prob(embd_cm_tst)
        p_asvcm = self.forward_ASVCM_prob(embd_asv_tst, embd_cm_tst)
        p_spoofprint = self.forward_Spoofprint_prob(embd_cm_enr, embd_cm_tst)
        p_spoofasvprint = self.forward_SpoofASVprint_prob(embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst)
        x = p_asvcm * p_sv
        return x

    def calc_loss(self, embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst, labels, keys):
        # p_sv = self.forward_SV_prob(embd_asv_enr, embd_asv_tst)
        # loss_sv = self.loss_sv(p_sv, labels.unsqueeze(1).float())
        # labels_cm = 2 - keys - labels
        # p_cm = self.forward_CM_prob(embd_cm_tst)
        # loss_cm = self.loss_cm(p_cm, labels_cm.unsqueeze(1).float())
        # loss = loss_cm + loss_sv
        # p_spoofprint = self.forward_Spoofprint_prob(embd_cm_enr, embd_cm_tst)
        # loss_spoofprint = self.loss_spoofprint(p_spoofprint, labels_cm.unsqueeze(1).float())
        # loss += loss_spoofprint
        p_final = self.forward(embd_asv_enr, embd_asv_tst, embd_cm_enr, embd_cm_tst)
        loss = self.loss_sv(p_final, labels.unsqueeze(1).float())
        return loss


