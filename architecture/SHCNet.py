from torch.nn.parameter import Parameter
from utils import *


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


def shift_back(inputs, bias_list):
    for i in range(opt.channel):
        inputs[:, i, :, :opt.size] = \
            inputs[:, i, :, bias_list[i]:bias_list[i] + opt.size]
    return inputs[:, :, :, :opt.size]


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ELU(alpha=1.0, inplace=True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            if i == 0:
                m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            else:
                m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x) + x
        return res


class HFC(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(HFC, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1)
        self.conv3 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        return x1, img


class MaskAttention(nn.Module):
    def __init__(
            self, n_feat):
        super(MaskAttention, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat)

    def forward(self, mask_shift, bias_list):
        mask_shift = self.conv1(mask_shift)
        attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask_shift)))
        res = mask_shift * attn_map
        mask_shift = res + mask_shift
        mask_emb = shift_back(mask_shift, bias_list)
        return mask_emb


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ELU(alpha=1.0, inplace=False),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return y


class SHCNet(nn.Module):
    def __init__(self, iter_stage=4, conv=default_conv):
        super(SHCNet, self).__init__()
        act = nn.ELU(alpha=1.0, inplace=True)
        self.iter_stage = iter_stage
        channel = opt.channel
        n_resblocks = 1
        n_feats = 160
        kernel_size = 3
        self.stage = 2

        # mask attention
        self.mm0 = MaskAttention(channel)
        self.mm1 = MaskAttention(channel)
        # phiX
        self.phi_head = nn.ModuleList([])
        self.phi_body = nn.ModuleList([])
        # phiT
        self.phiT_head = nn.ModuleList([])
        self.phiT_body = nn.ModuleList([])
        # rho
        self.rho = nn.ParameterList([])
        self.chattn = nn.ModuleList([])
        # proximal mapping
        # embedding
        self.embedding = nn.ModuleList([])
        # bottleneck
        self.bottleneck = nn.ModuleList([])
        # encoder
        self.encoder_layers_list = nn.ModuleList([])
        # soft
        self.soft_thr = nn.ParameterList([])
        # decoder
        self.decoder_layers_list = nn.ModuleList([])
        # crf
        self.crf_list = nn.ModuleList([])
        # Output projection
        self.mapping = nn.ModuleList([])

        # hierarchical feature
        self.hf_fea = nn.ModuleList([])
        self.hf_use = nn.ModuleList([])
        self.hf_concat = nn.ModuleList([])

        for s in range(self.iter_stage):

            # phiX
            self.phi_head.append(nn.Conv2d(
                channel, channel, kernel_size, padding=(kernel_size // 2), bias=True
            ))
            phi_conv = [conv(channel * 2, n_feats, kernel_size)]
            for _ in range(n_resblocks):
                phi_conv.append(ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1))
            phi_conv.append(nn.Conv2d(n_feats, channel, kernel_size, padding=(kernel_size // 2), bias=True))
            phi_conv = nn.Sequential(*phi_conv)
            self.phi_body.append(phi_conv)

            # phiT
            self.phiT_head.append(nn.Conv2d(channel, channel, kernel_size, padding=(kernel_size // 2), bias=True))
            phiT_conv = [conv(channel * 2, n_feats, kernel_size)]
            for _ in range(n_resblocks):
                phiT_conv.append(ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1))
            phiT_conv.append(nn.Conv2d(n_feats, channel, kernel_size, padding=(kernel_size // 2), bias=True))
            phiT_conv = nn.Sequential(*phiT_conv)
            self.phiT_body.append(phiT_conv)

            # rho
            self.rho.append(Parameter(torch.Tensor([0.5])))
            self.chattn.append(ChannelAttention(channel))

            # proximal mapping
            # embedding
            self.embedding.append(nn.Conv2d(channel, channel, 3, 1, 1, bias=False))

            # encoder
            encoder_layers = nn.ModuleList([])
            dim_stage = channel
            for i in range(self.stage):
                encoder_layers.append(nn.ModuleList([
                    ResBlock(conv, dim_stage, kernel_size, act=act, res_scale=1),
                    nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                ]))
                dim_stage *= 2
            self.encoder_layers_list.append(encoder_layers)

            # soft
            self.soft_thr.append(Parameter(torch.Tensor([0.01])))

            # bottleneck
            self.bottleneck.append(ResBlock(conv, dim_stage, kernel_size, act=act, res_scale=1))

            # decoder
            decoder_layers = nn.ModuleList([])
            for i in range(self.stage):
                decoder_layers.append(nn.ModuleList([
                    nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                    nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                    ResBlock(conv, dim_stage // 2, kernel_size, act=act, res_scale=1),
                ]))
                dim_stage //= 2
            self.decoder_layers_list.append(decoder_layers)

            # Output projection
            self.mapping.append(nn.Conv2d(channel, channel, 3, 1, 1, bias=False))

            # hf
            self.hf_fea.append(nn.Conv2d(opt.channel, opt.channel, kernel_size=3, stride=1, padding=1))
            self.hf_use.append(nn.Conv2d(opt.channel * (s + 1), opt.channel, kernel_size=1, stride=1, padding=0))
            self.hf_concat.append(HFC(n_feat=opt.channel, kernel_size=3, bias=False))

    def forward(self, y, x, input_mask, bias_list):
        '''
        :param y: [batch_size, height, width + channel_bias]
        :param x: [batch_size, channel, height, width]
        :param input_mask: [batch_size, time, height, width]
        :return: x: [batch_size, channel * time, height, width]
        '''

        input_x = torch.clone(x)
        shift_mask = shift(input_mask, bias_list=bias_list)
        mask_whc = torch.sum(shift_mask, dim=2)
        HT = []

        X_ori = torch.clone(input_x)

        for s in range(self.iter_stage):
            mask_attn0 = self.mm0(mask_whc, bias_list)
            phi_head = self.phi_head[s](input_x)
            phi_attn = mask_attn0 * phi_head
            phi_conv = self.phi_body[s](torch.cat([phi_attn, phi_head], dim=1))

            phiX = shift_sum(phi_conv, bias_list)
            phiX_minus_y = phiX - y
            phiX_minus_y = init_input(phiX_minus_y, bias_list=bias_list)

            # phiT
            mask_attn1 = self.mm1(mask_whc, bias_list)
            phiT_head = self.phiT_head[s](phiX_minus_y)
            phiT_attn = mask_attn1 * phiT_head
            phiT_conv = self.phiT_body[s](torch.cat([phiT_attn, phiT_head], dim=1))

            # rho
            rho = self.rho[s] * self.chattn[s](phiT_conv)

            # gradient descent
            r = input_x - rho * phiT_conv

            # hf_use
            fea_r = self.hf_fea[s](r)
            if s != 0:
                HT.append(fea_r)
                r = self.hf_use[s](torch.cat(HT, 1))
                HT.pop()

            # Embedding
            fea = self.embedding[s](r)

            # Encoder
            fea_encoder = []
            for (Res, FeaDownSample) in self.encoder_layers_list[s]:
                fea = Res(fea)
                fea_encoder.append(fea)
                fea = FeaDownSample(fea)

            # soft thresh
            fea = self.bottleneck[s](fea)

            # Decoder
            for i, (FeaUpSample, Fution, Res) in enumerate(self.decoder_layers_list[s]):
                fea = FeaUpSample(fea)
                fea = Fution(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
                fea = Res(fea)

            # Mapping
            input_x = self.mapping[s](fea) + r

            # hierarchical
            Ht, input_x = self.hf_concat[s](input_x, X_ori)
            HT.append(Ht)
            pass

        return input_x
