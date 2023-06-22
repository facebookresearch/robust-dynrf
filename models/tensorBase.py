# Copyright (c) Meta Platforms, Inc. and affiliates.

import time

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

from renderer import *


def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1.0 - torch.exp(-sigma * dist)

    T = torch.cumprod(
        torch.cat(
            [torch.ones(alpha.shape[0], 1).to(alpha.device), 1.0 - alpha + 1e-10], -1
        ),
        -1,
    )

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]


def RGBRender(xyz_sampled, viewdirs, features):
    rgb = features
    return rgb


class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume, tSize):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[:3], tSize)
        self.gridSize = torch.LongTensor(
            [alpha_volume.shape[-2], alpha_volume.shape[-3], alpha_volume.shape[-4]]
        ).to(self.device)
        self.tSize = tSize

    def sample_alpha(self, xyz_sampled, t):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(
            torch.permute(
                self.alpha_volume.to(xyz_sampled.get_device())[:, 0], (0, 4, 1, 2, 3)
            ),
            xyz_sampled.view(1, -1, 1, 1, 3),
            align_corners=True,
        )
        t_int = torch.round((t + 1) / 2 * (self.tSize - 1)).long()
        t_int_onehot = torch.nn.functional.one_hot(t_int, num_classes=self.tSize)
        alpha_vals = torch.permute(alpha_vals[0, :, :, 0, 0], (1, 0))[
            t_int_onehot.bool()
        ]

        alpha_vals = alpha_vals.view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (
            xyz_sampled - self.aabb.to(xyz_sampled.get_device())[0]
        ) * self.invgridSize.to(xyz_sampled.get_device()) - 1


class MLPRender_Fea(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, time):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender_Fea_TimeEmbedding(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea_TimeEmbedding, self).__init__()

        self.in_mlpC = 2 * feape * inChanel + inChanel
        self.in_view = 2 * viewpe * 3 + 3
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC + self.in_view, 3)

        self.mlp = torch.nn.Sequential(
            layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True)
        )
        self.mlp_view = torch.nn.Sequential(layer3)
        torch.nn.init.constant_(self.mlp_view[-1].bias, 0)

    def forward(self, pts, viewdirs, features, time):
        indata = [features]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        indata_view = [viewdirs]
        if self.viewpe > 0:
            indata_view += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        inter_features = self.mlp(mlp_in)
        mlp_view_in = torch.cat([inter_features] + indata_view, dim=-1)
        rgb = self.mlp_view(mlp_view_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender_Fea_late_view(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea_late_view, self).__init__()

        self.in_mlpC = 2 * feape * inChanel + inChanel + 2 * 10 * 3 + 3 + 2 * 8 * 1 + 1
        self.in_view = 2 * viewpe * 3 + 3
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC + self.in_view, 3)

        self.mlp = torch.nn.Sequential(
            layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True)
        )
        self.mlp_view = torch.nn.Sequential(layer3)
        torch.nn.init.constant_(self.mlp_view[-1].bias, 0)

    def forward(self, pts, viewdirs, features, time):
        indata = [features]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        indata += [pts]
        indata += [positional_encoding(pts, 10)]
        indata += [time]
        indata += [positional_encoding(time, 8)]
        indata_view = [viewdirs.detach()]
        if self.viewpe > 0:
            indata_view += [positional_encoding(viewdirs.detach(), self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        inter_features = self.mlp(mlp_in)
        mlp_view_in = torch.cat([inter_features] + indata_view, dim=-1)
        rgb = self.mlp_view(mlp_view_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender_Fea_woView(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea_woView, self).__init__()

        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender_PE(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + (3 + 2 * pospe * 3) + inChanel  #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + inChanel
        self.viewpe = viewpe

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class TensorBase(torch.nn.Module):
    def __init__(
        self,
        aabb,
        gridSize,
        tSize,
        device,
        density_n_comp=8,
        appearance_n_comp=24,
        app_dim=27,
        shadingMode="MLP_PE",
        alphaMask=None,
        near_far=[2.0, 6.0],
        density_shift=-10,
        alphaMask_thres=0.001,
        distance_scale=25,
        rayMarch_weight_thres=0.0001,
        pos_pe=6,
        view_pe=6,
        fea_pe=6,
        featureC=128,
        step_ratio=2.0,
        fea2denseAct="softplus",
    ):
        super(TensorBase, self).__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device = device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio
        self.tSizeFixed = tSize

        self.update_stepSize(gridSize, tSize)

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.comp_w = [1, 1, 1]

        self.init_svd_volume(gridSize[0], device)

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = (
            shadingMode,
            pos_pe,
            view_pe,
            fea_pe,
            featureC,
        )
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == "MLP_PE":
            self.renderModule = MLPRender_PE(
                self.app_dim, view_pe, pos_pe, featureC
            ).to(device)
        elif shadingMode == "MLP_Fea":
            self.renderModule = MLPRender_Fea(
                self.app_dim, view_pe, fea_pe, featureC
            ).to(device)
        elif shadingMode == "MLP_Fea_TimeEmbedding":
            self.renderModule = MLPRender_Fea_TimeEmbedding(
                self.app_dim, view_pe, fea_pe, featureC
            ).to(device)
        elif shadingMode == "MLP_Fea_late_view":
            self.renderModule = MLPRender_Fea_late_view(
                self.app_dim, view_pe, fea_pe, featureC
            ).to(device)
        elif shadingMode == "MLP_Fea_woView":
            self.renderModule = MLPRender_Fea_woView(
                self.app_dim, view_pe, fea_pe, featureC
            ).to(device)
        elif shadingMode == "MLP":
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == "RGB":
            assert self.app_dim == 3
            self.renderModule = RGBRender
        else:
            print("Unrecognized shading module")
            exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)

    def update_stepSize(self, gridSize, tSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

        self.tSize = torch.unsqueeze(torch.tensor(tSize), 0).to(
            self.device
        )  # fix the voxel size in t dimension
        print("t grid size: ", self.tSize)

    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass

    def get_forward_backward_scene_flow(self, unnormalized_pts, t_sampled):
        pass

    def get_forward_backward_scene_flow_point(
        self, unnormalized_pts, t_sampled, weights, rays
    ):
        pass

    def get_forward_backward_scene_flow_point_single(self, pts_map_NDC, t_sampled):
        pass

    def warp_coordinate(self, xyz_sampled, t_sampled):
        pass

    def compute_blendingfeature(self, xyz_sampled, t_sampled, time_embedding_sampled):
        pass

    def compute_blendingfeature_noVoxel(
        self, xyz_sampled, t_sampled, time_embedding_sampled
    ):
        pass

    def compute_densityfeature(self, xyz_sampled, t_sampled, time_embedding_sampled):
        pass

    def compute_appfeature(self, xyz_sampled, t_sampled):
        pass

    def normalize_coord(self, xyz_sampled):
        return (
            xyz_sampled - self.aabb.to(xyz_sampled.get_device())[0]
        ) * self.invaabbSize.to(xyz_sampled.get_device()) - 1

    def unnormalize_coord(self, xyz_sampled):
        return (xyz_sampled + 1) / self.invaabbSize.to(
            xyz_sampled.get_device()
        ) + self.aabb.to(xyz_sampled.get_device())[0]

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        pass

    def get_kwargs(self):
        return {
            "aabb": self.aabb,
            "gridSize": self.gridSize.tolist(),
            "tSize": self.tSize.item(),
            "density_n_comp": self.density_n_comp,
            "appearance_n_comp": self.app_n_comp,
            "app_dim": self.app_dim,
            "density_shift": self.density_shift,
            "alphaMask_thres": self.alphaMask_thres,
            "distance_scale": self.distance_scale,
            "rayMarch_weight_thres": self.rayMarch_weight_thres,
            "fea2denseAct": self.fea2denseAct,
            "near_far": self.near_far,
            "step_ratio": self.step_ratio,
            "shadingMode": self.shadingMode,
            "pos_pe": self.pos_pe,
            "view_pe": self.view_pe,
            "fea_pe": self.fea_pe,
            "featureC": self.featureC,
        }

    def save(self, se3_poses, focal_ratio_refine, path):
        kwargs = self.get_kwargs()
        kwargs["se3_poses"] = se3_poses
        kwargs["focal_ratio_refine"] = focal_ratio_refine
        ckpt = {"kwargs": kwargs, "state_dict": self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({"alphaMask.shape": alpha_volume.shape})
            ckpt.update({"alphaMask.mask": np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({"alphaMask.aabb": self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if "alphaMask.aabb" in ckpt.keys():
            length = np.prod(ckpt["alphaMask.shape"])
            alpha_volume = torch.from_numpy(
                np.unpackbits(ckpt["alphaMask.mask"])[:length].reshape(
                    ckpt["alphaMask.shape"]
                )
            )
            self.alphaMask = AlphaGridMask(
                self.device,
                ckpt["alphaMask.aabb"].to(self.device),
                alpha_volume.float().to(self.device),
            )
        self.load_state_dict(ckpt["state_dict"])

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = (
            (self.aabb.to(rays_pts.get_device())[0] > rays_pts)
            | (rays_pts > self.aabb.to(rays_pts.get_device())[1])
        ).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = t_min[..., None] + step

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(
            dim=-1
        )

        return rays_pts, interpx, ~mask_outbbox

    def sample_ray_contracted(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        inner_N_samples = N_samples - N_samples // 2
        outer_N_samples = N_samples // 2
        # inner
        interpx_inner = (
            torch.linspace(near, 2.0, inner_N_samples + 1).unsqueeze(0).to(rays_o)
        )
        if is_train:
            interpx_inner[:, :-1] += (
                torch.rand_like(interpx_inner).to(rays_o)
                * ((2.0 - near) / inner_N_samples)
            )[:, :-1]
        interpx_inner = (interpx_inner[:, 1:] + interpx_inner[:, :-1]) * 0.5
        # sample outer
        rng = torch.arange(outer_N_samples + 1)[None].float()
        if is_train:
            rng[:, :-1] += (torch.rand_like(rng).to(rng))[:, :-1]
        rng = torch.flip(rng, [1])
        rng = (rng[:, 1:] + rng[:, :-1]) * 0.5
        interpx_outer = 1.0 / (
            1 / (far) + (1 / 2.0 - 1 / (far)) * rng / outer_N_samples
        ).to(rays_o.device)
        interpx = torch.cat((interpx_inner, interpx_outer), -1)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]

        rays_pts_norm, _ = torch.max(torch.abs(rays_pts.clone()), dim=-1)
        contract_mask = rays_pts_norm > 1.0
        rays_pts[contract_mask] = (2 - 1 / rays_pts_norm[contract_mask])[..., None] * (
            rays_pts[contract_mask] / rays_pts_norm[contract_mask][..., None]
        )

        mask_outbbox = torch.zeros_like(rays_pts[..., 0]) > 0
        return rays_pts, interpx, ~mask_outbbox

    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, gridSize[0]),
                torch.linspace(0, 1, gridSize[1]),
                torch.linspace(0, 1, gridSize[2]),
            ),
            -1,
        ).to(self.device)
        dense_xyz = self.aabb[0] * (1 - samples) + self.aabb[1] * samples

        alpha = torch.zeros_like(dense_xyz[..., 0])
        alpha = torch.tile(
            torch.unsqueeze(alpha, -1), (1, 1, 1, int(self.tSize.item()))
        )
        for i in range(gridSize[0]):
            for t_idx, t in enumerate(range(self.tSize.item())):
                alpha[i, :, :, t_idx] = self.compute_alpha(
                    dense_xyz[i].view(-1, 3),
                    t / (self.tSize.item() - 1.0) * 2.0 - 1.0,
                    self.stepSize,
                ).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200, 200, 200)):
        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = torch.permute(alpha[0], (0, 4, 1, 2, 3))
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1)
        alpha = torch.permute(alpha[0], (1, 2, 3, 0))
        alpha[alpha >= self.alphaMask_thres] = 1
        alpha[alpha < self.alphaMask_thres] = 0

        xyz_min_all_t = []
        xyz_max_all_t = []

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha, self.tSize.item())

        for t in range(self.tSize.item()):
            valid_xyz = dense_xyz[alpha[..., t] > 0.5]

            xyz_min = valid_xyz.amin(0)
            xyz_max = valid_xyz.amax(0)

            xyz_min_all_t.append(xyz_min)
            xyz_max_all_t.append(xyz_max)

        xyz_min, xyz_max = torch.stack(xyz_min_all_t, 0).amin(0), torch.stack(
            xyz_max_all_t, 0
        ).amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(
            f"bbox: {xyz_min, xyz_max} alpha rest %%%f" % (total / total_voxels * 100)
        )
        return new_aabb

    @torch.no_grad()
    def filtering_rays(
        self, all_rays, all_rgbs, N_samples=256, chunk=10240 * 5, bbox_only=False
    ):
        print("========> filtering rays ...")
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(
                    -1
                )  # .clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(
                    -1
                )  # .clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _, _ = self.sample_ray(
                    rays_o, rays_d, N_samples=N_samples, is_train=False
                )
                mask_inbbox = (
                    self.alphaMask.sample_alpha(xyz_sampled).view(
                        xyz_sampled.shape[:-1]
                    )
                    > 0
                ).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(
            f"Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}"
        )
        return all_rays[mask_filtered], all_rgbs[mask_filtered]

    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def compute_alpha(self, xyz_locs, t, length=1):
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs, t)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            t_sampled = t * torch.ones(xyz_sampled.shape).to(self.device)[..., 0]
            sigma_feature = self.compute_densityfeature(xyz_sampled, t_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma

        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])

        return alpha

    def forward(
        self,
        rays_chunk,
        ts_chunk,
        timeembeddings_chunk,
        xyz_sampled,
        z_vals,
        ray_valid,
        white_bg=True,
        is_train=False,
        ray_type="ndc",
        N_samples=-1,
    ):
        viewdirs = rays_chunk[:, 3:6]
        if ray_type == "ndc":
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        elif ray_type == "contract":
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )
            viewdirs_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * viewdirs_norm
            viewdirs = viewdirs / viewdirs_norm
        else:
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        blending = torch.zeros((*xyz_sampled.shape[:2], 1), device=xyz_sampled.device)

        if ray_valid.any():
            normalized_xyz_sampled = self.normalize_coord(xyz_sampled)
            if timeembeddings_chunk is not None:
                sigma_feature = self.compute_densityfeature(
                    normalized_xyz_sampled[ray_valid],
                    torch.tile(
                        torch.unsqueeze(ts_chunk, -1),
                        (1, normalized_xyz_sampled.shape[1]),
                    )[ray_valid],
                    torch.tile(
                        torch.unsqueeze(timeembeddings_chunk, 1),
                        (1, normalized_xyz_sampled.shape[1], 1),
                    )[ray_valid],
                )
            else:
                sigma_feature = self.compute_densityfeature(
                    normalized_xyz_sampled[ray_valid],
                    torch.tile(
                        torch.unsqueeze(ts_chunk, -1),
                        (1, normalized_xyz_sampled.shape[1]),
                    )[ray_valid],
                    None,
                )

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            app_features = self.compute_appfeature(
                normalized_xyz_sampled[app_mask],
                torch.tile(
                    torch.unsqueeze(ts_chunk, -1), (1, normalized_xyz_sampled.shape[1])
                )[app_mask],
                None,
            )
            if timeembeddings_chunk is not None:
                valid_rgbs = self.renderModule(
                    normalized_xyz_sampled[app_mask],
                    viewdirs[app_mask],
                    app_features,
                    torch.tile(
                        torch.unsqueeze(timeembeddings_chunk, 1),
                        (1, normalized_xyz_sampled.shape[1], 1),
                    )[app_mask],
                )
            else:
                valid_rgbs = self.renderModule(
                    normalized_xyz_sampled[app_mask],
                    viewdirs[app_mask],
                    app_features,
                    torch.tile(
                        torch.unsqueeze(ts_chunk, -1),
                        (1, normalized_xyz_sampled.shape[1]),
                    )[app_mask][..., None],
                )
            rgb[app_mask] = valid_rgbs

        xyz_prime = self.warp_coordinate(
            xyz_sampled,
            torch.tile(torch.unsqueeze(ts_chunk, -1), (1, xyz_sampled.shape[1])),
        )

        pts_ref = xyz_sampled

        if xyz_prime is None:
            return (
                None,
                None,
                None,
                pts_ref,
                weight,
                None,
                rgb,
                sigma,
                z_vals,
                dists * self.distance_scale,
            )

        if ray_valid.any():
            blending_features = self.compute_blendingfeature(
                normalized_xyz_sampled[ray_valid],
                torch.tile(
                    torch.unsqueeze(ts_chunk, -1), (1, normalized_xyz_sampled.shape[1])
                )[ray_valid],
                None,
            )
            blending[ray_valid] = torch.sigmoid(blending_features)[:, None]

        weights_d = weight

        return (
            None,
            None,
            blending[..., 0],
            pts_ref,
            weights_d,
            xyz_prime,
            rgb,
            sigma,
            z_vals,
            dists * self.distance_scale,
        )
