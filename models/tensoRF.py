# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn.functional as F

from .tensorBase import TensorBase, positional_encoding

from renderer import *


class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, tSize, device, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, tSize, device, **kargs)

    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line = self.init_one_svd(
            self.density_n_comp, self.gridSize, 0.1, device
        )
        self.app_plane, self.app_line = self.init_one_svd(
            self.app_n_comp, self.gridSize, 0.1, device
        )
        self.basis_mat = torch.nn.Linear(
            sum(self.app_n_comp), self.app_dim, bias=False
        ).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(
                torch.nn.Parameter(
                    scale
                    * torch.randn(
                        (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                    )
                )
            )  #
            line_coef.append(
                torch.nn.Parameter(
                    scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))
                )
            )

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(
            line_coef
        ).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [
            {"params": self.density_line, "lr": lr_init_spatialxyz},
            {"params": self.density_plane, "lr": lr_init_spatialxyz},
            {"params": self.app_line, "lr": lr_init_spatialxyz},
            {"params": self.app_plane, "lr": lr_init_spatialxyz},
            {"params": self.basis_mat.parameters(), "lr": lr_init_network},
        ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [
                {"params": self.renderModule.parameters(), "lr": lr_init_network}
            ]
        return grad_vars

    def vectorDiffs(self, vector_comps):
        total = 0

        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]

            dotp = torch.matmul(
                vector_comps[idx].view(n_comp, n_size),
                vector_comps[idx].view(n_comp, n_size).transpose(-1, -2),
            )
            non_diagonal = dotp.view(-1)[1:].view(n_comp - 1, n_comp + 1)[..., :-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)

    def density_L1(self):
        A = torch.permute(
            self.density_plane[0][..., None]
            @ self.density_line[0][..., 0][:, :, None, None],
            (0, 1, 3, 2, 4),
        )
        B = torch.permute(
            self.density_plane[1][..., None]
            @ self.density_line[1][..., 0][:, :, None, None],
            (0, 1, 3, 4, 2),
        )
        C = torch.permute(
            self.density_plane[2][..., None]
            @ self.density_line[2][..., 0][:, :, None, None],
            (0, 1, 4, 3, 2),
        )
        return torch.mean(
            torch.abs(self.feature2density(torch.sum(torch.cat([A, B, C], 1), 1)))
        )

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total
                + reg(self.density_plane[idx]) * 1e-2
                + reg(self.density_line[idx]) * 1e-3
            )
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = (
                total + reg(self.app_plane[idx]) * 1e-2 + reg(self.app_line[idx]) * 1e-3
            )
        return total

    def compute_densityfeature(self, xyz_sampled, t_sampled, time_embedding_sampled):
        # plane + line basis
        coordinate_plane = torch.stack(
            (
                xyz_sampled[..., self.matMode[0]],
                xyz_sampled[..., self.matMode[1]],
                xyz_sampled[..., self.matMode[2]],
            )
        ).view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(
                self.density_plane[idx_plane],
                coordinate_plane[[idx_plane]],
                align_corners=True,
            ).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(
                self.density_line[idx_plane],
                coordinate_line[[idx_plane]],
                align_corners=True,
            ).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(
                plane_coef_point * line_coef_point, dim=0
            )

        return sigma_feature

    def compute_appfeature(self, xyz_sampled, t_sampled, time_embedding_sampled):
        # plane + line basis
        coordinate_plane = torch.stack(
            (
                xyz_sampled[..., self.matMode[0]],
                xyz_sampled[..., self.matMode[1]],
                xyz_sampled[..., self.matMode[2]],
            )
        ).view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(
                F.grid_sample(
                    self.app_plane[idx_plane],
                    coordinate_plane[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            line_coef_point.append(
                F.grid_sample(
                    self.app_line[idx_plane],
                    coordinate_line[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_sampled.shape[:1])
            )
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(
            line_coef_point
        )

        return self.basis_mat((plane_coef_point * line_coef_point).T)

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    plane_coef[i].data,
                    size=(res_target[mat_id_1], res_target[mat_id_0]),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    line_coef[i].data,
                    size=(res_target[vec_id], 1),
                    mode="bilinear",
                    align_corners=True,
                )
            )

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(
            self.app_plane, self.app_line, res_target
        )
        self.density_plane, self.density_line = self.up_sampling_VM(
            self.density_plane, self.density_line, res_target
        )

        self.update_stepSize(res_target, 1)
        print(f"upsamping to {res_target}")

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (
            xyz_max - self.aabb[0]
        ) / self.units
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[..., t_l[mode0] : b_r[mode0], :]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[..., t_l[mode0] : b_r[mode0], :]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[
                    ..., t_l[mode1] : b_r[mode1], t_l[mode0] : b_r[mode0]
                ]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[
                    ..., t_l[mode1] : b_r[mode1], t_l[mode0] : b_r[mode0]
                ]
            )

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]), 1)


class TensorVMSplit_TimeEmbedding(TensorBase):
    def __init__(self, aabb, gridSize, tSize, device, **kargs):
        super(TensorVMSplit_TimeEmbedding, self).__init__(
            aabb, gridSize, tSize, device, **kargs
        )

        self.layer1 = torch.nn.Linear(1 + 8 * 2 * 1, 64).to(device)
        self.layer2 = torch.nn.Linear(64, 30).to(device)
        self.layer3 = torch.nn.Linear((3 + 10 * 2 * 3) + 30, 64).to(device)
        self.layer4 = torch.nn.Linear(64, 64).to(device)
        self.layer5 = torch.nn.Linear(64, 3).to(device)

        self.density_layer1 = torch.nn.Linear(
            sum(self.density_n_comp) * 3 + 3 + 10 * 2 * 3 + 1 + 8 * 2 * 1, 64
        ).to(device)
        self.density_layer2 = torch.nn.Linear(64, 1).to(device)

        self.blending_layer1 = torch.nn.Linear(
            sum(self.density_n_comp) * 3 + 3 + 10 * 2 * 3 + 1 + 8 * 2 * 1, 64
        ).to(device)
        self.blending_layer2 = torch.nn.Linear(64, 1).to(device)

        layer_sf_1 = torch.nn.Linear(
            4 * 2 * 4 + 4, 64
        )  # 16 frequencies, 4 (x, y, z, t)
        layer_sf_2 = torch.nn.Linear(64, 64)
        layer_sf_3 = torch.nn.Linear(64, 64)
        layer_sf_4 = torch.nn.Linear(64, 6)
        self.scene_flow_mlp = torch.nn.Sequential(
            layer_sf_1,
            torch.nn.ReLU(inplace=True),
            layer_sf_2,
            torch.nn.ReLU(inplace=True),
            layer_sf_3,
            torch.nn.ReLU(inplace=True),
            layer_sf_4,
        ).to(device)

    def init_svd_volume(self, res, device):
        self.blending_plane, self.blending_line = self.init_one_svd(
            self.density_n_comp, self.gridSize, 0.1, device
        )
        self.density_plane, self.density_line = self.init_one_svd(
            self.density_n_comp, self.gridSize, 0.1, device
        )
        self.app_plane, self.app_line = self.init_one_svd(
            self.app_n_comp, self.gridSize, 0.1, device
        )
        self.basis_mat = torch.nn.Linear(
            sum(self.app_n_comp) * 3, self.app_dim, bias=False
        ).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(
                torch.nn.Parameter(
                    scale
                    * torch.randn(
                        (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                    )
                )
            )  #
            line_coef.append(
                torch.nn.Parameter(
                    scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))
                )
            )

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(
            line_coef
        ).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [
            {"params": self.density_line, "lr": lr_init_spatialxyz},
            {"params": self.density_plane, "lr": lr_init_spatialxyz},
            {"params": self.blending_line, "lr": lr_init_spatialxyz},
            {"params": self.blending_plane, "lr": lr_init_spatialxyz},
            {"params": self.app_line, "lr": lr_init_spatialxyz},
            {"params": self.app_plane, "lr": lr_init_spatialxyz},
            {"params": self.basis_mat.parameters(), "lr": lr_init_network},
            {"params": self.scene_flow_mlp.parameters(), "lr": lr_init_network},
            {"params": self.layer1.parameters(), "lr": lr_init_network},
            {"params": self.layer2.parameters(), "lr": lr_init_network},
            {"params": self.layer3.parameters(), "lr": lr_init_network},
            {"params": self.layer4.parameters(), "lr": lr_init_network},
            {"params": self.layer5.parameters(), "lr": lr_init_network},
            {"params": self.density_layer1.parameters(), "lr": lr_init_network},
            {"params": self.density_layer2.parameters(), "lr": lr_init_network},
            {"params": self.blending_layer1.parameters(), "lr": lr_init_network},
            {"params": self.blending_layer2.parameters(), "lr": lr_init_network},
        ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [
                {"params": self.renderModule.parameters(), "lr": lr_init_network}
            ]
        return grad_vars

    def density_L1(self):
        A = torch.permute(
            self.density_plane[0][..., None]
            @ self.density_line[0][..., 0][:, :, None, None],
            (0, 1, 3, 2, 4),
        )
        B = torch.permute(
            self.density_plane[1][..., None]
            @ self.density_line[1][..., 0][:, :, None, None],
            (0, 1, 3, 4, 2),
        )
        C = torch.permute(
            self.density_plane[2][..., None]
            @ self.density_line[2][..., 0][:, :, None, None],
            (0, 1, 4, 3, 2),
        )
        return torch.mean(
            torch.abs(self.feature2density(torch.sum(torch.cat([A, B, C], 1), 1)))
        )

    def blending_L1(self):
        A = torch.permute(
            self.blending_plane[0][..., None]
            @ self.blending_line[0][..., 0][:, :, None, None],
            (0, 1, 3, 2, 4),
        )
        B = torch.permute(
            self.blending_plane[1][..., None]
            @ self.blending_line[1][..., 0][:, :, None, None],
            (0, 1, 3, 4, 2),
        )
        C = torch.permute(
            self.blending_plane[2][..., None]
            @ self.blending_line[2][..., 0][:, :, None, None],
            (0, 1, 4, 3, 2),
        )
        return torch.mean(
            torch.abs(self.feature2density(torch.sum(torch.cat([A, B, C], 1), 1)))
        )

    def TV_loss_blending(self, reg):
        total = 0
        for idx in range(len(self.blending_plane)):
            total = (
                total
                + reg(self.blending_plane[idx]) * 1e-2
                + reg(self.blending_line[idx]) * 1e-3
            )
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total
                + reg(self.density_plane[idx]) * 1e-2
                + reg(self.density_line[idx]) * 1e-3
            )
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = (
                total + reg(self.app_plane[idx]) * 1e-2 + reg(self.app_line[idx]) * 1e-3
            )
        return total

    def get_forward_backward_scene_flow(self, unnormalized_pts, t_sampled):
        flattened_unnormalized_pts = unnormalized_pts.view(-1, 3)
        flattened_t_sampled = (
            (t_sampled[:, None]).repeat(1, unnormalized_pts.shape[1]).view(-1, 1)
        )
        indata = [self.normalize_coord(flattened_unnormalized_pts)]
        indata += [
            positional_encoding(self.normalize_coord(flattened_unnormalized_pts), 4)
        ]
        indata += [flattened_t_sampled]
        indata += [positional_encoding(flattened_t_sampled, 4)]
        scene_flow = self.scene_flow_mlp(torch.cat(indata, -1)).view(
            unnormalized_pts.shape[0], unnormalized_pts.shape[1], 6
        )
        scene_flow_f = scene_flow[..., 0:3]
        scene_flow_b = scene_flow[..., 3:6]
        return scene_flow_f, scene_flow_b

    def get_forward_backward_scene_flow_point(
        self, unnormalized_pts, t_sampled, weights, rays
    ):
        flattened_unnormalized_pts = unnormalized_pts.view(-1, 3)
        flattened_t_sampled = (
            (t_sampled[:, None]).repeat(1, unnormalized_pts.shape[1]).view(-1, 1)
        )
        indata = [self.normalize_coord(flattened_unnormalized_pts)]
        indata += [
            positional_encoding(self.normalize_coord(flattened_unnormalized_pts), 4)
        ]
        indata += [flattened_t_sampled]
        indata += [positional_encoding(flattened_t_sampled, 4)]
        scene_flow_f_pts = self.scene_flow_mlp(torch.cat(indata, -1)).view(
            unnormalized_pts.shape[0], unnormalized_pts.shape[1], 6
        )
        scene_flow_b_pts = -scene_flow_f_pts
        acc_map = torch.sum(weights, -1)[:, None]
        scene_flow_f_pt = torch.sum(
            weights[..., None] * (unnormalized_pts + scene_flow_f_pts), -2
        )
        scene_flow_f_pt = scene_flow_f_pt + (1.0 - acc_map) * (
            rays[:, :3] + rays[:, 3:]
        )
        scene_flow_b_pt = torch.sum(
            weights[..., None] * (unnormalized_pts + scene_flow_b_pts), -2
        )
        scene_flow_b_pt = scene_flow_b_pt + (1.0 - acc_map) * (
            rays[:, :3] + rays[:, 3:]
        )
        pts_map_NDC = torch.sum(weights[..., None] * unnormalized_pts, -2)
        pts_map_NDC = pts_map_NDC + (1.0 - acc_map) * (rays[:, :3] + rays[:, 3:])
        return (
            scene_flow_f_pt,
            scene_flow_b_pt,
            (
                torch.abs(scene_flow_f_pt - pts_map_NDC)
                + torch.abs(scene_flow_b_pt - pts_map_NDC)
            )
            / 2.0,
        )

    def get_forward_backward_scene_flow_point_single(self, pts_map_NDC, t_sampled):
        indata = [self.normalize_coord(pts_map_NDC)]
        indata += [positional_encoding(self.normalize_coord(pts_map_NDC), 4)]
        indata += [t_sampled[..., None]]
        indata += [positional_encoding(t_sampled[..., None], 4)]
        scene_flow = self.scene_flow_mlp(torch.cat(indata, -1))
        scene_flow_f = scene_flow[..., 0:3]
        scene_flow_b = scene_flow[..., 3:6]
        return (
            pts_map_NDC + scene_flow_f,
            pts_map_NDC + scene_flow_b,
            scene_flow_f,
            scene_flow_b,
        )

    def warp_coordinate(self, unnormalized_xyz_sampled, t_sampled):
        indata = [t_sampled[..., None]]
        indata += [positional_encoding(t_sampled[..., None], 8)]
        mlp_in = torch.cat(indata, dim=-1)
        t_out = self.layer2(torch.nn.ReLU(inplace=True)(self.layer1(mlp_in)))
        indata_xyz = [self.normalize_coord(unnormalized_xyz_sampled)]
        indata_xyz += [
            positional_encoding(self.normalize_coord(unnormalized_xyz_sampled), 10)
        ]
        delta_xyz = self.layer5(
            torch.nn.ReLU(inplace=True)(
                self.layer4(
                    torch.nn.ReLU(inplace=True)(
                        self.layer3(
                            torch.cat([torch.cat(indata_xyz, -1), t_out], dim=-1)
                        )
                    )
                )
            )
        )
        return unnormalized_xyz_sampled + delta_xyz

    def compute_blendingfeature(self, xyz_sampled, t_sampled, time_embedding_sampled):
        xyz_prime_sampled = self.normalize_coord(
            self.warp_coordinate(self.unnormalize_coord(xyz_sampled), t_sampled)
        )

        # plane + line basis
        coordinate_plane = torch.stack(
            (
                xyz_prime_sampled[..., self.matMode[0]],
                xyz_prime_sampled[..., self.matMode[1]],
                xyz_prime_sampled[..., self.matMode[2]],
            )
        ).view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (
                xyz_prime_sampled[..., self.vecMode[0]],
                xyz_prime_sampled[..., self.vecMode[1]],
                xyz_prime_sampled[..., self.vecMode[2]],
            )
        )
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        # stride 1
        for idx_plane in range(len(self.blending_plane)):
            plane_coef_point.append(
                F.grid_sample(
                    self.blending_plane[idx_plane],
                    coordinate_plane[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
            line_coef_point.append(
                F.grid_sample(
                    self.blending_line[idx_plane],
                    coordinate_line[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
        # stride 2
        for idx_plane in range(len(self.blending_plane)):
            plane_coef_point.append(
                F.grid_sample(
                    self.blending_plane[idx_plane][:, :, ::2, ::2],
                    coordinate_plane[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
            line_coef_point.append(
                F.grid_sample(
                    self.blending_line[idx_plane][:, :, ::2],
                    coordinate_line[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
        # stride 4
        for idx_plane in range(len(self.blending_plane)):
            plane_coef_point.append(
                F.grid_sample(
                    self.blending_plane[idx_plane][:, :, ::4, ::4],
                    coordinate_plane[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
            line_coef_point.append(
                F.grid_sample(
                    self.blending_line[idx_plane][:, :, ::4],
                    coordinate_line[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(
            line_coef_point
        )

        indata = [(plane_coef_point * line_coef_point).T]
        indata += [xyz_sampled]
        indata += [positional_encoding(xyz_sampled, 10)]
        indata += [t_sampled[..., None]]
        indata += [positional_encoding(t_sampled[..., None], 8)]

        sigma_feature = self.blending_layer2(
            torch.nn.ReLU(inplace=True)(self.blending_layer1(torch.cat(indata, -1)))
        )
        return sigma_feature[..., 0]

    def compute_blendingfeature_noVoxel(
        self, xyz_sampled, t_sampled, time_embedding_sampled
    ):
        indata = [xyz_sampled]
        indata += [positional_encoding(xyz_sampled, 5)]
        indata += [t_sampled[..., None]]
        indata += [positional_encoding(t_sampled[..., None], 4)]

        sigma_feature = self.blending_noVoxel_layer4(
            torch.nn.ReLU(inplace=True)(
                self.blending_noVoxel_layer1(torch.cat(indata, -1))
            )
        )
        return sigma_feature[..., 0]

    def compute_densityfeature(self, xyz_sampled, t_sampled, time_embedding_sampled):
        xyz_prime_sampled = self.normalize_coord(
            self.warp_coordinate(self.unnormalize_coord(xyz_sampled), t_sampled)
        )

        # plane + line basis
        coordinate_plane = torch.stack(
            (
                xyz_prime_sampled[..., self.matMode[0]],
                xyz_prime_sampled[..., self.matMode[1]],
                xyz_prime_sampled[..., self.matMode[2]],
            )
        ).view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (
                xyz_prime_sampled[..., self.vecMode[0]],
                xyz_prime_sampled[..., self.vecMode[1]],
                xyz_prime_sampled[..., self.vecMode[2]],
            )
        )
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        # stride 1
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point.append(
                F.grid_sample(
                    self.density_plane[idx_plane],
                    coordinate_plane[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
            line_coef_point.append(
                F.grid_sample(
                    self.density_line[idx_plane],
                    coordinate_line[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
        # stride 2
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point.append(
                F.grid_sample(
                    self.density_plane[idx_plane][:, :, ::2, ::2],
                    coordinate_plane[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
            line_coef_point.append(
                F.grid_sample(
                    self.density_line[idx_plane][:, :, ::2],
                    coordinate_line[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
        # stride 4
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point.append(
                F.grid_sample(
                    self.density_plane[idx_plane][:, :, ::4, ::4],
                    coordinate_plane[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
            line_coef_point.append(
                F.grid_sample(
                    self.density_line[idx_plane][:, :, ::4],
                    coordinate_line[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(
            line_coef_point
        )

        indata = [(plane_coef_point * line_coef_point).T]
        indata += [xyz_sampled]
        indata += [positional_encoding(xyz_sampled, 10)]
        indata += [t_sampled[..., None]]
        indata += [positional_encoding(t_sampled[..., None], 8)]

        sigma_feature = self.density_layer2(
            torch.nn.ReLU(inplace=True)(self.density_layer1(torch.cat(indata, -1)))
        )
        return sigma_feature[..., 0]

    def compute_appfeature(self, xyz_sampled, t_sampled, time_embedding_sampled):
        xyz_prime_sampled = self.normalize_coord(
            self.warp_coordinate(self.unnormalize_coord(xyz_sampled), t_sampled)
        )

        # plane + line basis
        coordinate_plane = torch.stack(
            (
                xyz_prime_sampled[..., self.matMode[0]],
                xyz_prime_sampled[..., self.matMode[1]],
                xyz_prime_sampled[..., self.matMode[2]],
            )
        ).view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (
                xyz_prime_sampled[..., self.vecMode[0]],
                xyz_prime_sampled[..., self.vecMode[1]],
                xyz_prime_sampled[..., self.vecMode[2]],
            )
        )
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        # stride 1
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(
                F.grid_sample(
                    self.app_plane[idx_plane],
                    coordinate_plane[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
            line_coef_point.append(
                F.grid_sample(
                    self.app_line[idx_plane],
                    coordinate_line[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
        # stride 2
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(
                F.grid_sample(
                    self.app_plane[idx_plane][:, :, ::2, ::2],
                    coordinate_plane[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
            line_coef_point.append(
                F.grid_sample(
                    self.app_line[idx_plane][:, :, ::2],
                    coordinate_line[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
        # stride 4
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(
                F.grid_sample(
                    self.app_plane[idx_plane][:, :, ::4, ::4],
                    coordinate_plane[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
            line_coef_point.append(
                F.grid_sample(
                    self.app_line[idx_plane][:, :, ::4],
                    coordinate_line[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_prime_sampled.shape[:1])
            )
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(
            line_coef_point
        )

        return self.basis_mat((plane_coef_point * line_coef_point).T)

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    plane_coef[i].data,
                    size=(res_target[mat_id_1], res_target[mat_id_0]),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    line_coef[i].data,
                    size=(res_target[vec_id], 1),
                    mode="bilinear",
                    align_corners=True,
                )
            )

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(
            self.app_plane, self.app_line, res_target
        )
        self.density_plane, self.density_line = self.up_sampling_VM(
            self.density_plane, self.density_line, res_target
        )
        self.blending_plane, self.blending_line = self.up_sampling_VM(
            self.blending_plane, self.blending_line, res_target
        )

        self.update_stepSize(res_target, 1)
        print(f"upsamping to {res_target}")

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (
            xyz_max - self.aabb[0]
        ) / self.units
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[..., t_l[mode0] : b_r[mode0], :]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[..., t_l[mode0] : b_r[mode0], :]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[
                    ..., t_l[mode1] : b_r[mode1], t_l[mode0] : b_r[mode0]
                ]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[
                    ..., t_l[mode1] : b_r[mode1], t_l[mode0] : b_r[mode0]
                ]
            )

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]), 1)
