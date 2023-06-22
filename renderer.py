# Copyright (c) Meta Platforms, Inc. and affiliates.

import sys
import os
import imageio
import numpy as np
import torch
from tqdm.auto import tqdm

from dataLoader.ray_utils import get_rays
from dataLoader.ray_utils import (
    get_ray_directions_blender,
    get_rays,
    ndc_rays_blender,
    get_rays_with_batch,
)
from dataLoader.ray_utils import ndc_rays_blender
from models.tensoRF import *
from utils import visualize_depth_numpy, visualize_depth, rgb_lpips, rgb_ssim
from camera import lie, pose
from flow_viz import flow_to_image


def OctreeRender_trilinear_fast(
    rays,
    ts,
    timeembeddings,
    tensorf,
    xyz_sampled,
    z_vals_input,
    ray_valid,
    chunk=4096,
    N_samples=-1,
    ray_type="ndc",
    white_bg=True,
    is_train=False,
    device="cuda",
):
    (
        rgbs,
        depth_maps,
        blending_maps,
        pts_refs,
        weights_ds,
        delta_xyzs,
        rgb_points,
        sigmas,
        z_vals,
        dists,
    ) = ([], [], [], [], [], [], [], [], [], [])
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        ts_chunk = ts[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        xyz_sampled_chunk = xyz_sampled[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(
            device
        )
        z_vals_chunk = z_vals_input[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(
            device
        )
        ray_valid_chunk = ray_valid[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(
            device
        )

        timeembeddings_chunk = None
        if timeembeddings is not None:
            timeembeddings_chunk = timeembeddings[
                chunk_idx * chunk : (chunk_idx + 1) * chunk
            ].to(device)

        (
            rgb_map,
            depth_map,
            blending_map,
            pts_ref,
            weights_d,
            xyz_prime,
            rgb_point,
            sigma,
            z_val,
            dist,
        ) = tensorf(
            rays_chunk,
            ts_chunk,
            timeembeddings_chunk,
            xyz_sampled_chunk,
            z_vals_chunk,
            ray_valid_chunk,
            is_train=is_train,
            white_bg=white_bg,
            ray_type=ray_type,
            N_samples=N_samples,
        )
        delta_xyz = xyz_prime - xyz_sampled_chunk

        if blending_map is None:
            rgbs.append(rgb_map)
            depth_maps.append(depth_map)
            pts_refs.append(pts_ref)
            weights_ds.append(weights_d)
            rgb_points.append(rgb_point)
            sigmas.append(sigma)
            z_vals.append(z_val)
            dists.append(dist)
            continue

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        blending_maps.append(blending_map)
        pts_refs.append(pts_ref)
        weights_ds.append(weights_d)
        delta_xyzs.append(delta_xyz)
        rgb_points.append(rgb_point)
        sigmas.append(sigma)
        z_vals.append(z_val)
        dists.append(dist)
    if len(blending_maps) == 0:
        return (
            torch.cat(rgbs),
            torch.cat(depth_maps),
            None,
            torch.cat(pts_refs),
            torch.cat(weights_ds),
            None,
            None,
            torch.cat(rgb_points),
            torch.cat(sigmas),
            torch.cat(z_vals),
            torch.cat(dists),
        )
    else:
        return (
            torch.cat(rgbs),
            torch.cat(depth_maps),
            torch.cat(blending_maps),
            torch.cat(pts_refs),
            torch.cat(weights_ds),
            torch.cat(delta_xyzs),
            None,
            torch.cat(rgb_points),
            torch.cat(sigmas),
            torch.cat(z_vals),
            torch.cat(dists),
        )


def sampleXYZ(tensorf, rays_train, N_samples, ray_type="ndc", is_train=False):
    if ray_type == "ndc":
        xyz_sampled, z_vals, ray_valid = tensorf.sample_ray_ndc(
            rays_train[:, :3],
            rays_train[:, 3:6],
            is_train=is_train,
            N_samples=N_samples,
        )
    elif ray_type == "contract":
        xyz_sampled, z_vals, ray_valid = tensorf.sample_ray_contracted(
            rays_train[:, :3],
            rays_train[:, 3:6],
            is_train=is_train,
            N_samples=N_samples,
        )
    else:
        xyz_sampled, z_vals, ray_valid = tensorf.sample_ray(
            rays_train[:, :3],
            rays_train[:, 3:6],
            is_train=is_train,
            N_samples=N_samples,
        )
    z_vals = torch.tile(z_vals, (xyz_sampled.shape[0], 1))
    return xyz_sampled, z_vals, ray_valid


def raw2outputs(
    rgb_s,
    sigma_s,
    rgb_d,
    sigma_d,
    dists,
    blending,
    z_vals,
    rays_chunk,
    is_train=False,
    ray_type="ndc",
):
    """Transforms model's predictions to semantically meaningful values.
    Args:
      raw_d: [num_rays, num_samples along ray, 4]. Prediction from Dynamic model.
      raw_s: [num_rays, num_samples along ray, 4]. Prediction from Static model.
      z_vals: [num_rays, num_samples along ray]. Integration time.
      rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
      disp_map: [num_rays]. Disparity map. Inverse of depth map.
      acc_map: [num_rays]. Sum of weights along each ray.
      weights: [num_rays, num_samples]. Weights assigned to each sampled color.
      depth_map: [num_rays]. Estimated distance to object.
    """

    # Function for computing density from model prediction. This value is
    # strictly between [0, 1].
    def raw2alpha(sigma, dist):
        return 1.0 - torch.exp(-sigma * dist)

    # # Add noise to model's predictions for density. Can be used to
    # # regularize network during training (prevents floater artifacts).
    # noise = 0.
    # if raw_noise_std > 0.:
    #     noise = torch.randn(raw_d[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point.
    alpha_d = raw2alpha(sigma_d, dists)  # [N_rays, N_samples]
    alpha_s = raw2alpha(sigma_s, dists)  # [N_rays, N_samples]
    alphas = 1.0 - (1.0 - alpha_s) * (1.0 - alpha_d)  # [N_rays, N_samples]

    T_d = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_d.shape[0], 1)).to(alpha_d.device),
                1.0 - alpha_d + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]
    T_s = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_s.shape[0], 1)).to(alpha_s.device),
                1.0 - alpha_s + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]
    T_full = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha_d.shape[0], 1)).to(alpha_d.device),
                (1.0 - alpha_d * blending) * (1.0 - alpha_s * (1.0 - blending)) + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]

    # Compute weight for RGB of each sample along each ray.  A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    weights_d = alpha_d * T_d
    weights_s = alpha_s * T_s
    weights_d = weights_d / (torch.sum(weights_d, -1, keepdim=True) + 1e-10)
    weights_full = (alpha_d * blending + alpha_s * (1.0 - blending)) * T_full

    # Computed weighted color of each sample along each ray.
    rgb_map_d = torch.sum(weights_d[..., None] * rgb_d, -2)
    rgb_map_s = torch.sum(weights_s[..., None] * rgb_s, -2)
    rgb_map_full = torch.sum(
        (T_full * alpha_d * blending)[..., None] * rgb_d
        + (T_full * alpha_s * (1.0 - blending))[..., None] * rgb_s,
        -2,
    )

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map_d = torch.sum(weights_d, -1)
    acc_map_s = torch.sum(weights_s, -1)
    acc_map_full = torch.sum(weights_full, -1)

    if is_train and torch.rand((1,)) < 0.5:
        rgb_map_d = rgb_map_d + (1.0 - acc_map_d[..., None])
        rgb_map_s = rgb_map_s + (1.0 - acc_map_s[..., None])
        rgb_map_full = rgb_map_full + torch.relu(1.0 - acc_map_full[..., None])

    # Estimated depth map is expected distance.
    depth_map_d = torch.sum(weights_d * z_vals, -1)
    depth_map_s = torch.sum(weights_s * z_vals, -1)
    depth_map_full = torch.sum(weights_full * z_vals, -1)
    if ray_type == "ndc":
        depth_map_d = depth_map_d + (1.0 - acc_map_d) * (
            rays_chunk[..., 2] + rays_chunk[..., -1]
        )
        depth_map_s = depth_map_s + (1.0 - acc_map_s) * (
            rays_chunk[..., 2] + rays_chunk[..., -1]
        )
        depth_map_full = depth_map_full + torch.relu(1.0 - acc_map_full) * (
            rays_chunk[..., 2] + rays_chunk[..., -1]
        )
    elif ray_type == "contract":
        depth_map_d = depth_map_d + (1.0 - acc_map_d) * 256.0
        depth_map_s = depth_map_s + (1.0 - acc_map_s) * 256.0
        depth_map_full = depth_map_full + torch.relu(1.0 - acc_map_full) * 256.0

    rgb_map_d = rgb_map_d.clamp(0, 1)
    rgb_map_s = rgb_map_s.clamp(0, 1)
    rgb_map_full = rgb_map_full.clamp(0, 1)

    # Computed dynamicness
    dynamicness_map = torch.sum(weights_full * blending, -1)
    dynamicness_map = dynamicness_map + torch.relu(1.0 - acc_map_full) * 0.0

    return (
        rgb_map_full,
        depth_map_full,
        acc_map_full,
        weights_full,
        rgb_map_s,
        depth_map_s,
        acc_map_s,
        weights_s,
        rgb_map_d,
        depth_map_d,
        acc_map_d,
        weights_d,
        dynamicness_map,
    )


@torch.no_grad()
def render(
    test_dataset,
    poses_mtx,
    focal_ratio_refine,
    tensorf_static,
    tensorf,
    args,
    renderer,
    savePath=None,
    N_vis=5,
    prtx="",
    N_samples=-1,
    white_bg=False,
    ray_type="ndc",
    compute_extra_metrics=True,
    device="cuda",
):
    (
        rgb_maps_tb,
        depth_maps_tb,
        blending_maps_tb,
        gt_rgbs_tb,
        induced_flow_f_tb,
        induced_flow_b_tb,
        induced_flow_s_f_tb,
        induced_flow_s_b_tb,
        delta_xyz_tb,
        rgb_maps_s_tb,
        depth_maps_s_tb,
        rgb_maps_d_tb,
        depth_maps_d_tb,
    ) = ([], [], [], [], [], [], [], [], [], [], [], [], [])

    W, H = test_dataset.img_wh
    directions = get_ray_directions_blender(
        H, W, [focal_ratio_refine, focal_ratio_refine]
    ).to(
        poses_mtx.device
    )  # (H, W, 3)
    all_rays = []
    for i in range(poses_mtx.shape[0]):
        c2w = poses_mtx[i]
        rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
        if ray_type == "ndc":
            rays_o, rays_d = ndc_rays_blender(
                H, W, focal_ratio_refine, 1.0, rays_o, rays_d
            )
        all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
    all_rays = torch.stack(all_rays, 0)
    if args.multiview_dataset:
        # duplicate poses for multiple time instances
        all_rays = torch.tile(all_rays, (args.N_voxel_t, 1, 1))

    ii, jj = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    grid = torch.from_numpy(np.stack([ii, jj], -1)).view(-1, 2).to(device)

    img_eval_interval = 1 if N_vis < 0 else max(all_rays.shape[0] // N_vis, 1)
    idxs = list(range(0, all_rays.shape[0], img_eval_interval))
    for idx, samples in enumerate(all_rays[0::img_eval_interval]):
        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])
        ts = test_dataset.all_ts[idx].view(-1)

        N_rays_all = rays.shape[0]
        chunk = 1024
        pose_f = poses_mtx[min(idx + 1, poses_mtx.shape[0] - 1), :3, :4]
        pose_b = poses_mtx[max(idx - 1, 0), :3, :4]
        rgb_map_list = []
        rgb_map_s_list = []
        rgb_map_d_list = []
        dynamicness_map_list = []
        depth_map_list = []
        depth_map_s_list = []
        depth_map_d_list = []
        induced_flow_f_list = []
        induced_flow_b_list = []
        induced_flow_s_f_list = []
        induced_flow_s_b_list = []
        weights_d_list = []
        delta_xyz_list = []
        for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
            rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
            ts_chunk = ts[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
            grid_chunk = grid[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
            xyz_sampled, z_vals, ray_valid = sampleXYZ(
                tensorf,
                rays_chunk,
                N_samples=N_samples,
                ray_type=ray_type,
                is_train=False,
            )
            # static
            (
                _,
                _,
                _,
                pts_ref_s,
                _,
                _,
                rgb_point_static,
                sigma_static,
                _,
                _,
            ) = tensorf_static(
                rays_chunk,
                ts_chunk,
                None,
                xyz_sampled,
                z_vals,
                ray_valid,
                is_train=False,
                white_bg=white_bg,
                ray_type=ray_type,
                N_samples=N_samples,
            )
            # dynamic
            (
                _,
                _,
                blending,
                pts_ref,
                _,
                xyz_prime,
                rgb_point_dynamic,
                sigma_dynamic,
                z_val_dynamic,
                dist_dynamic,
            ) = tensorf(
                rays_chunk,
                ts_chunk,
                None,
                xyz_sampled,
                z_vals,
                ray_valid,
                is_train=False,
                white_bg=white_bg,
                ray_type=ray_type,
                N_samples=N_samples,
            )
            delta_xyz = xyz_prime - xyz_sampled
            # blending
            (
                rgb_map,
                depth_map_full,
                acc_map_full,
                weights_full,
                rgb_map_s,
                depth_map_s,
                acc_map_s,
                weights_s,
                rgb_map_d,
                depth_map_d,
                acc_map_d,
                weights_d,
                dynamicness_map,
            ) = raw2outputs(
                rgb_point_static,
                sigma_static,
                rgb_point_dynamic.to(device),
                sigma_dynamic.to(device),
                dist_dynamic.to(device),
                blending,
                z_val_dynamic.to(device),
                rays_chunk,
                ray_type=ray_type,
            )
            # scene flow
            scene_flow_f, scene_flow_b = tensorf.module.get_forward_backward_scene_flow(
                pts_ref, ts_chunk
            )
            pts_f = pts_ref + scene_flow_f
            pts_b = pts_ref + scene_flow_b
            induced_flow_f, _ = induce_flow(
                H,
                W,
                focal_ratio_refine,
                torch.tile(pose_f[None], (weights_d.shape[0], 1, 1)),
                weights_d,
                pts_f,
                grid_chunk,
                rays_chunk,
                ray_type=ray_type,
            )
            induced_flow_b, _ = induce_flow(
                H,
                W,
                focal_ratio_refine,
                torch.tile(pose_b[None], (weights_d.shape[0], 1, 1)),
                weights_d,
                pts_b,
                grid_chunk,
                rays_chunk,
                ray_type=ray_type,
            )
            # induced flow for static
            induced_flow_s_f, _ = induce_flow(
                H,
                W,
                focal_ratio_refine,
                torch.tile(pose_f[None], (weights_s.shape[0], 1, 1)),
                weights_s,
                pts_ref_s,
                grid_chunk,
                rays_chunk,
                ray_type=ray_type,
            )
            induced_flow_s_b, _ = induce_flow(
                H,
                W,
                focal_ratio_refine,
                torch.tile(pose_b[None], (weights_s.shape[0], 1, 1)),
                weights_s,
                pts_ref_s,
                grid_chunk,
                rays_chunk,
                ray_type=ray_type,
            )
            # gather chunks
            rgb_map_list.append(rgb_map)
            rgb_map_s_list.append(rgb_map_s)
            rgb_map_d_list.append(rgb_map_d)
            dynamicness_map_list.append(dynamicness_map)
            depth_map_list.append(depth_map_full)
            depth_map_s_list.append(depth_map_s)
            depth_map_d_list.append(depth_map_d)
            induced_flow_f_list.append(induced_flow_f)
            induced_flow_b_list.append(induced_flow_b)
            induced_flow_s_f_list.append(induced_flow_s_f)
            induced_flow_s_b_list.append(induced_flow_s_b)
            weights_d_list.append(weights_d)
            delta_xyz_list.append(delta_xyz)
        rgb_map = torch.cat(rgb_map_list)
        rgb_map_s = torch.cat(rgb_map_s_list)
        rgb_map_d = torch.cat(rgb_map_d_list)
        dynamicness_map = torch.cat(dynamicness_map_list)
        depth_map = torch.cat(depth_map_list)
        depth_map_s = torch.cat(depth_map_s_list)
        depth_map_d = torch.cat(depth_map_d_list)
        induced_flow_f = torch.cat(induced_flow_f_list)
        induced_flow_b = torch.cat(induced_flow_b_list)
        induced_flow_s_f = torch.cat(induced_flow_s_f_list)
        induced_flow_s_b = torch.cat(induced_flow_s_b_list)
        weights_d = torch.cat(weights_d_list)
        delta_xyzs = torch.cat(delta_xyz_list)

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map_s = rgb_map_s.clamp(0.0, 1.0)
        rgb_map_d = rgb_map_d.clamp(0.0, 1.0)
        blending_map = dynamicness_map.clamp(0.0, 1.0)

        rgb_map_s, depth_map_s = rgb_map_s.reshape(H, W, 3), depth_map_s.reshape(H, W)
        rgb_map_d, depth_map_d = rgb_map_d.reshape(H, W, 3), depth_map_d.reshape(H, W)
        rgb_map, depth_map, blending_map = (
            rgb_map.reshape(H, W, 3),
            depth_map.reshape(H, W),
            blending_map.reshape(H, W),
        )
        if ray_type == "contract":
            depth_map_s = -1.0 / (depth_map_s + 1e-6)
            depth_map_d = -1.0 / (depth_map_d + 1e-6)
            depth_map = -1.0 / (depth_map + 1e-6)

        gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)

        viz_induced_flow_f = torch.from_numpy(
            flow_to_image(induced_flow_f.view(H, W, 2).detach().cpu().numpy())
        )
        viz_induced_flow_b = torch.from_numpy(
            flow_to_image(induced_flow_b.view(H, W, 2).detach().cpu().numpy())
        )
        viz_induced_flow_s_f = torch.from_numpy(
            flow_to_image(induced_flow_s_f.view(H, W, 2).detach().cpu().numpy())
        )
        viz_induced_flow_s_b = torch.from_numpy(
            flow_to_image(induced_flow_s_b.view(H, W, 2).detach().cpu().numpy())
        )

        rgb_maps_tb.append(rgb_map)  # HWC
        rgb_maps_s_tb.append(rgb_map_s)  # HWC
        rgb_maps_d_tb.append(rgb_map_d)  # HWC
        depth_maps_tb.append(depth_map)  # CHW
        depth_maps_s_tb.append(depth_map_s)  # CHW
        depth_maps_d_tb.append(depth_map_d)  # CHW
        blending_maps_tb.append(blending_map[None].repeat(3, 1, 1))  # CHW
        gt_rgbs_tb.append(gt_rgb)  # HWC
        induced_flow_f_tb.append(viz_induced_flow_f)
        induced_flow_b_tb.append(viz_induced_flow_b)
        induced_flow_s_f_tb.append(viz_induced_flow_s_f)
        induced_flow_s_b_tb.append(viz_induced_flow_s_b)
        delta_xyz_sum = torch.sum(weights_d[..., None] * delta_xyzs, 1).view(H, W, 3)
        delta_xyz_tb.append(
            ((delta_xyz_sum / torch.max(torch.abs(delta_xyz_sum))) + 1.0) / 2.0
        )

    tmp_list = []
    tmp_list.extend(depth_maps_tb)
    tmp_list.extend(depth_maps_s_tb)
    tmp_list.extend(depth_maps_d_tb)
    all_depth = torch.stack(tmp_list)
    depth_map_min = torch.min(all_depth).item()
    depth_map_max = torch.max(all_depth).item()
    for idx, (depth_map, depth_map_s, depth_map_d) in enumerate(
        zip(depth_maps_tb, depth_maps_s_tb, depth_maps_d_tb)
    ):
        depth_maps_tb[idx] = visualize_depth(
            torch.clamp(depth_map, min=depth_map_min, max=depth_map_max),
            (depth_map_min, depth_map_max),
        )[0]
        depth_maps_s_tb[idx] = visualize_depth(
            torch.clamp(depth_map_s, min=depth_map_min, max=depth_map_max),
            (depth_map_min, depth_map_max),
        )[0]
        depth_maps_d_tb[idx] = visualize_depth(
            torch.clamp(depth_map_d, min=depth_map_min, max=depth_map_max),
            (depth_map_min, depth_map_max),
        )[0]

    monodepth_tb = []
    for i in range(test_dataset.all_disps.shape[0]):
        monodepth_tb.append(visualize_depth(test_dataset.all_disps[i])[0])

    return (
        rgb_maps_tb,
        depth_maps_tb,
        blending_maps_tb,
        gt_rgbs_tb,
        induced_flow_f_tb,
        induced_flow_b_tb,
        induced_flow_s_f_tb,
        induced_flow_s_b_tb,
        delta_xyz_tb,
        rgb_maps_s_tb,
        depth_maps_s_tb,
        rgb_maps_d_tb,
        depth_maps_d_tb,
        monodepth_tb,
    )


@torch.no_grad()
def evaluation(
    test_dataset,
    poses_mtx,
    focal_ratio_refine,
    tensorf_static,
    tensorf,
    args,
    renderer,
    savePath=None,
    N_vis=5,
    prtx="",
    N_samples=-1,
    white_bg=False,
    ray_type="ndc",
    compute_extra_metrics=True,
    device="cuda",
):
    (
        PSNRs,
        rgb_maps,
        depth_maps,
        rgb_maps_s,
        depth_maps_s,
        rgb_maps_d,
        depth_maps_d,
        blending_maps,
    ) = ([], [], [], [], [], [], [], [])
    near_fars = []
    ssims, l_alex, l_vgg = [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)
    os.makedirs(savePath + "_static/rgbd", exist_ok=True)
    os.makedirs(savePath + "_dynamic/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far

    W, H = test_dataset.img_wh
    directions = get_ray_directions_blender(
        H, W, [focal_ratio_refine, focal_ratio_refine]
    ).to(
        poses_mtx.device
    )  # (H, W, 3)
    all_rays = []
    for i in range(poses_mtx.shape[0]):
        c2w = poses_mtx[i]
        rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
        if ray_type == "ndc":
            rays_o, rays_d = ndc_rays_blender(
                H, W, focal_ratio_refine, 1.0, rays_o, rays_d
            )
        all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
    all_rays = torch.stack(all_rays, 0)
    if args.multiview_dataset:
        # duplicate poses for multiple time instances
        all_rays = torch.tile(all_rays, (args.N_voxel_t, 1, 1))

    img_eval_interval = 1 if N_vis < 0 else max(all_rays.shape[0] // N_vis, 1)
    idxs = list(range(0, all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(
        enumerate(all_rays[0::img_eval_interval]), file=sys.stdout
    ):
        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])
        ts = test_dataset.all_ts[idx].view(-1)

        N_rays_all = rays.shape[0]
        chunk = 512
        rgb_map_list = []
        depth_map_list = []
        rgb_map_s_list = []
        depth_map_s_list = []
        rgb_map_d_list = []
        depth_map_d_list = []
        dynamicness_map_list = []
        for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
            rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
            ts_chunk = ts[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
            xyz_sampled, z_vals, ray_valid = sampleXYZ(
                tensorf,
                rays_chunk,
                N_samples=N_samples,
                ray_type=ray_type,
                is_train=False,
            )
            # static
            _, _, _, _, _, _, rgb_point_static, sigma_static, _, _ = tensorf_static(
                rays_chunk,
                ts_chunk,
                None,
                xyz_sampled,
                z_vals,
                ray_valid,
                is_train=False,
                white_bg=white_bg,
                ray_type=ray_type,
                N_samples=N_samples,
            )
            # dynamic
            (
                _,
                _,
                blending,
                pts_ref,
                _,
                _,
                rgb_point_dynamic,
                sigma_dynamic,
                z_val_dynamic,
                dist_dynamic,
            ) = tensorf(
                rays_chunk,
                ts_chunk,
                None,
                xyz_sampled,
                z_vals,
                ray_valid,
                is_train=False,
                white_bg=white_bg,
                ray_type=ray_type,
                N_samples=N_samples,
            )
            # blending
            (
                rgb_map,
                depth_map_full,
                acc_map_full,
                weights_full,
                rgb_map_s,
                depth_map_s,
                acc_map_s,
                weights_s,
                rgb_map_d,
                depth_map_d,
                acc_map_d,
                weights_d,
                dynamicness_map,
            ) = raw2outputs(
                rgb_point_static,
                sigma_static,
                rgb_point_dynamic.to(device),
                sigma_dynamic.to(device),
                dist_dynamic.to(device),
                blending,
                z_val_dynamic.to(device),
                rays_chunk,
                ray_type=ray_type,
            )
            # gather chunks
            rgb_map_list.append(rgb_map)
            depth_map_list.append(depth_map_full)
            rgb_map_s_list.append(rgb_map_s)
            depth_map_s_list.append(depth_map_s)
            rgb_map_d_list.append(rgb_map_d)
            depth_map_d_list.append(depth_map_d)
            dynamicness_map_list.append(dynamicness_map)
        rgb_map = torch.cat(rgb_map_list)
        depth_map = torch.cat(depth_map_list)
        rgb_map_s = torch.cat(rgb_map_s_list)
        depth_map_s = torch.cat(depth_map_s_list)
        rgb_map_d = torch.cat(rgb_map_d_list)
        depth_map_d = torch.cat(depth_map_d_list)
        dynamicness_map = torch.cat(dynamicness_map_list)

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map_s = rgb_map_s.clamp(0.0, 1.0)
        rgb_map_d = rgb_map_d.clamp(0.0, 1.0)
        blending_map = dynamicness_map.clamp(0.0, 1.0)

        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )
        rgb_map_s, depth_map_s = (
            rgb_map_s.reshape(H, W, 3).cpu(),
            depth_map_s.reshape(H, W).cpu(),
        )
        rgb_map_d, depth_map_d = (
            rgb_map_d.reshape(H, W, 3).cpu(),
            depth_map_d.reshape(H, W).cpu(),
        )
        blending_map = blending_map.reshape(H, W).cpu()

        if ray_type == "contract":
            near_fars.append(
                (
                    torch.quantile(depth_map_s, 0.01).item(),
                    torch.quantile(depth_map_s, 0.99).item(),
                )
            )
        else:
            near_fars.append(
                (
                    torch.quantile(1.0 / (depth_map_s + 1e-6), 0.01).item(),
                    torch.quantile(1.0 / (depth_map_s + 1e-6), 0.99).item(),
                )
            )
        if ray_type == "contract":
            depth_map_s = -1.0 / (depth_map_s + 1e-6)
            depth_map_d = -1.0 / (depth_map_d + 1e-6)
            depth_map = -1.0 / (depth_map + 1e-6)

        np.save(f"{savePath}_static/rgbd/{prtx}{idx:03d}.npy", depth_map_s)
        np.save(f"{savePath}/rgbd/{prtx}{idx:03d}.npy", depth_map)

        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "alex", tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "vgg", tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")
        rgb_map_s = (rgb_map_s.numpy() * 255).astype("uint8")
        rgb_map_d = (rgb_map_d.numpy() * 255).astype("uint8")
        blending_map = (blending_map.numpy() * 255).astype("uint8")
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        rgb_maps_s.append(rgb_map_s)
        depth_maps_s.append(depth_map_s)
        rgb_maps_d.append(rgb_map_d)
        depth_maps_d.append(depth_map_d)
        blending_maps.append(blending_map)
        if savePath is not None:
            imageio.imwrite(f"{savePath}/{prtx}{idx:03d}.png", rgb_map, format="png")
            imageio.imwrite(
                f"{savePath}_static/{prtx}{idx:03d}.png", rgb_map_s, format="png"
            )
            imageio.imwrite(
                f"{savePath}_dynamic/{prtx}{idx:03d}.png", rgb_map_d, format="png"
            )
            imageio.imwrite(
                f"{savePath}_dynamic/{prtx}{idx:03d}_blending.png",
                blending_map,
                format="png",
            )

    depth_list = []
    depth_list.extend(depth_maps)
    # tmp_list.extend(depth_maps_s)
    # tmp_list.extend(depth_maps_d)
    # all_depth = torch.stack(tmp_list)
    # depth_map_min = torch.min(all_depth).item()
    # depth_map_max = torch.max(all_depth).item()
    # all_depth = torch.stack(tmp_list)
    # depth_map_min = torch.quantile(all_depth[:, ::4, ::4], 0.05).item()
    # depth_map_max = torch.quantile(all_depth[:, ::4, ::4], 0.95).item()
    # for idx in range(len(rgb_maps)):
    #     depth_maps[idx] = visualize_depth_numpy(torch.clamp(depth_maps[idx], min=depth_map_min, max=depth_map_max).numpy(), (depth_map_min, depth_map_max))[0]
    #     depth_maps_s[idx] = visualize_depth_numpy(torch.clamp(depth_maps_s[idx], min=depth_map_min, max=depth_map_max).numpy(), (depth_map_min, depth_map_max))[0]
    #     depth_maps_d[idx] = visualize_depth_numpy(torch.clamp(depth_maps_d[idx], min=depth_map_min, max=depth_map_max).numpy(), (depth_map_min, depth_map_max))[0]
    #     if savePath is not None:
    #         imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', depth_maps[idx], format='png')
    #         imageio.imwrite(f'{savePath}_static/rgbd/{prtx}{idx:03d}.png', depth_maps_s[idx], format='png')
    #         imageio.imwrite(f'{savePath}_dynamic/rgbd/{prtx}{idx:03d}.png', depth_maps_d[idx], format='png')

    imageio.mimwrite(
        f"{savePath}/{prtx}video.mp4",
        np.stack(rgb_maps),
        fps=30,
        quality=8,
        format="ffmpeg",
        output_params=["-f", "mp4"],
    )
    # imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])
    imageio.mimwrite(
        f"{savePath}_static/{prtx}video.mp4",
        np.stack(rgb_maps_s),
        fps=30,
        quality=8,
        format="ffmpeg",
        output_params=["-f", "mp4"],
    )
    # imageio.mimwrite(f'{savePath}_static/{prtx}depthvideo.mp4', np.stack(depth_maps_s), fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])
    imageio.mimwrite(
        f"{savePath}_dynamic/{prtx}video.mp4",
        np.stack(rgb_maps_d),
        fps=30,
        quality=8,
        format="ffmpeg",
        output_params=["-f", "mp4"],
    )
    # imageio.mimwrite(f'{savePath}_dynamic/{prtx}depthvideo.mp4', np.stack(depth_maps_d), fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])
    # imageio.mimwrite(f'{savePath}_dynamic/{prtx}blending.mp4', np.stack(blending_maps), fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f"{savePath}/{prtx}mean.txt", np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f"{savePath}/{prtx}mean.txt", np.asarray([psnr]))

    return PSNRs, near_fars, depth_list


@torch.no_grad()
def evaluation_path(
    test_dataset,
    focal_ratio_refine,
    tensorf_static,
    tensorf,
    c2ws,
    renderer,
    savePath=None,
    N_vis=5,
    prtx="",
    N_samples=-1,
    white_bg=False,
    ray_type="ndc",
    compute_extra_metrics=True,
    device="cuda",
    change_view=True,
    change_time=None,
    evaluation=False,
    render_focal=None,
):
    (
        PSNRs,
        rgb_maps,
        depth_maps,
        rgb_maps_s,
        depth_maps_s,
        rgb_maps_d,
        depth_maps_d,
        blending_maps,
    ) = ([], [], [], [], [], [], [], [])
    ssims, l_alex, l_vgg = [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)
    os.makedirs(savePath + "_static/rgbd", exist_ok=True)
    os.makedirs(savePath + "_dynamic/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):
        if render_focal is not None:
            focal_ratio_refine = render_focal[idx]

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)

        W, H = test_dataset.img_wh
        directions = get_ray_directions_blender(
            H, W, [focal_ratio_refine, focal_ratio_refine]
        ).to(
            c2w.device
        )  # (H, W, 3)

        rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
        if ray_type == "ndc":
            rays_o, rays_d = ndc_rays_blender(
                H, W, focal_ratio_refine, 1.0, rays_o, rays_d
            )
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        if change_time is "change":
            ts = (
                round(idx / (len(c2ws) - 1) * (len(c2ws) - 1)) / (len(c2ws) - 1) * 2.0
                - 1.0
            ) * torch.ones(
                W * H
            )  # discrete time rendering
        else:
            ts = change_time * torch.ones(W * H)  # first time instance

        N_rays_all = rays.shape[0]
        chunk = 8192
        allposes_refine_train = torch.tile(c2w.cpu()[None], (rays.shape[0], 1, 1))
        rgb_map_list = []
        depth_map_list = []
        rgb_map_s_list = []
        depth_map_s_list = []
        rgb_map_d_list = []
        depth_map_d_list = []
        dynamicness_map_list = []
        weights_s_list = []
        for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
            rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
            ts_chunk = ts[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
            allposes_refine_train_chunk = allposes_refine_train[
                chunk_idx * chunk : (chunk_idx + 1) * chunk
            ].to(device)
            xyz_sampled, z_vals, ray_valid = sampleXYZ(
                tensorf,
                rays_chunk,
                N_samples=N_samples,
                ray_type=ray_type,
                is_train=False,
            )
            # static
            _, _, _, _, _, _, rgb_point_static, sigma_static, _, _ = tensorf_static(
                rays_chunk,
                ts_chunk,
                None,
                xyz_sampled,
                z_vals,
                ray_valid,
                is_train=False,
                white_bg=white_bg,
                ray_type=ray_type,
                N_samples=N_samples,
            )
            # dynamic
            (
                _,
                _,
                blending,
                pts_ref,
                _,
                _,
                rgb_point_dynamic,
                sigma_dynamic,
                z_val_dynamic,
                dist_dynamic,
            ) = tensorf(
                rays_chunk,
                ts_chunk,
                None,
                xyz_sampled,
                z_vals,
                ray_valid,
                is_train=False,
                white_bg=white_bg,
                ray_type=ray_type,
                N_samples=N_samples,
            )
            # blending
            (
                rgb_map,
                depth_map_full,
                acc_map_full,
                weights_full,
                rgb_map_s,
                depth_map_s,
                acc_map_s,
                weights_s,
                rgb_map_d,
                depth_map_d,
                acc_map_d,
                weights_d,
                dynamicness_map,
            ) = raw2outputs(
                rgb_point_static,
                sigma_static,
                rgb_point_dynamic.to(device),
                sigma_dynamic.to(device),
                dist_dynamic.to(device),
                blending,
                z_val_dynamic.to(device),
                rays_chunk,
                ray_type=ray_type,
            )
            # gather chunks
            rgb_map_list.append(rgb_map)
            depth_map_list.append(depth_map_full)
            rgb_map_s_list.append(rgb_map_s)
            depth_map_s_list.append(depth_map_s)
            rgb_map_d_list.append(rgb_map_d)
            depth_map_d_list.append(depth_map_d)
            dynamicness_map_list.append(dynamicness_map)
            weights_s_list.append(weights_s)
        rgb_map = torch.cat(rgb_map_list)
        depth_map = torch.cat(depth_map_list)
        rgb_map_s = torch.cat(rgb_map_s_list)
        depth_map_s = torch.cat(depth_map_s_list)
        rgb_map_d = torch.cat(rgb_map_d_list)
        depth_map_d = torch.cat(depth_map_d_list)
        dynamicness_map = torch.cat(dynamicness_map_list)
        weights_s__ = torch.cat(weights_s_list)
        weights_s__ = weights_s__.reshape(H, W, -1)

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map_s = rgb_map_s.clamp(0.0, 1.0)
        rgb_map_d = rgb_map_d.clamp(0.0, 1.0)
        blending_map = dynamicness_map.clamp(0.0, 1.0)

        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )
        rgb_map_s, depth_map_s = (
            rgb_map_s.reshape(H, W, 3).cpu(),
            depth_map_s.reshape(H, W).cpu(),
        )
        rgb_map_d, depth_map_d = (
            rgb_map_d.reshape(H, W, 3).cpu(),
            depth_map_d.reshape(H, W).cpu(),
        )
        blending_map = blending_map.reshape(H, W).cpu()
        if ray_type == "contract":
            depth_map_s = -1.0 / (depth_map_s + 1e-6)
            depth_map_d = -1.0 / (depth_map_d + 1e-6)
            depth_map = -1.0 / (depth_map + 1e-6)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")
        rgb_map_s = (rgb_map_s.numpy() * 255).astype("uint8")
        rgb_map_d = (rgb_map_d.numpy() * 255).astype("uint8")
        blending_map = (blending_map.numpy() * 255).astype("uint8")
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        rgb_maps_s.append(rgb_map_s)
        depth_maps_s.append(depth_map_s)
        rgb_maps_d.append(rgb_map_d)
        depth_maps_d.append(depth_map_d)
        blending_maps.append(blending_map)
        if savePath is not None:
            if evaluation:
                imageio.imwrite(
                    f"{savePath}/{prtx}v000_t{idx:03d}.png", rgb_map, format="png"
                )
            else:
                imageio.imwrite(
                    f"{savePath}/{prtx}{idx:03d}.png", rgb_map, format="png"
                )
            imageio.imwrite(
                f"{savePath}_static/{prtx}{idx:03d}.png", rgb_map_s, format="png"
            )
            imageio.imwrite(
                f"{savePath}_dynamic/{prtx}{idx:03d}.png", rgb_map_d, format="png"
            )
            imageio.imwrite(
                f"{savePath}_dynamic/{prtx}{idx:03d}_blending.png",
                blending_map,
                format="png",
            )

    if evaluation:
        return

    depth_list = []
    depth_list.extend(depth_maps)
    # tmp_list.extend(depth_maps_s)
    # tmp_list.extend(depth_maps_d)
    # all_depth = torch.stack(tmp_list)

    # for idx in range(len(rgb_maps)):
    #     depth_maps[idx] = visualize_depth_numpy(torch.clamp(depth_maps[idx], min=depth_map_min, max=depth_map_max).numpy(), (depth_map_min, depth_map_max))[0]
    #     depth_maps_s[idx] = visualize_depth_numpy(torch.clamp(depth_maps_s[idx], min=depth_map_min, max=depth_map_max).numpy(), (depth_map_min, depth_map_max))[0]
    #     depth_maps_d[idx] = visualize_depth_numpy(torch.clamp(depth_maps_d[idx], min=depth_map_min, max=depth_map_max).numpy(), (depth_map_min, depth_map_max))[0]
    #     if savePath is not None:
    #         imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', depth_maps[idx], format='png')
    #         imageio.imwrite(f'{savePath}_static/rgbd/{prtx}{idx:03d}.png', depth_maps_s[idx], format='png')
    #         imageio.imwrite(f'{savePath}_dynamic/rgbd/{prtx}{idx:03d}.png', depth_maps_d[idx], format='png')

    # if not evaluation:
    imageio.mimwrite(
        f"{savePath}/{prtx}video.mp4",
        np.stack(rgb_maps),
        fps=30,
        quality=8,
        format="ffmpeg",
        output_params=["-f", "mp4"],
    )
    # imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])
    imageio.mimwrite(
        f"{savePath}_static/{prtx}video.mp4",
        np.stack(rgb_maps_s),
        fps=30,
        quality=8,
        format="ffmpeg",
        output_params=["-f", "mp4"],
    )
    # imageio.mimwrite(f'{savePath}_static/{prtx}depthvideo.mp4', np.stack(depth_maps_s), fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])
    imageio.mimwrite(
        f"{savePath}_dynamic/{prtx}video.mp4",
        np.stack(rgb_maps_d),
        fps=30,
        quality=8,
        format="ffmpeg",
        output_params=["-f", "mp4"],
    )
    # imageio.mimwrite(f'{savePath}_dynamic/{prtx}depthvideo.mp4', np.stack(depth_maps_d), fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])
    # imageio.mimwrite(f'{savePath}_dynamic/{prtx}blending.mp4', np.stack(blending_maps), fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f"{savePath}/{prtx}mean.txt", np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f"{savePath}/{prtx}mean.txt", np.asarray([psnr]))

    return PSNRs, depth_list


def NDC2world(pts, H, W, f):
    # NDC coordinate to world coordinate
    pts_z = 2 / (torch.clamp(pts[..., 2:], min=-1.0, max=1 - 1e-6) - 1)
    pts_x = -pts[..., 0:1] * pts_z * W / 2 / f
    pts_y = -pts[..., 1:2] * pts_z * H / 2 / f
    pts_world = torch.cat([pts_x, pts_y, pts_z], -1)

    return pts_world


def world2NDC(pts_world, H, W, f):
    o0 = -1.0 / (W / (2.0 * f)) * pts_world[..., 0:1] / pts_world[..., 2:]
    o1 = -1.0 / (H / (2.0 * f)) * pts_world[..., 1:2] / pts_world[..., 2:]
    o2 = 1.0 + 2.0 * 1 / pts_world[..., 2:]
    pts = torch.cat([o0, o1, o2], -1)

    return pts


def contract2world(pts_contract):
    pts_norm, _ = torch.max(torch.abs(pts_contract.clone()), dim=-1)
    contract_mask = pts_norm > 1.0
    scale = -1 / (pts_norm - 2)
    pts_world = pts_contract
    pts_world[~contract_mask] = pts_contract[~contract_mask]
    pts_world[contract_mask] = (
        pts_contract[contract_mask]
        / (pts_norm[contract_mask][:, None])
        * scale[contract_mask][:, None]
    )  # TODO: NaN?
    return pts_world


def render_single_3d_point(H, W, f, c2w, pt_NDC):
    """Render 3D position along each ray and project it to the image plane."""
    w2c = c2w[:, :3, :3].transpose(1, 2)

    # NDC coordinate to world coordinate
    pts_map_world = NDC2world(pt_NDC, H, W, f)

    # World coordinate to camera coordinate
    # Translate
    pts_map_world = pts_map_world - c2w[..., 3]
    # Rotate
    pts_map_cam = torch.sum(torch.mul(pts_map_world[..., None, :], w2c[:, :3, :3]), -1)
    # pts_map_cam = torch.sum(pts_map_world[..., None, :] * w2c[:3, :3], -1)

    # Camera coordinate to 2D image coordinate
    pts_plane = torch.cat(
        [
            pts_map_cam[..., 0:1] / (-pts_map_cam[..., 2:]) * f + W * 0.5,
            -pts_map_cam[..., 1:2] / (-pts_map_cam[..., 2:]) * f + H * 0.5,
        ],
        -1,
    )
    # pts_disparity = 1.0 / pts_map_cam[..., 2:]

    pts_map_cam_NDC = world2NDC(pts_map_cam, H, W, f)

    return pts_plane, ((pts_map_cam_NDC[:, 2:] + 1.0) / 2.0)


def render_3d_point(H, W, f, c2w, weights, pts, rays, ray_type="ndc"):
    """Render 3D position along each ray and project it to the image plane."""
    w2c = c2w[:, :3, :3].transpose(1, 2)

    # Rendered 3D position in NDC coordinate
    acc_map = torch.sum(weights, -1)[:, None]
    pts_map_NDC = torch.sum(weights[..., None] * pts, -2)
    if ray_type == "ndc":
        pts_map_NDC = pts_map_NDC + (1.0 - acc_map) * (rays[:, :3] + rays[:, 3:])
    elif ray_type == "contract":
        farest_pts = rays[:, :3] + rays[:, 3:] * 256.0
        # convert to contract domain
        farest_pts_norm, _ = torch.max(torch.abs(farest_pts.clone()), dim=-1)
        contract_mask = farest_pts_norm > 1.0
        farest_pts[contract_mask] = (2 - 1 / farest_pts_norm[contract_mask])[
            ..., None
        ] * (farest_pts[contract_mask] / farest_pts_norm[contract_mask][..., None])
        pts_map_NDC = pts_map_NDC + (1.0 - acc_map) * farest_pts

    # NDC coordinate to world coordinate
    if ray_type == "ndc":
        pts_map_world = NDC2world(pts_map_NDC, H, W, f)
    elif ray_type == "contract":
        pts_map_world = contract2world(pts_map_NDC)

    # World coordinate to camera coordinate
    # Translate
    pts_map_world = pts_map_world - c2w[..., 3]
    # Rotate
    pts_map_cam = torch.sum(torch.mul(pts_map_world[..., None, :], w2c[:, :3, :3]), -1)

    # Camera coordinate to 2D image coordinate
    pts_plane = torch.cat(
        [
            pts_map_cam[..., 0:1] / (-pts_map_cam[..., 2:]) * f + W * 0.5,
            -pts_map_cam[..., 1:2] / (-pts_map_cam[..., 2:]) * f + H * 0.5,
        ],
        -1,
    )

    pts_map_cam_NDC = world2NDC(pts_map_cam, H, W, f)

    return pts_plane, pts_map_cam_NDC[:, 2:]


def induce_flow_single(H, W, focal, pose_neighbor, pts_3d_neighbor, pts_2d):
    # Render 3D position along each ray and project it to the neighbor frame's image plane.
    pts_2d_neighbor, _ = render_single_3d_point(
        H, W, focal, pose_neighbor, pts_3d_neighbor
    )
    induced_flow = pts_2d_neighbor - pts_2d

    return induced_flow


def induce_flow(
    H, W, focal, pose_neighbor, weights, pts_3d_neighbor, pts_2d, rays, ray_type="ndc"
):
    # Render 3D position along each ray and project it to the neighbor frame's image plane.
    pts_2d_neighbor, induced_disp = render_3d_point(
        H, W, focal, pose_neighbor, weights, pts_3d_neighbor, rays, ray_type
    )
    induced_flow = pts_2d_neighbor - pts_2d

    return induced_flow, induced_disp
