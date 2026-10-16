# Adapted from SC2-PCR (Chen et al., CVPR 2022), https://github.com/ZhiChen902/SC2-PCR
# Original code is licensed under the MIT License (see the original repository).
import numpy as np
import torch


def integrate_trans(R, t):
    if len(R.shape) == 3:
        trans = torch.eye(4, dtype=R.dtype)[None].repeat(R.shape[0], 1, 1).to(R.device)
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t
    else:
        trans = torch.eye(4).to(R.device)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans


def se3_transform(pts, trans):
    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0, 2, 1) + trans[:, :3, 3:4]
        return trans_pts.permute(0, 2, 1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T


def rigid_transform_3d(A, B, weights=None, weight_threshold=0):
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights = weights.clone()
    weights[weights < weight_threshold] = 0

    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (
        torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (
        torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

    Am = A - centroid_A
    Bm = B - centroid_B

    Weight = torch.diag_embed(weights)
    H = Am.permute(0, 2, 1) @ Weight @ Bm

    U, S, Vt = torch.linalg.svd(H.cpu())
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
    delta_UV = torch.det(Vt.permute(0, 2, 1) @ U.permute(0, 2, 1))
    eye = torch.eye(3, dtype=A.dtype)[None, :, :].repeat(bs, 1, 1).to(A.device)
    eye[:, -1, -1] = delta_UV
    R = Vt.permute(0, 2, 1) @ eye @ U.permute(0, 2, 1)
    t = centroid_B.permute(0, 2, 1) - R @ centroid_A.permute(0, 2, 1)
    return integrate_trans(R, t)


class SC2PCRMatcherOriginal:
    def __init__(self, inlier_threshold=1.0, d_thre=1.0, num_iterations=10,
                ratio=0.2, nms_radius=0.1, max_points=8000, k1=30, k2=20):
        self.inlier_threshold = inlier_threshold
        self.d_thre = d_thre
        self.num_iterations = num_iterations
        self.ratio = ratio
        self.nms_radius = nms_radius
        self.max_points = max_points
        self.k1 = k1
        self.k2 = k2

    def cal_leading_eigenvector(self, M, method='power'):
        if method == 'power':
            leading_eig = torch.ones_like(M[:, :, 0:1])
            leading_eig_last = leading_eig
            for i in range(self.num_iterations):
                leading_eig = torch.bmm(M, leading_eig)
                leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
                if torch.allclose(leading_eig, leading_eig_last):
                    break
                leading_eig_last = leading_eig
            leading_eig = leading_eig.squeeze(-1)
            return leading_eig
        else:
            raise NotImplementedError

    def pick_seeds(self, dists, scores, R, max_num):
        assert scores.shape[0] == 1
        score_relation = scores.T >= scores
        score_relation = score_relation.bool() | (dists[0] >= R).bool()
        is_local_max = score_relation.min(-1)[0].float()
        score_local_max = scores * is_local_max
        sorted_score = torch.argsort(score_local_max, dim=1, descending=True)
        return_idx = sorted_score[:, 0: max_num].detach()
        return return_idx

    def cal_seed_trans(self, seeds, SC2_measure, src_keypts, tgt_keypts):
        bs, num_corr, num_channels = SC2_measure.shape[0], SC2_measure.shape[1], SC2_measure.shape[2]
        k1 = self.k1
        k2 = self.k2
        if k1 > num_channels:
            k1 = 4
            k2 = 4

        sorted_score = torch.argsort(SC2_measure, dim=2, descending=True)
        knn_idx = sorted_score[:, :, 0: k1]
        idx_tmp = knn_idx.contiguous().view([bs, -1])
        idx_tmp = idx_tmp[:, :, None]
        idx_tmp = idx_tmp.expand(-1, -1, 3)

        src_knn = src_keypts.gather(dim=1, index=idx_tmp).view([bs, -1, k1, 3])
        tgt_knn = tgt_keypts.gather(dim=1, index=idx_tmp).view([bs, -1, k1, 3])
        src_dist = ((src_knn[:, :, :, None, :] - src_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        tgt_dist = ((tgt_knn[:, :, :, None, :] - tgt_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        cross_dist = torch.abs(src_dist - tgt_dist)
        local_hard_SC_measure = (cross_dist < self.d_thre).float()
        local_SC2_measure = torch.matmul(local_hard_SC_measure[:, :, :1, :], local_hard_SC_measure)

        sorted_score = torch.argsort(local_SC2_measure, dim=3, descending=True)
        knn_idx_fine = sorted_score[:, :, :, 0: k2]

        num = knn_idx_fine.shape[1]
        knn_idx_fine_flat = knn_idx_fine.contiguous().view([bs, num, -1])
        final_corr_idx = knn_idx.gather(dim=2, index=knn_idx_fine_flat)

        knn_idx_fine = knn_idx_fine_flat[:, :, :, None]
        knn_idx_fine = knn_idx_fine.expand(-1, -1, -1, 3)
        src_knn_fine = src_knn.gather(dim=2, index=knn_idx_fine).view([bs, -1, k2, 3])
        tgt_knn_fine = tgt_knn.gather(dim=2, index=knn_idx_fine).view([bs, -1, k2, 3])

        src_dist = ((src_knn_fine[:, :, :, None, :] - src_knn_fine[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        tgt_dist = ((tgt_knn_fine[:, :, :, None, :] - tgt_knn_fine[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        cross_dist = torch.abs(src_dist - tgt_dist)
        local_SC_measure = torch.clamp(1 - cross_dist ** 2 / self.d_thre ** 2, min=0)
        local_SC2_measure = local_SC_measure
        local_SC2_measure = local_SC2_measure.view([-1, k2, k2])

        local_SC2_measure[:, torch.arange(local_SC2_measure.shape[1]), torch.arange(local_SC2_measure.shape[1])] = 0
        total_weight = self.cal_leading_eigenvector(local_SC2_measure, method='power')
        total_weight = total_weight.view([bs, -1, k2])
        total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + 1e-6)

        total_weight = total_weight.view([-1, k2])
        src_knn2, tgt_knn2 = src_knn_fine.view([-1, k2, 3]), tgt_knn_fine.view([-1, k2, 3])

        seedwise_trans = rigid_transform_3d(src_knn2, tgt_knn2, total_weight)
        seedwise_trans = seedwise_trans.view([bs, -1, 4, 4])

        pred_position = torch.einsum('bsnm,bmk->bsnk', seedwise_trans[:, :, :3, :3],
                                     src_keypts.permute(0, 2, 1)) + seedwise_trans[:, :, :3, 3:4]
        pred_position = pred_position.permute(0, 1, 3, 2)
        L2_dis = torch.norm(pred_position - tgt_keypts[:, None, :, :], dim=-1)
        seedwise_fitness = torch.sum((L2_dis < self.inlier_threshold).float(), dim=-1)
        batch_best_guess = seedwise_fitness.argmax(dim=1)
        final_trans = seedwise_trans.gather(
            dim=1, index=batch_best_guess[:, None, None, None].expand(-1, -1, 4, 4)).squeeze(1)

        return final_trans, seedwise_trans, seedwise_fitness, total_weight.view([bs, -1, k2]), final_corr_idx

    def post_refinement(self, initial_trans, src_keypts, tgt_keypts, it_num, weights=None):
        assert initial_trans.shape[0] == 1
        inlier_threshold_list = [self.inlier_threshold] * it_num
        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:
            warped_src_keypts = se3_transform(src_keypts, initial_trans)
            L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
            pred_inlier = (L2_dis < inlier_threshold)[0]
            inlier_num = torch.sum(pred_inlier)
            if abs(int(inlier_num - previous_inlier_num)) < 1:
                break
            else:
                previous_inlier_num = inlier_num
            initial_trans = rigid_transform_3d(
                A=src_keypts[:, pred_inlier, :],
                B=tgt_keypts[:, pred_inlier, :],
                weights=1 / (1 + (L2_dis / inlier_threshold) ** 2)[:, pred_inlier],
            )
        return initial_trans

    def SC2_PCR(self, src_keypts, tgt_keypts):
        bs, num_corr = src_keypts.shape[0], tgt_keypts.shape[1]
        diag = {}

        if num_corr > self.max_points:
            src_keypts = src_keypts[:, :self.max_points, :]
            tgt_keypts = tgt_keypts[:, :self.max_points, :]
            num_corr = self.max_points
        diag['num_corr_used'] = num_corr

        src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
        target_dist = torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
        cross_dist = torch.abs(src_dist - target_dist)

        SC_dist_thre = self.d_thre
        SC_measure = torch.clamp(1.0 - cross_dist ** 2 / SC_dist_thre ** 2, min=0)
        hard_SC_measure = (cross_dist < SC_dist_thre).float()

        confidence = self.cal_leading_eigenvector(SC_measure, method='power')
        diag['confidence'] = confidence
        seeds = self.pick_seeds(src_dist, confidence, R=self.nms_radius, max_num=int(num_corr * self.ratio))
        diag['seeds'] = seeds

        SC2_dist_thre = self.d_thre / 2
        hard_SC_measure_tight = (cross_dist < SC2_dist_thre).float()
        seed_hard_SC_measure = hard_SC_measure.gather(dim=1, index=seeds[:, :, None].expand(-1, -1, num_corr))
        seed_hard_SC_measure_tight = hard_SC_measure_tight.gather(dim=1, index=seeds[:, :, None].expand(-1, -1, num_corr))
        SC2_measure = torch.matmul(seed_hard_SC_measure_tight, hard_SC_measure_tight) * seed_hard_SC_measure
        diag['SC2_measure'] = SC2_measure

        final_trans, seedwise_trans, seedwise_fitness, seed_weights, seed_knn_idx = self.cal_seed_trans(
            seeds, SC2_measure, src_keypts, tgt_keypts)
        diag['seedwise_trans'] = seedwise_trans
        diag['seedwise_fitness'] = seedwise_fitness
        diag['seed_weights'] = seed_weights
        diag['seed_knn_idx'] = seed_knn_idx
        diag['pre_refinement_trans'] = final_trans

        final_trans = self.post_refinement(final_trans, src_keypts, tgt_keypts, 20)
        diag['final_trans'] = final_trans
        diag['num_corr'] = num_corr

        return final_trans, diag
