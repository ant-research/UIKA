def get_sing_batch_smpl_data(smpl_data, bidx):
    smpl_data_single_batch = {}
    for k, v in smpl_data.items():
        smpl_data_single_batch[k] = v[bidx]  # e.g. body_pose: [B, N_v, 21, 3] -> [N_v, 21, 3]
        if k == "betas" or (k == "joint_offset") or (k == "face_offset"):
            smpl_data_single_batch[k] = v[bidx:bidx+1]  # e.g. betas: [B, 100] -> [1, 100]
    return smpl_data_single_batch


def get_single_view_smpl_data(smpl_data, vidx):
    smpl_data_single_view = {}        
    for k, v in smpl_data.items():
        assert v.shape[0] == 1
        if k == "betas" or (k == "joint_offset") or (k == "face_offset") or (k == "transform_mat_neutral_pose"):
            smpl_data_single_view[k] = v  # e.g. betas: [1, 100] -> [1, 100]
        else:
            smpl_data_single_view[k] = v[:, vidx: vidx + 1]  # e.g. body_pose: [1, N_v, 21, 3] -> [1, 1, 21, 3]
    return smpl_data_single_view
