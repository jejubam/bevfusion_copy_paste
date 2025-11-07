import torch

src = "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth"
dst = "bevfusion_lidar_spconv2_converted.pth"
ckpt = torch.load(src, map_location="cpu")

state = ckpt.get("state_dict", ckpt)

def needs_convert(name, w):
    if not isinstance(w, torch.Tensor): return False
    if w.ndim != 5: return False
    # 대표적으로 spconv3d weight 후보 모듈들
    hit = any(k in name for k in [
        "pts_middle_encoder", "sparse_conv", "spconv", "encoder_layers", "conv_input", "conv_out"
    ])
    return hit

converted = {}
for k, v in state.items():
    if needs_convert(k, v):
        # v.shape 예) (out, kD, kH, kW, in) 혹은 (out, in, kD, kH, kW)
        if v.shape[0] <= 128 and v.shape[-1] <= 256:
            # 흔한 케이스: [out, kD, kH, kW, in] -> [kD, kH, kW, in, out]
            v2 = v.permute(1,2,3,4,0).contiguous()
        else:
            # 예비: [out, in, kD, kH, kW] -> [kD, kH, kW, in, out]
            v2 = v.permute(2,3,4,1,0).contiguous()
        converted[k] = v2
        #print("converted", k, v.shape, "->", v2.shape)
    else:
        converted[k] = v

ckpt["state_dict"] = converted
torch.save(ckpt, dst)
print("saved:", dst)
