import os
import torch
import copy
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
import alpamayo1_5.nav_utils as nav_utils
import alpamayo1_5.helper as helper
from alpamayo1_5.load_custom_dataset import load_custom_dataset

device = "cuda"
base_dir = "../datasets/route_1"
seg_dir = os.path.join(base_dir, "segment_06")

print("Loading model...")
model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16, attn_implementation="eager").to(device)
processor = helper.get_processor(model.tokenizer)

def run_cmd(nav_cmd, padding="black"):
    torch.cuda.manual_seed_all(42)
    data = load_custom_dataset(seg_dir, 0)
    
    if padding == "duplicate":
        front_wide = data["image_frames"][:1]
        data["image_frames"] = torch.cat([front_wide]*4, dim=0)
        data["camera_indices"] = torch.tensor([0, 1, 2, 6], dtype=torch.int64)

    messages = helper.create_message(
        data["image_frames"].flatten(0, 1),
        camera_indices=data.get("camera_indices"),
        nav_text=nav_cmd,
    )
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, continue_final_message=True, return_dict=True, return_tensors="pt"
    )
    model_inputs = helper.to_device(
        {"tokenized_data": inputs, "ego_history_xyz": data["ego_history_xyz"], "ego_history_rot": data["ego_history_rot"]}, device
    )

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        pred_xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
            diffusion_kwargs={"use_classifier_free_guidance": True, "inference_guidance_weight": 1.5, "temperature": 0.6}
        )
    cot = extra["cot"][0][0]
    if isinstance(cot, list) and len(cot) > 0: cot = str(cot[0])
    print(f"\nCondition: '{nav_cmd}', pad={padding}")
    print(f"CoT: {cot}")
    pred = pred_xyz.cpu().numpy()[0, 0, 0]
    print(f"Pred Y disp at t=10: {pred[10, 1]:.3f}, at t=63: {pred[63, 1]:.3f}")

run_cmd("Turn left")
run_cmd("Turn left in 30m")
run_cmd("Turn right", padding="duplicate")
run_cmd("Turn left", padding="duplicate")
