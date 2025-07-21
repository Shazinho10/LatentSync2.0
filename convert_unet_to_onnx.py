import torch
from omegaconf import OmegaConf
from latentsync.models.unet import UNet3DConditionModel

# --- NECESSARY WRAPPER: Handles the 5D to 4D reshape for ONNX ---
class UNetExportWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, latent, timestep, encoder_hidden_states):
        batch_size, channels, num_frames, height, width = latent.shape
        latent_reshaped = latent.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        return self.unet(latent_reshaped, timestep, encoder_hidden_states)

def export_unet(unet_config_path, inference_ckpt_path, onnx_out_path):
    device = "cpu"
    print(f"--- Running ONNX export on {device.upper()} to conserve memory ---")

    print(f"Loading UNet from checkpoint: {inference_ckpt_path}")

    config = OmegaConf.load(unet_config_path)

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        inference_ckpt_path,
        device=device,
    )
    unet.eval()

    print("UNet model loaded.")

    batch_size = 1
    num_frames = 16

    dummy_latent = torch.randn(batch_size, 13, num_frames, 64, 64, device=device)
    dummy_timestep = torch.tensor(951, device=device)
    dummy_audio_embeds = torch.randn(batch_size * num_frames, 50, 384, device=device)

    print("\nExporting with verified 5D shapes:")
    print(f"  - Latent: {dummy_latent.shape}")
    print(f"  - Timestep: {dummy_timestep.shape}")
    print(f"  - Audio Embeds: {dummy_audio_embeds.shape}\n")

    torch.onnx.export(
        unet,
        (dummy_latent, dummy_timestep, dummy_audio_embeds),
        onnx_out_path,
        opset_version=17,
        input_names=['latent', 'timestep', 'encoder_hidden_states'],
        output_names=['output'],
        dynamic_axes={
            'latent': {0: 'batch_size', 2: 'num_frames'},
            'encoder_hidden_states': {0: 'audio_batch_size'},
        },
        verbose=False
    )

    print(f"Successfully exported UNet to {onnx_out_path}")

if __name__ == "__main__":
    export_unet(
        unet_config_path="configs/unet/stage1_512.yaml",
        inference_ckpt_path="checkpoints/latentsync_unet.pt",
        onnx_out_path="latentsync_unet.onnx"
    )