import torch
from torchinfo import summary

def get_model_complexity(model, input_size, device='cpu'):
    """
    Calculate GFlops and Params using torchinfo.
    """
    try:
        # Create a dummy input
        # Input size expected: (B, C, 2T+1, H, W) or (B, C, H, W) depending on model
        # SymFormer takes (B, 1, 3, 512, 512) for example if T=1
        
        # Parse input_size formatted as (H, W) or (C, H, W)
        # We need to know the specific input shape expected by the model's forward
        
        # Run a small forward to inspect device
        # model = model.to(device)
        
        # Use torchinfo
        # Note: batch_dim=0, col_names=["input_size", "output_size", "num_params", "mult_adds"]
        
        # Construct input_data depending on config
        # Assuming SymFormer with T=1, input is (1, 3, H, W) essentially inside the model logic
        # But proper shape is (B, slices, H, W)
        
        # Let's rely on passing the correct input_size tuple including batch and channels
        # For SymFormer: (1, 3, 512, 512) for T=1
        
        stats = summary(model, input_size=input_size, verbose=0, device=device)
        
        params = stats.total_params
        # mult_adds is roughly MACs. GFlops ~= 2 * MACs
        macs = stats.total_mult_adds
        gflops = 2 * macs / 1e9
        
        return {
            'Params (M)': params / 1e6,
            'GFlops': gflops,
            'MACs (G)': macs / 1e9
        }
        
    except Exception as e:
        print(f"Complexity calculation failed: {e}")
        return {
            'Params (M)': 0,
            'GFlops': 0,
            'MACs (G)': 0
        }
