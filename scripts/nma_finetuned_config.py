config = {
    # This section defines where your fine-tuned model and logs will be saved.
    "file_path": {
        "save_dir": "models/esmdance-mutant-nma-fine-tuned/", 
    },
    
    # General training settings.
    "training": {
        "random_seed": 42,
        "dropout": 0.1,
        
        # You should adjust these based on how often you want to save and log.
        # For a shorter fine-tuning run, you'll want to save more frequently.
        "save_per_epoch": 1, # It's easier to think in epochs for fine-tuning.
        
        # --- CRITICAL CHANGE 1: Feature Indices ---
        # This now correctly maps to your 3 residue-level and 3 pairwise NMA features.
        "res_feature_idx": {
            'nma_residue1': [0],
            'nma_residue2': [1],
            'nma_residue3': [2],
        },
        "pair_feature_idx": {
            'nma_pair1': [0],
            'nma_pair2': [1],
            'nma_pair3': [2],
        },
    },

    # --- CRITICAL CHANGE 2: Simplified Training Schedule ---
    # This block is now tailored for your specific fine-tuning task.
    "esmdance": {
        "freeze_esm": True,      # Correct for ESMDance fine-tuning.
        "randomize_esm": False,
        
        # All your sequences are 158, so we set one max_len.
        # We add a little buffer, but it could be exactly 158.
        "max_len": 256,
        
        # Define training by epochs, which is more intuitive for a fixed dataset.
        "num_epochs": 20, # Adjust this based on how your loss behaves.
        
        # Set a single batch size. Adjust based on your GPU memory.
        "batch_size": 4,
        
        # Gradient accumulation helps simulate a larger batch size.
        # Effective batch size = batch_size * gradient_accumulation_steps
        # Example: 8 * 4 = 32
        "gradient_accumulation_steps": 4, 
    },

    # Optimizer settings. These are generally good starting points.
    "optimizer": {
        "peak_lr": 1e-4,
        "epsilon": 1e-8,
        "betas": (0.9, 0.98),
        "weight_decay": 0.01,
        "warmup_steps": 200, # Number of steps for learning rate warmup.
    },

    # --- CRITICAL CHANGE 3: Model Output Dimensions ---
    "model_35M": {
        "model_id": "facebook/esm2_t12_35M_UR50D",
        "atten_dim": 240,
        "embed_dim": 480,
        
        # These now match your NMA-only data.
        "pair_out_dim": 3, # Was 13. Now 3 for your 3 ANM correlation matrices.
        "res_out_dim": 3,  # Was 50. Now 3 for your 3 GNM fluctuation vectors.
    },
}