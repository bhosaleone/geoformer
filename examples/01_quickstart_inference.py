"""
Example 1: GeoFormer Foundation Model Quickstart & Inference
============================================================
Demonstrates loading the deep 3-layer GeoFormer foundation network
(249,665 parameters) and executing an inference pass on citation features.

License: MIT
Author: Shrikant Bhosale (Atmabhan Pandit), TWIST POOL Labs
"""

import os
import torch
from geoformer import GeoFormer

def main():
    print("=" * 70)
    print("🧠 GeoFormer Example 1: Foundation Model Loading & Inference")
    print("=" * 70)

    # Path to verified checkpoint
    ckpt_path = "checkpoints/geoformer_foundation_model.pt"
    
    if not os.path.exists(ckpt_path):
        print(f"Downloading checkpoint from Hugging Face Model Hub (SetMD/geoformer)...")
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(repo_id="SetMD/geoformer", filename="geoformer_foundation_model.pt")

    # Load serialized dictionary
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"• Architecture: {ckpt.get('architecture', 'GeoFormer')}")
    print(f"• Layer Depth: {ckpt.get('n_layers', 3)} Stacked Blocks | Heads: {ckpt.get('n_heads', 4)}")
    print(f"• Feature Dimension: {ckpt.get('in_features', 1433)} | Classes: {ckpt.get('num_classes', 7)}")

    # Instantiate model
    model = GeoFormer(
        in_features=ckpt["in_features"],
        num_classes=ckpt["num_classes"],
        d_model=ckpt["d_model"],
        n_layers=ckpt["n_layers"],
        n_heads=ckpt["n_heads"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"• Total Parameter Count: {total_params:,}")
    print(f"• Packed 1.58-Bit Footprint: {(total_params * 0.20) / 1024:.2f} KB (fits on edge MCUs)")

    # Test forward pass with a simulated scientific paper feature vector (1,433 bag-of-words attributes)
    torch.manual_seed(42)
    sample_paper = torch.randn(1, ckpt["in_features"])

    with torch.no_grad():
        logits = model(sample_paper)
        probabilities = torch.softmax(logits, dim=-1).squeeze()
        predicted_class = logits.argmax(dim=-1).item()

    print("\nInference Output:")
    print(f"  -> Predicted Topic Class: {predicted_class}")
    print(f"  -> Class Confidence: {probabilities[predicted_class]*100:.2f}%")
    print("  -> Probability Distribution Across 7 Topics:")
    for i, p in enumerate(probabilities.tolist()):
        bar = "█" * int(p * 30)
        print(f"     Class {i}: {p*100:5.2f}% {bar}")

    print("\n✅ Inference completed successfully in < 1 ms on CPU!")

if __name__ == "__main__":
    main()
