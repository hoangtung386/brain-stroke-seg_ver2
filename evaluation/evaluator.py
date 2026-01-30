import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from .metrics import MetricCalculator
from .complexity import get_model_complexity
from .visualization import visualize_overlay_predictions, plot_metrics_comparison, plot_confusion_matrix
from sklearn.metrics import confusion_matrix

class Evaluator:
    def __init__(self, model, val_loader, device, config, num_samples=-1):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.num_samples = num_samples  # -1 means all samples
        self.output_dir = 'evaluation_results'
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.metric_calc = MetricCalculator(config.NUM_CLASSES, device)
        
    def run(self):
        print("Starting comprehensive evaluation...")
        
        # 1. Model Complexity
        print("Calculating model complexity...")
        # SymFormer Input: (B, 2T+1, H, W). Summary expects (B, *input_size)
        input_shape = (1, 2*self.config.T + 1, *self.config.IMAGE_SIZE)
        complexity = get_model_complexity(self.model, input_shape, self.device)
        print(f"Complexity: {complexity}")
        
        # 2. Inference & Metrics
        self.model.eval()
        all_metrics = []
        all_preds = []
        all_labels = []
        
        print(f"Evaluating on {len(self.val_loader)} batches...")
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                # Handle both 2-tuple (images, masks) and 3-tuple (images, masks, metadata)
                if len(batch) == 3:
                    images, masks, metadata = batch
                else:
                    images, masks = batch
                    metadata = None
                
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Move metadata to device if present
                if metadata is not None:
                    metadata = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                               for k, v in metadata.items()}
                
                # Forward
                outputs, aligned_slices, alignment_params, _, _ = self.model(
                    images, metadata=metadata, return_alignment=True
                )
                
                # --- ALIGN MASK TO MATCH IMAGE ---
                # 1. Get center slice params
                num_slices = images.shape[1]
                center_idx = num_slices // 2
                center_params = alignment_params[center_idx]
                
                # 2. Prepare mask
                if masks.ndim == 3:
                    masks_for_align = masks.unsqueeze(1).float()
                else:
                    masks_for_align = masks.float()
                
                # 3. Apply Transform
                # Use nearest for masks to preserve classes
                if hasattr(self.model, 'module'):
                    align_net = self.model.module.alignment_net
                else:
                    align_net = self.model.alignment_net
                    
                aligned_masks, _ = align_net.apply_transform(
                    masks_for_align, 
                    center_params, 
                    mode='nearest'
                )
                
                # 4. Convert back
                if masks.ndim == 3:
                    masks = aligned_masks.long().squeeze(1)
                else:
                    masks = aligned_masks.long()

                preds = torch.argmax(outputs, dim=1)
                
                # Compute batch metrics
                batch_metrics = self.metric_calc.compute_batch(outputs, masks)
                all_metrics.extend(batch_metrics)
                
                # Store for confusion matrix
                all_preds.append(preds.cpu().numpy().flatten())
                all_labels.append(masks.cpu().numpy().flatten())
                
        # 3. Aggregate Metrics with CI
        summary_stats = self.metric_calc.aggregate_and_ci(all_metrics)
        
        # 4. Save Report
        self.save_report(summary_stats, complexity)
        
        # 5. Visualization
        print("Generating visualizations...")
        self.visualize_qualitative()
        
        # 6. Confusion Matrix
        print("Generating confusion matrix...")
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        cm = confusion_matrix(all_labels, all_preds, labels=range(self.config.NUM_CLASSES))
        plot_confusion_matrix(cm, ['Background', 'Region'], self.output_dir)
        
        return summary_stats
        
    def save_report(self, stats, complexity):
        report_path = os.path.join(self.output_dir, 'final_evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("       SYMFORMER EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("1. MODEL COMPLEXITY\n")
            f.write("-" * 30 + "\n")
            for k, v in complexity.items():
                f.write(f"{k:<15}: {v:.4f}\n")
            f.write("\n")
            
            f.write("2. PERFORMANCE METRICS (with 95% CI)\n")
            f.write("-" * 60 + "\n")
            
            # Group by class
            f.write(f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'CI-95%':<10}\n")
            f.write("-" * 60 + "\n")
            
            for k, v in stats.items():
                if isinstance(v, dict) and 'mean' in v:
                    mean_val = v['mean']
                    std_val = v.get('std', 0.0)  # Default to 0 if std not present
                    ci_val = v.get('ci_95', 0.0)  # Default to 0 if ci_95 not present
                    f.write(f"{k:<20} {mean_val:.4f}     {std_val:.4f}     {ci_val:.4f}\n")
                    
            f.write("\n")
            f.write("="*60 + "\n")
            
        print(f"Report saved to {report_path}")
        
    def visualize_qualitative(self):
        # Determine actual number of samples to visualize
        if self.num_samples == -1:
            # Use all validation samples
            num_vis_samples = len(self.val_loader.dataset)
        else:
            num_vis_samples = self.num_samples
        
        print(f"Generating {num_vis_samples} overlay visualizations...")
        
        class_names = ['Background', 'Stroke']
        visualize_overlay_predictions(
            self.model,
            self.val_loader,
            self.device,
            class_names,
            self.output_dir,
            num_samples=num_vis_samples
        )
