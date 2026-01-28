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
    def __init__(self, model, val_loader, device, config):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.config = config
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
                outputs, _, _ = self.model(images, metadata) # SymFormer returns (out, clusters, map)
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
                    f.write(f"{k:<20} {v['mean']:.4f}     {v['std']:.4f}     {v.get('ci_95', 0):.4f}\n")
                    
            f.write("\n")
            f.write("="*60 + "\n")
            
        print(f"Report saved to {report_path}")
        
    def visualize_qualitative(self):
        class_names = ['Background', 'Stroke']
        visualize_overlay_predictions(
            self.model,
            self.val_loader,
            self.device,
            class_names,
            self.output_dir,
            num_samples=5
        )
