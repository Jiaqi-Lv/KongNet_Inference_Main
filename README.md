# KongNet Inference Main

Whole slide image (WSI) Inference pipeline for KongNet models supporting multiple histopathology datasets including MIDOG, PanNuke and MONKEY.

## 🚀 Features

- **Multi-dataset Support**: MIDOG, PanNuke and MONKEY inference pipelines
- **Whole Slide Image Processing**: Efficient WSI inference through multiprocessing and caching
- **Pre-trained Models**: Ready-to-use model weights from HuggingFace for immediate inference
- **Model Ensemble**: Support for multiple model checkpoints and ensemble inference
- **Test Time Augmentation**: Built-in TTA with configurable transformations
- **Quality Control**: Built-in tissue detection using GrandQC

## 📋 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### System Requirements
- Python 3.10+
- CUDA-compatible GPU (required)
- Sufficient RAM for WSI processing (Recommended at least 32GB)

## 🚀 Quick Start

### 1. Basic MIDOG Inference

```bash
# Process all WSI files in a directory
python inference_MIDOG.py --input_dir /path/to/wsi/files --output_dir /path/to/results

# Process a single WSI file
python inference_MIDOG.py --input_dir /path/to/wsi/files --single_wsi sample.svs --output_dir /path/to/results
```

### 2. Basic PanNuke Inference

```bash
# Process all WSI files in a directory
python inference_panNuke.py --input_dir /path/to/wsi/files --output_dir /path/to/results

# Process a single WSI file
python inference_panNuke.py --input_dir /path/to/wsi/files --single_wsi sample.svs --output_dir /path/to/results
```

## 📖 Usage Examples

### Basic Usage

#### MIDOG
```bash
# Minimal command - uses default paths
python inference_MIDOG.py

# Specify input and output directories
python inference_MIDOG.py \
    --input_dir /data/midog/wsi \
    --output_dir /results/midog \
    --cache_dir /tmp/cache
```

#### PanNuke
```bash
# Minimal command - uses default paths
python inference_panNuke.py

# Specify input and output directories
python inference_panNuke.py \
    --input_dir /data/pannuke/wsi \
    --output_dir /results/pannuke \
    --cache_dir /tmp/cache
```

### Model Configuration

#### MIDOG
```bash
# Use local model weights (bypass HuggingFace download)
python inference_MIDOG.py \
    --local_weights /path/to/model1.pth /path/to/model2.pth \
    --input_dir /data/wsi

# Model ensemble with multiple checkpoints
python inference_MIDOG.py \
    --checkpoint_name "KongNet_Det_MIDOG_1.pth" \
    --additional_checkpoints "KongNet_Det_MIDOG_2.pth" "KongNet_Det_MIDOG_3.pth" \
    --input_dir /data/wsi
```

#### PanNuke
```bash
# Use local model weights (bypass HuggingFace download)
python inference_panNuke.py \
    --local_weights /path/to/pannuke_model1.pth /path/to/pannuke_model2.pth \
    --input_dir /data/wsi

# Use specific HuggingFace repository and checkpoint
python inference_panNuke.py \
    --hf_repo_id "TIACentre/KongNet_PanNuke" \
    --checkpoint_name "KongNet_PanNuke_1.pth" \
    --input_dir /data/wsi

# Model ensemble with multiple checkpoints
python inference_panNuke.py \
    --checkpoint_name "KongNet_PanNuke_1.pth" \
    --additional_checkpoints "KongNet_PanNuke_2.pth" "KongNet_PanNuke_3.pth" \
    --input_dir /data/wsi
```

### Processing Options

#### MIDOG
```bash
# Disable test time augmentation for faster inference
python inference_MIDOG.py \
    --no_tta \
    --input_dir /data/wsi

# Process single WSI with mask
python inference_MIDOG.py \
    --single_wsi "sample.svs" \
    --mask_dir /path/to/masks \
    --input_dir /data/wsi
```

#### PanNuke
```bash
# Disable test time augmentation for faster inference
python inference_panNuke.py \
    --no_tta \
    --input_dir /data/wsi

# Process single WSI with mask
python inference_panNuke.py \
    --single_wsi "sample.svs" \
    --mask_dir /path/to/masks \
    --input_dir /data/wsi
```

### Advanced Examples

#### MIDOG
```bash
# Full configuration example
python inference_MIDOG.py \
    --input_dir /data/midog/test_set \
    --output_dir /results/midog_predictions \
    --cache_dir /fast_ssd/cache \
    --weights_dir ./pretrained_models \
    --hf_repo_id "TIACentre/KongNet_MIDOG" \
    --checkpoint_name "KongNet_Det_MIDOG_1.pth" \
    --additional_checkpoints "KongNet_Det_MIDOG_2.pth" \
    --mask_dir /data/midog/tissue_masks

# Batch processing with custom weights
python inference_MIDOG.py \
    --local_weights ./models/midog_fold1.pth ./models/midog_fold2.pth ./models/midog_fold3.pth \
    --input_dir /data/test_cases \
    --output_dir /results/ensemble_predictions \
    --no_tta
```

#### PanNuke
```bash
# Full configuration example
python inference_panNuke.py \
    --input_dir /data/pannuke/test_set \
    --output_dir /results/pannuke_predictions \
    --cache_dir /fast_ssd/cache \
    --weights_dir ./pretrained_models \
    --hf_repo_id "TIACentre/KongNet_PanNuke" \
    --checkpoint_name "KongNet_PanNuke_1.pth" \
    --additional_checkpoints "KongNet_PanNuke_2.pth" \
    --mask_dir /data/pannuke/tissue_masks

# Batch processing with custom weights
python inference_panNuke.py \
    --local_weights ./models/pannuke_fold1.pth ./models/pannuke_fold2.pth ./models/pannuke_fold3.pth \
    --input_dir /data/test_cases \
    --output_dir /results/ensemble_predictions \
    --no_tta

# High-throughput processing with optimized settings
python inference_panNuke.py \
    --input_dir /data/large_dataset \
    --output_dir /results/batch_pannuke \
    --cache_dir /nvme_ssd/cache \
    --no_tta \
    --local_weights ./best_pannuke_model.pth
```

## 🔧 Command Line Arguments

### Required Arguments (Both Scripts)
- `--input_dir`: Directory containing WSI files
- `--output_dir`: Directory to save results

### Model Configuration

#### MIDOG
- `--hf_repo_id`: HuggingFace repository ID (default: "TIACentre/KongNet_MIDOG")
- `--checkpoint_name`: Main checkpoint filename (default: "KongNet_Det_MIDOG_1.pth")
- `--additional_checkpoints`: Additional checkpoints for ensemble
- `--local_weights`: Use local weight files instead of HuggingFace
- `--weights_dir`: Directory to store downloaded weights (default: "./model_weights")

#### PanNuke
- `--hf_repo_id`: HuggingFace repository ID (default: "TIACentre/KongNet_PanNuke")
- `--checkpoint_name`: Main checkpoint filename (default: "KongNet_PanNuke_1.pth")
- `--additional_checkpoints`: Additional checkpoints for ensemble
- `--local_weights`: Use local weight files instead of HuggingFace
- `--weights_dir`: Directory to store downloaded weights (default: "./model_weights")

### Processing Options (Both Scripts)
- `--no_tta`: Disable test time augmentation
- `--single_wsi`: Process only specified WSI filename
- `--mask_dir`: Directory containing tissue masks
- `--cache_dir`: Directory for intermediate caching (default: "/home/u1910100/cloud_workspace/data/cache")

## 📝 Output Format

### MIDOG Output
- **Format**: JSON with detection results
- **Location**: `{output_dir}/{wsi_name}_detections.json`
- **Content**: 
  ```json
  {
    "detections": [
      {
        "x": 1234,
        "y": 5678,
        "confidence": 0.95,
        "class": "mitotic"
      }
    ],
    "metadata": {
      "wsi_name": "sample.svs",
      "processing_time": 45.2,
      "model_version": "KongNet_Det_MIDOG_1"
    }
  }
  ```

### PanNuke Output
- **Format**: HDF5 with segmentation masks
- **Location**: `{output_dir}/{wsi_name}_segmentation.h5`
- **Classes**: 6 cell types (Neoplastic, Inflammatory, Connective, Dead, Epithelial, Overall)
- **Content**:
  ```python
  # Load results
  import h5py
  with h5py.File('sample_segmentation.h5', 'r') as f:
      neoplastic_mask = f['neoplastic'][:]
      inflammatory_mask = f['inflammatory'][:]
      connective_mask = f['connective'][:]
      dead_mask = f['dead'][:]
      epithelial_mask = f['epithelial'][:]
      overall_mask = f['overall'][:]
  ```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   # Disable model ensemble (Default option)
   ```

2. **Model Weight Download Fails**
   ```bash
   # Check internet connection and try again
   # Or use local weights:
   python inference_MIDOG.py --local_weights /path/to/model.pth
   python inference_panNuke.py --local_weights /path/to/model.pth
   ```

3. **WSI File Not Found**
   ```bash
   # Check file extensions are supported: .svs, .tif, .tiff, .ndpi, .mrxs
   # Verify file permissions and path
   ```

4. **Insufficient RAM**
   ```bash
   # Reduce number of workers
   # Reduce batch size
   ```

### Performance Optimization

- **Use SSD for cache storage** (`--cache_dir /ssd/path`)
- **Use appropriate batch size for your GPU**
- **Disable TTA for faster inference** (`--no_tta`) (detection quality trade-off)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Citation

If you use this code in your research, please cite:

```bibtex
@article{kongnet2024,
  title={KongNet: A Multi-headed Deep Learning Model for Accurate Detection and Classification of Nuclei in Histopathology Images},
  author={Jiaqi Lv, Esha Sadia Nasir, Kesi Xu, Mostafa Jahanifar, Brinder Singh Chohan, Behnaz Elhaminia, Shan E Ahmed Raza},
  year={2025}
}
```

## 📚 References

- [2025 MIDOG Challenge](https://midog.grand-challenge.org/)
- [PanNuke Dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)
- [PyTorch Documentation](https://pytorch.org/docs/)

---