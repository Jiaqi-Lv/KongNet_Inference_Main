# KongNet Inference Main

Whole slide image (WSI) Inference pipeline for KongNet models supporting multiple histopathology datasets including MIDOG, PanNuke and MONKEY.

## 🚀 Features

- **Multi-dataset Support**: MIDOG, PanNuke and MONKEY inference pipelines
- **Whole Slide Image Processing**: Efficient WSI inference through multiprocessing and caching
- **Pre-trained Models**: Ready-to-use model weights from HuggingFace for immediate inference
- **Quality Control**: Built-in tissue detection using GrandQC

## 📋 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### System Requirements
- Python 3.10+
- CUDA-compatible GPU
- Sufficient RAM for WSI processing (Recommanded at least 32GB)

## 🚀 Quick Start

### 1. Download Model Weights

The pre-trained model weights are automatically downloaded from HuggingFace when needed. Alternatively, you can manually place them in the `model_weights/` directory.

### 2. MIDOG Inference

```bash
python inference_MIDOG.py --input_path /path/to/input/images --output_path /path/to/output
```

### 3. PanNuke Inference

```bash
python inference_panNuke.py --input_path /path/to/input/images --output_path /path/to/output
```

## 📝 Output Format

### MIDOG Output
- **Format**: JSON with bounding boxes
- **Classes**: Mitotic figures, non-mitotic nuclei
- **Coordinates**: Pixel coordinates in original WSI

### PanNuke Output
- **Format**: HDF5 with segmentation masks
- **Classes**: 6 cell types (Neoplastic, Inflammatory, Connective, Dead, Epithelial)
- **Masks**: Instance segmentation masks

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Disable model ensembling (Disabled by default)

3. **Model Weight Download Fails**
   - Check internet connection
   - Manually download from Hugging Face

### Performance Optimization

- Use SSD for Cache storage
- Increase `num_workers`
- Use appropriate batch size for your GPU memory

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

- [MIDOG Dataset](https://midog.grand-challenge.org/)
- [PanNuke Dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Note**: This inference pipeline is designed for research purposes. For clinical applications, please ensure proper validation and regulatory compliance.