# Exploring Scaling Laws of CTR Model for Online Performance Improvement

> **A scalable CTR model inspired by LLM to explore scaling laws **  
> 🔗 [Paper (RecSys '25)](https://doi.org/10.1145/3705328.3748046) | 💻 [Code](https://github.com/laiweijiang/SUAN)

## 🧱 Model Architecture

### SUAN (Stacked UABs)
<img width="1280" height="548" alt="image" src="https://github.com/user-attachments/assets/c7b8d2cb-e8b3-4310-846a-ff45a478054f" />

Each **Unified Attention Block (UAB)** contains:
- **Self-Attention**: Spatiotemporal behavior modeling
- **Cross-Attention**: User profile-guided importance scoring  
- **Dual Alignment Attention**: Feature selection
- **RMSNorm** + **SwiGLU FFN** (LLM-inspired)

> 📌 Input: **Target-aware sequence** = User behaviors + candidates  
> 📌 Output: `P(click|S,p,c) = σ(MLP(E_block[-1,:], e_p, e_other))`

## 📁 Open-Sourced Components

Due to industrial deployment constraints, we release:
### ✅ 1. Core Model Code
- File: `./handle_layer/handle_lib/handle_rec_unit.py`
- Key classes:
  - `Mix1k_SUAN`: For industrial dataset
  - `Eleme_SUAN`: For Eleme dataset

### ✅ 2. Experiment Configs
- `exp/user1/Mix1k_SUAN/`: Industrial dataset config
- `exp/user1/Eleme_SUAN/`: Eleme dataset config

## 📚 Citation
```bibtex
@inproceedings{lai2025exploring,
  title={Exploring Scaling Laws of CTR Model for Online Performance Improvement},
  author={Lai, Weijiang and Jin, Beihong and Zhang, Jiongyan and Zheng, Yiyuan and Dong, Jian and Cheng, Jia and Lei, Jun and Wang, Xingxing},
  booktitle={Proceedings of the Nineteenth ACM Conference on Recommender Systems},
  pages={114--123},
  year={2025},
  organization={ACM}
}
```

## 📬 Contact
- Email: laiweijiang22@otcaix.iscas.ac.cn
- Affiliation: Institute of Software, Chinese Academy of Sciences
- GitHub: [https://github.com/laiweijiang/SUAN](https://github.com/laiweijiang/SUAN)

⭐ **Star us if you find it useful!**

