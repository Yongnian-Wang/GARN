# GARN:Global attention retinex network for low light image enhancement

# Tensorflow 
Tensorflow implementation of GARN

## Requirements
1. Python 3.7 
2. Tensorflow 1.14
3. opencv
4. cuda 10.1

### Dataset
LOL https://daooshee.github.io/BMVC2018website/  
SICE https://github.com/csjcai/SICE#learning-a-deep-single-image-contrast-enhancer-from-multi-exposure-images  
MIT 5K http://data.csail.mit.edu/graphics/fivek/  
VV https://sites.google.com/site/vonikakis/datasets  
SID http://vladlen.info/publications/learning-see-dark/  

### Folder structure
Download the GARN_code first.
The following shows the basic folder structure.
```

├── data
│   ├── test_data # testing data. 
│   │   ├── LIME 
│   │   └── MEF
│   │   └── NPE
│   │   └── ...
│   ├── train_data 
│   ├── val_data 
├── decomposition_net_train.py # decomposition_net_train code
├── adjustment_net_train.py # adjustment_net_train code
├── reflectance_restoration_net_train.py # reflectance_restoration_net_train code
├── evaluate.py # test code
├── model.py # GARN network
├── utils.py # Data set processing
├── checkpoint # Training model data
│   ├── decom_net_train #  A pre-trained
│   ├── illumination_adjust_net_train #  A pre-trained
│   ├── Restoration_net_train #  A pre-trained
```

## Bibtex
```
@article{wang2023global,
  title={Global attention retinex network for low light image enhancement},
  author={Wang, Yongnian and Zhang, Zhibin},
  journal={Journal of Visual Communication and Image Representation},
  pages={103795},
  year={2023},
  publisher={Elsevier}
}
```



