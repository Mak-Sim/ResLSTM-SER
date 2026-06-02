# RAVDESS Dataset

The RAVDESS dataset is **not included** in this repository due to its size and licensing terms.

## Download

Download the *"Audio_Speech_Actors_01-24.zip"* from one of these sources:

- [Kaggle: RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- [Zenodo (official)](https://zenodo.org/record/1188976)

## Expected Directory Structure

Extract the archive so the resulting structure is:

```
data/ravdess/audio_speech_actors_01-24/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   ├── 03-01-01-01-01-02-01.wav
│   └── ...
├── Actor_02/
│   └── ...
└── Actor_24/
    └── ...
```

## Extracted Features

After running `python train.py --mode extract`, preprocessed features will be stored in:

```
data/ravdess/extracted/
├── X_0_mfcc_34_n_chroma_12_nfft_4096_hop_size_2048.npy
├── X_1_mfcc_34_n_chroma_12_nfft_4096_hop_size_2048.npy
├── ...
├── y_labels_mfcc_34_n_chroma_12_nfft_4096_hop_size_2048.npy
├── IDs_mfcc_34_n_chroma_12_nfft_4096.npy
├── global_mean.npy
└── global_std.npy
```

## Citation

```bibtex
@article{livingstone2018ravdess,
  title={The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): 
         A dynamic, multimodal set of facial and vocal expressions in North American English},
  author={Livingstone, Steven R and Russo, Frank A},
  journal={PLoS ONE},
  volume={13},
  number={5},
  pages={e0196391},
  year={2018},
  publisher={Public Library of Science}
}
```
