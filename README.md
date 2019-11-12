# improvedsegan
This repository is an extension of GAN based speech enhancement model called SEGAN, and we present two modifications to make the training model more robust and stable. The details of implementation and introduction of the device recorded version of VCTK dataset (DRVCTK) are explained in [our paper](http://arxiv.org/abs/1911.03952).   
The SEGAN original paper can be found [here](https://arxiv.org/abs/1703.09452) and the script is publicly available [here](https://github.com/santi-pdp/segan). The source code, and content of this README file is mostly based on the original SEGAN project. 

### Introduction

In this work a Generative Adversarial approach has been taken to do speech enhancement (i.e. removing noise from corrupted speech signals) with a fully convolutional architecture.

This model deals with raw speech waveforms on many noise conditions at different SNRs (52 at training time and 24 during test). It also models the speech characteristics from many speakers mixed within the same structure (without any supervision of identities), which makes the generative structure generalizable in the noise and speaker dimensions.

**All the project is developed with TensorFlow**. There are two repositories that were good references on how GANs are defined and deployed:

* [improved-gan](https://github.com/openai/improved-gan): implementing improvements to train GANs in a more stable way
*  [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow): implementation of the DCGAN in tensorflow

### Dependencies

* Python 2.7
* TensorFlow 0.12

You can install the requirements either to your virtualenv or the system via pip with:

```
pip install -r requirements.txt
```

### Data

In this work, two speech enhancement datasets are used. The first one is device recorded version of VCTK corpus which can be found in [Edinburgh DataShare](https://129.215.41.53/handle/10283/3038). The second dataset is [(Valentini et al. 2016)](http://ssw9.net/papers/ssw9_PS2-4_Valentini-Botinhao.pdf) which also can be found in [Edinburgh DataShare](http://datashare.is.ed.ac.uk/handle/10283/1942). However, the following script downloads and prepares the second dataset for TensorFlow format:

```
./prepare_data.sh
```

Or alternatively download the dataset, convert the wav files to 16kHz sampling and set the `noisy` and `clean` training files paths in the config file `e2e_maker.cfg` in `cfg/`. Please pay attention in addition of clean and noisy sets you need to prepare the baseline set and add the address to the e2e_maker.cfg file. Then run the script:

```
python make_tfrecords.py --force-gen --cfg cfg/e2e_maker.cfg
```

### Training

Once you have the TFRecords file created in `data/segan.tfrecords` you can simply run the training process with:

```
./train_segan.sh
```

By default this will take all the available GPUs in your system, if any. Otherwise it will just take the CPU.

**NOTE:** If you want to specify a subset of GPUs to work on, you can do so with the `CUDA_VISIBLE_DEVICES="0, 1, <etc>"` flag in the python execution within the training script. In the case of having two GPUs they'll be identified as 0 and 1, so we could just take the first GPU with: `CUDA_VISIBLE_DEVICES="0"`.


### Loading model and prediction

First, you need to train the model to have the checkpoint files in segan_v1 folder.

Then the `main.py` script has the option to process a wav file through the G network (inference mode), where the user MUST specify the trained weights file and the configuration of the trained network. In the case of the v1 SEGAN presented in the paper, the options would be:

```
CUDA_VISIBLE_DEVICES="" python main.py --init_noise_std 0. --save_path segan_v1 \
                                       --batch_size 100 --g_nl prelu --weights SEGAN_full \
                                       --test_wav <wav_filename> --clean_save_path <clean_save_dirpath>
```

To make things easy, there is a bash script called `clean_wav.sh` that accepts as input argument the test filename and
the save path.

### Modifications ###
Currently input arguments of clean_wav.sh can be test filename or test foldername. The structure for multiple wav testing is improved. In this implementation, for the last chunk instead of zero padding, we pre-padded it with previous samples. We implement residual output in generator using element-wise sum of noisy input with enhanced output. We implement directional initialization in GAN with using 2 interations in generator. In the first iteration of first phase, the loss function will be computed between the clean speech and output of the generator, and in the second iteration will be computed between pre-trained baseline model and output of the generator. With --baseline_remove_epoch input argument in main.py you can set the epoch threshold for removing the extra baseline iteration in generator. In cfg/e2e_maker.cfg in addition of clean and noisy folder, you must set the address of baseline folder.



### Segan Authors

* **Santiago Pascual** (TALP-UPC, BarcelonaTech)
* **Antonio Bonafonte** (TALP-UPC, BarcelonaTech)
* **Joan Serrà** (Telefónica Research, Barcelona)

### Contact

* **Seyyed Saeed Sarfjoo** (NII, Tokyo)
e-mail: saeed.sarfjoo@ozu.edu.tr
* **Santi Pascual**
e-mail: santi.pascual@upc.edu


### Related papers

```
@article{sarfjoo2019improved,
  title={Transformation of low-quality device-recorded speech to high-quality speech using improved SEGAN model},
  author={Sarfjoo, Seyyed Saeed and Wang, Xin and Henter, Eje Gustav and Lorenzo-Trueba, Jaime and Takaki, Shinji and Yamagishi, Junichi},
  journal={arXiv preprint arXiv:1911.03952},
  year={2019}
}

@article{pascual2017segan,
  title={SEGAN: Speech enhancement generative adversarial network},
  author={Pascual, Santiago and Bonafonte, Antonio and Serra, Joan},
  journal={arXiv preprint arXiv:1703.09452},
  year={2017}
}
```
