
# Detecting Replay Attacks Using Multi-Channel Audio: A Neural Network-Based Method

With the rapidly growing number of security sensitive systems that use voice as the primary input, it becomes increasingly important to address these systems’ potential vulnerability to replay attacks. Previous efforts to address this concern have focused primarily on single-channel audio. In this work, we introduce a novel neural network-based replay attack detection model that further leverages spatial information of multi-channel audio and is able to significantly improve the replay attack detection performance.

## Updates
June, 30, 2020:
- New uploads, might contain bugs

## Dataset
**ReMASC Corpus** (92.3GB)

Data we used in the paper. Please download it from **\[[IEEE DataPort](https://ieee-dataport.org/open-access/remasc-realistic-replay-attack-corpus-voice-controlled-systems)]**,  it is free. You will need an [IEEE account](https://ieee-dataport.org/faq/how-do-i-access-dataset-ieee-dataport) to download, which is also free.

The complete set consists of two disjoint set:

- **Core Set**: the suggest training and development set.
- **Evaluation Set**: the suggest evaluation set. 

We also include an unsplited complete set in the .zip package in case you want to customerize the data split. It is just a simple union of the core set and the evaluation set. 


## Cite us:  
If you use our neural network code, please cite the following paper:

Yuan Gong, Jian Yang, Christian Poellabauer, ["Detecting Replay Attacks Using Multi-Channel Audio: A Neural Network-Based Method"](https://arxiv.org/abs/2003.08225)  IEEE Signal Processing Letters, 2020.

If you also use the data, please cite the following paper:

Yuan Gong, Jian Yang, Jacob Huber, Mitchell MacKnight, Christian Poellabauer, ["ReMASC: Realistic Replay Attack Corpus for Voice Controlled Systems"](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/1541.html), Interspeech 2019.

## How to run the code?
**1. Prepare the dataset**
1) Clone the Github reporsitory. Download the ReMASC corpus from **\[[here](https://ieee-dataport.org/open-access/remasc-realistic-replay-attack-corpus-voice-controlled-systems)]** (it is completely free) and place it in the ``data/`` directory. 

2) In ``src/constants.py``, line 9, change the ``PROJ_PATH`` to your project path.

3) In ``src/uniform_sample_rate.py``, line 14, change to your desiring sampling rate (in this work, we use 44100), then run ``python src/uniform_sample_rate.py``.

4) Valid the data preparsion by running ``python src/data_loader.py``, you should see a plot of the waveform from the dataset.

**2. Select the hyper-parameters and run experiment**

In ``src/full_exp.py``, line 81-91, the hyper-parameters are defined:

```python
bsize_list = [64]
lr_list = [1e-5]
rdevice_list = [1, 2, 3, 4]
audio_len_list = [1.0]
filter_num_list = [64]
sr_list = [44100]
mch_setting = [True]
frame_time_list = [0.02]
```
where ``bsize_list`` defines the list of batch size, ``lr_list`` defines the list of learning rate, ``rdevice_list`` defines the list of recording devices, ``audio_len_list`` defines the list of used audio length, ``filter_num_list`` defines the list of convolution filter number in the first layer, ``sr`` list defines the list of sampling rate (you must first convert the sample rate using ``src/uniform_sample_rate.py`` before runing experiments), ``mch_setting`` defines if use real multi-channel, this should be ``True`` unless you are running an ablation study, ``frame_time_list`` defines a list of frame window size in second. Note you can test different settings in one run by adding multiple values in a list (e.g., ``bsize_list=[8, 16, 32, 64]``), all hyper-parameter combination will be tested. Nevertheless, the running time grows exponentially.

**3. Use your own model**
The model we propose is in ``src/model.py``, you can revise it or use your own model to replace it. 

## Questions

If you have a question, please rasie an issue in this Github reporsity. You can also contact Yuan Gong (ygong1@nd.edu).
