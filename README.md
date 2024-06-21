# CoAug

This is the companion code for the research paper 
"[CoAug: Combining Augmentation of Labels and Labeling Rules]()". 
The code allows the users to reproduce and extend the results reported in the study. 
Please cite the above paper when reporting, reproducing or extending the results.

### Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication "[CoAug: Combining Augmentation of Labels and Labeling Rules]()". It will neither be
maintained nor monitored in any way.

## 1. Setup 

Setup environment by running `source bin/init.sh`. This will 

- Install and setup environment with correct dependencies.
- Download the QuIP models for CoAug+QuIP experiments
- We assume python and python venv is already installed in the system. The script has been verified to run with python 3.8.

## 2. Experiments in Paper
In this section, we introduce how to reimplement the experiments in our paper. We already include all needed datasets and rule files in this repo.

At the start of every experiment, please run `bash bin/setup.sh` to setup the right environment.

### TaLLOR
`python train_proto.py --dataset ${dataset} --encoder ${encoder} --mode tallor --seed ${seed} --rule_topk 20 --ap_threshold 0.75`

where 
`${dataset}` is one of bc5cdr/ncbi_disease/conll2003/wikigold
`${encoder}` is scibert (science-domain) or bert (general-domain)

### ProtoBERT
`python train_proto.py --dataset ${dataset} --encoder ${encoder} --mode proto --seed ${seed} --rule_topk 20 --ap_threshold 0.75`

where 
`${dataset}` is one of bc5cdr/ncbi_disease/conll2003/wikigold
`${encoder}` is scibert (science-domain) or bert (general-domain)

### CoAug + ProtoBERT
`python train_proto.py --dataset ${dataset} --encoder ${encoder} --mode coaug --seed ${seed} --rule_topk 20 --ap_threshold 0.75`

where 
`${dataset}` is one of bc5cdr/ncbi_disease/conll2003/wikigold
`${encoder}` is scibert (science-domain) or bert (general-domain)

### QuIP
`python train_quip.py --dataset ${dataset} --encoder ${encoder} --mode quip --seed ${seed} --rule_topk 20 --ap_threshold 0.75`

where 
`${dataset}` is one of bc5cdr/ncbi_disease/conll2003/wikigold
`${encoder}` is scibert (science-domain) or bert (general-domain)

### CoAug + QuIP
`python train_quip.py --dataset ${dataset} --encoder ${encoder} --mode coaug --seed ${seed} --rule_topk 20 --ap_threshold 0.75`

where 
`${dataset}` is one of bc5cdr/ncbi_disease/conll2003/wikigold
`${encoder}` is scibert (science-domain) or bert (general-domain)

The output will be in the experiment directory `exp_out/{dataset_name}/{tallor/proto/quip/coaug}/{ProtoBERT/QuIP}/{seed}/{timestamp}/`.

## 3. Dataset
4 datasets are preprocessed and included in this repository.

| Dataset      | Task code    | Dir               | Source  |
|--------------|--------------|-------------------|---------|
| BC5CDR       | bc5cdr       | data/bc5cdr       | [link](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/)|
| NCBI Disease | ncbi_disease | data/ncbi_disease | [link](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/)|
| CoNLL 2003   | conll2003    | data/conll2003    | [link](https://arxiv.org/pdf/cs/0306050v1.pdf)|
| Wiki Gold    | wikigold     | data/wikigold     | [link](https://huggingface.co/datasets/knilakshan20/wikigold)|

### Dataset Prepration
We follow **[TALLOR](https://github.com/JiachengLi1995/TALLOR)**'s process to prepare the dataset, which uses **[AutoPhrase](https://github.com/shangjingbo1226/AutoPhrase)** for preprocessing.
For details, please refer to **[TALLOR](https://github.com/JiachengLi1995/TALLOR)**'s introduction.



## 4. Contact

For any doubts or questions regarding the work, please contact Rakesh ([rrmenon@cs.unc.edu](mailto:rrmenon+coaug@cs.unc.edu)). For any bug or issues with the code, feel free to open a GitHub issue or pull request.

## 5. Citation

    @inproceedings{menon2023coaug,
        Author = {Menon, Rakesh R. and Wang, Bingqing and Araki, Jun and Zhou, Zhengyu and Feng, Zhe and Ren, Liu},
        Title = {{CoAug}: {C}ombining {A}ugmentation of {L}abels and {L}abeling {R}ules},
        booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
        month = jul,
        year= "2023",
        address = "Toronto, Canada",
        publisher = "Association for Computational Linguistics"
    }

## License

CoAug is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in CoAug, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).