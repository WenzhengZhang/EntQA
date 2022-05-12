# EntQA

This repo provides the code for our ICLR 2022 paper [EntQA: Entitly Linking as Question Answering](https://arxiv.org/pdf/2110.02369.pdf)

## Setup

```
conda create --name entqa python=3.8
conda activate entqa
pip install -r requirements.txt
conda install -c pytorch faiss-gpu cudatoolkit=11.0

```

## Download data & preprocess
All the preprocessed data can be downloaded [here](https://drive.google.com/drive/folders/1DQvfjKOuOoUE3YcYrg2GIvODaOEZXMdH?usp=sharing), you can skip following preprocess steps. 
Or preprocess by yourself: 
1. Download KILT wikipedia knowledge base [here](https://github.com/facebookresearch/KILT) and put it under a kb directory like /raw_kb/  \
2. Download BLINK pretrained retriever model [here](https://github.com/facebookresearch/BLINK)  \
3. Download AIDA CoNLL datasets [here](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads) and place them under a raw aida directory like /raw_aida/ \
4. Download entity title map dictionary [here](https://drive.google.com/file/d/1QE3N8S_tVkGhYz_5fjRahLHfkIwghi-4/view?usp=sharing) and put it under /raw_aida/ for remapping outdated entities of AIDA datasets to KILT wikipedia entity titles \
5. preprocess AIDA data and KILT kb by
```
python preprocess_data.py \
--raw_dir /raw_aida/  --out_aida_dir /retriever_input/  \
--raw_kb_path /raw_kb/[kilt wikipedia file name] --out_kb_path /kb/entities_kilt.json \
--max_ent_len 128  --instance_length 32 --stride 16 --pos_prop 1 --title_map_dir /raw_aida/

```

## Train Retriever 

Train retriever by 
```
python run_retriever.py \
--model /model_retriever/retriever.pt  --data_dir /retriever_input/   --kb_dir /kb/ \
--k 100 --num_cands 64  --pretrained_path /blink/BLINK/models/ --gpus 0,1,2,3  --max_len 42   \
--mention_bsz 4096 --entity_bsz 2048  --epochs 4  --B 4  --lr 2e-6  --rands_ratio 0.9   \
--logging_step 100 --warmup_proportion 0.2  --out_dir /reader_input/   
--gradient_accumulation_steps 2  --type_loss sum_log_nce   \
--cands_embeds_path /candidates_embeds/candidate_embeds.npy \
--blink  --add_topic
```
It takes 10 hours on 4 A100 GPUs to finish the retriever training experiments. It takes up 32G GPU memory for the main GPU and 23.8G GPU memory for other GPUs.
### Retriever Local Evaluation
1. You can train the retriever by yourself using the above scripts to get your trained retriever and entity embeddings. You can also Download our trained retriever [here](https://drive.google.com/file/d/1bHS5rxGbHJ5omQ-t8rjQogw7QJq-qYFO/view?usp=sharing), our cached entity embeddings [here](https://drive.google.com/file/d/1znMYd5HS80XpLpvpp_dFkQMbJiaFsQIn/view?usp=sharing). 
2. Use the above training scripts and set `--epochs` to be 0 for evaluation.
### Retrieval Results
| val Recall@100 | test Recall@100 | val LRAP | test LRAP | val passage-level Recall@100 | test passage-level Recall@100|
|:----------------:|:-----------------:|:----------:|:-----------:|:---------------------:|:---------------------:|
|     98.17%     |     96.62%      |  83.98%  |  82.65%   |      97.03%         |        94.59%       |


**Recall@k** is the percentage of total number of positive entities retrieved by the topk candidates with respect to the total number of gold entities for all the query passages. \
**passage-level Recall@k** is the percentage of the number of passages with all the gold entities retrieved in the topk candidates with respect to the number of passages. \
**LRAP** is [Label ranking average precision ](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_average_precision_score.html) which measures the multi-label ranking performance.



## Train Reader 

Train reader by

```
python run_reader.py  \
--model /model_reader/reader.pt   --data_dir /reader_input/  \
--C 64  --B 2  --L 180  --C_val 100  --gpus 0,1   --val_bsz 32 \
--gradient_accumulation_steps 2  --warmup_proportion 0.06  \
--epochs 4  --lr 1e-5 --thresd  0.05  --logging_steps 100  \
--k 3  --stride 16 --max_passage_len 32  --filter_span  \
--type_encoder squad2_electra_large  \
--type_span_loss sum_log  --type_rank_loss sum_log  \
--do_rerank  --add_topic  --results_dir /reader_results/  --kb_dir /kb/

```
It takes about 6 hours on 2 A100 GPUs to finish the reader training experiment. It takes up 36G GPU memory for the main GPU and 32G GPU memory for the other GPU.
### Reader Local Evaluation
1. You can follow the above instructions to train your reader or you can download our trained reader [here](https://drive.google.com/file/d/1A4I1fJZKxmROIE1fd0mdXN6b1emP_xt4/view?usp=sharing)
2. You can run retriever local evaluation for your trained retriever or our trained retriever to get reader input data. Or you can download our reader input data [here](https://drive.google.com/drive/folders/1xfEgXCREe6pbSmAsnidsMuVYMK_mlOao?usp=sharing)
3. Use the above reader training scripts and set `--epochs` to be 0 for evaluation.

### Reader Results

|   val F1  |  test F1  |  val Recall |  test Recall |  val Precision  |  test Precision  |
|:-----------:|:-----------:|:-------------:|:--------------:|:-----------------:|:------------------:|
|   87.32%  |   84.4%  |   90.23%    |    87.0%    |     84.6%      |      81.96%      |

## GERBIL evaluation
Our GERBIL evaluation steps follow [here](https://github.com/dalab/end2end_neural_el), specifically:
1. Download our snapshot of GERBIL repo [here](https://drive.google.com/file/d/1Sp-G9631ormzIYfenCDsaWgiBPVBkF6F/view?usp=sharing), our pretrained retriever [here](https://drive.google.com/file/d/1bHS5rxGbHJ5omQ-t8rjQogw7QJq-qYFO/view?usp=sharing), our cached entities embeddings [here](https://drive.google.com/file/d/1znMYd5HS80XpLpvpp_dFkQMbJiaFsQIn/view?usp=sharing) and our pretrained reader [here](https://drive.google.com/file/d/1A4I1fJZKxmROIE1fd0mdXN6b1emP_xt4/view?usp=sharing)
2. One one terminal/screen run GERBIL by:
```
cd gerbil
./start.sh

```
3. On another terminal/screen run:
```
cd gerbil-SpotWrapNifWS4Test/
mvn clean -Dmaven.tomcat.port=1235 tomcat:run

```
4. On a third terminal/screen run:
```
python gerbil_experiments/server.py  \
--gpus 0,1,2,3  --log_path  /logs/log.txt --blink_dir //blink_model/  \
--retriever_path /model_retriever/retriever.pt   \
--cands_embeds_path /candidates_embeds/candidate_embeds.npy   \
--ents_path /kb/entities_kilt.json  \
--reader_path /model_reader/reader.pt   \
--add_topic  --do_rerank  --bsz_retriever 8192  

```
Open the url http://localhost:1234/gerbil
- Configure experiment
- Select A2KB as experiment type
- Select strong evaluation as Matching
- In URI field write: http://localhost:1235/gerbil-spotWrapNifWS4Test/myalgorithm
- Name: whatever you wish
- Press add annotator button
- Select the datasets that you want to evaluate the model on
- Run experiment

### GERBIL Results
| AIDA testb| MSNBC| Der|K50|R128|R500|OKE15|OKE16|AVG|
|:---------:|:--------:|:----:|:---:|:----:|:----:|:-----:|:-----:|:----:|
|85.82%|72.09%|52.85%|64.46%|54.05%|41.93%|61.10%|51.34%|60.46%|
