# EntQA

This repo provides the code for our ICLR 2022 paper [EntQA: Entitly Linking as Question Answering](https://arxiv.org/pdf/2110.02369.pdf)

## Setup

```
conda create --name entqa python=3.8
conda activate entqa
pip -r install requirements.txt
conda install -c pytorch faiss-gpu cudatoolkit=11.0

```

## Download data & preprocess
1.Download KILT wikipedia knowledge base [here](https://github.com/facebookresearch/KILT) and put it under a kb directory like /raw_kb/  \
2. Download BLINK pretrained retriever model [here](https://github.com/facebookresearch/BLINK)
3. Download AIDA CoNLL datasets [here](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads) and place them under a raw aida directory like /raw_aida/ \
4. Download entity title map dictionary [here](https://drive.google.com/file/d/1QE3N8S_tVkGhYz_5fjRahLHfkIwghi-4/view?usp=sharing) and put it under /raw_aida/ for remapping outdated entities of AIDA datasets to KILT wikipedia entity titles \
5. preprocess AIDA data and KILT kb by
```
python preprocess_data.py \
--raw_dir /raw_aida/  --out_aida_dir /retriever_input/  \
--raw_kb_dir /raw_kb/ --out_kb_path /kb/entities_kilt.json \
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

## Train Reader 

## GERBIL evaluation
