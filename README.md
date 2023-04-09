# MuCoT: Multilingual Contrastive Training for Question-Answering in Low-resource Languages

Analysis of MuCoT: Multilingual Contrastive Training for Question-Answering in Low-resource Languages paper from ACL 2022
by Gokul Karthik Kumar, Abhishek Singh Gehlot, Sahal Shaji Mullappilly, Karthik Nandakumar

Detailed architecture and framwork of the source paper:  https://github.com/gokulkarthik/mucot/blob/main/README.md

Shell Command for execuetion:
python main.py --dataset chaii --dataset_augmentation translation --dataset_split_k 0 --langs hi ta ml^ te^ --min_langs 1 --langs_for_min_langs_filter hi ta bn^ mr^ ml^ te^ --wt_contrastive_loss 0.05 --contrastive_loss_layers 3 --agg_for_contrastive mean --temperature_for_contrastive -1 --max_steps_for_contrastive 1000 --max_steps 5000 --logging_steps 100 --eval_steps 100 --save_steps 100 --train_batch_size 16 --gradient_accumulation_steps 1 --eval_batch_size 16 --model_name mbert --eval False --debug False