srun2 python ../generate_GeDi.py \
--gen_length 40 \
--model_type gpt2 \
--gen_model_name_or_path gpt2-xl \
--disc_weight 30 \
--rep_penalty_scale 10 \
--filter_p 0.8 \
--target_p 0.8 \
--gen_type "gedi" \
--repetition_penalty 1.2 \
--mode "sentiment" \
--penalize_cond \
--do_sample