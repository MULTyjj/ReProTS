# 设置参数
model_name = "ReProTS"
train_epochs = 10
learning_rate = 0.01
llama_layers = 32

batch_size = 24
comment = "ReProTS-ECL"

# 遍历不同的预测长度
pred_lengths = [96, 192, 336, 720]
for pred_len in pred_lengths:
    cmd = f"""
    accelerate launch --mixed_precision bf16 run_main.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model_id ECL_512_{pred_len} \
      --model {model_name} \
      --data ECL \
      --features M \
      --seq_len 512 \
      --label_len 48 \
      --pred_len {pred_len} \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --batch_size {batch_size} \
      --learning_rate {learning_rate} \
      --llm_layers {llama_layers} \
      --train_epochs {train_epochs} \
      --model_comment {comment}
    """
    print("Running command:")
    print(cmd)
    os.system(cmd)
