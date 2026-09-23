# Multi run comparison

## Terminal 1 - setup
```bash
docker build -t mlagentbench-thesis .
docker run --gpus all --user root -w /MLAgentBench \
  --name mlagentbench-thesis-ctr \
  -p 8001:8000 -p 8002:8002 \
  -v ${PWD}:/MLAgentBench \
  -v /data:/data \
  -it mlagentbench-thesis
conda activate vllm_srv
python -m vllm.entrypoints.openai.api_server \
  --model RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a16 \
  --served-model-name llama-3.1-8B-Instruct \
  --port 8002 
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096
```

## Terminal 2 - `compare_strategies.sh` script logs
```bash
(autogpt) root@2982e6598c12:/MLAgentBench# ./scripts/compare_strategies.sh 

=== Best-of-N n=1 (1 run) ===
[best_of_n_1] run_id=bon1_1779281728 score=NA time=156s status=ok

=== Best-of-N n=3 (3 runs, pick best) ===
[best_of_n_3] run_id=bon3_1779281894_1 score=NA time=70s status=ok
^C
(autogpt) root@2982e6598c12:/MLAgentBench# ^C
(autogpt) root@2982e6598c12:/MLAgentBench# ^C
(autogpt) root@2982e6598c12:/MLAgentBench# ./scripts/compare_strategies.sh 

=== Best-of-N n=1 (1 run) ===
[best_of_n_1] run_id=bon1_1779282204 score=NA time=121s status=ok

=== Best-of-N n=3 (3 runs, pick best) ===
[best_of_n_3] run_id=bon3_1779282332_1 score=NA time=250s status=ok
[best_of_n_3] run_id=bon3_1779282586_2 score=NA time=30s status=ok
[best_of_n_3] run_id=bon3_1779282620_3 score=NA time=48s status=ok
[best_of_n_3] no scored winner (all runs returned score=NA — check eval.log in each run dir)

=== GridSearch (3 top_p × 2 best_of = 6 runs) ===
NOTE: some vLLM versions ignore 'best_of' server-side — verify in run logs if best_of>1 has effect
[gridsearch] run_id=gs_tp07_bo1_1779282674 score=NA time=52s status=ok
[gridsearch] run_id=gs_tp07_bo3_1779282732 score=NA time=120s status=ok
[gridsearch] run_id=gs_tp09_bo1_1779282861 score=NA time=230s status=ok
[gridsearch] run_id=gs_tp09_bo3_1779283095 score=NA time=51s status=ok
[gridsearch] run_id=gs_tp10_bo1_1779283153 score=NA time=53s status=ok
[gridsearch] run_id=gs_tp10_bo3_1779283212 score=NA time=127s status=ok

=== All runs complete. Results: results/comparison.csv ===

strategy,top_p,best_of,n_samples,run_id,score,wall_time_s,status
best_of_n_1,default,default,default,bon1_1779275522,NA,22,failed
best_of_n_3,default,default,default,bon3_1779275544_1,NA,226,ok
best_of_n_3,default,default,default,bon3_1779275774_2,NA,16,failed
best_of_n_3,default,default,default,bon3_1779275791_3,NA,209,failed
gridsearch,0.7,1,1,gs_tp07_bo1_1779276001,NA,271,ok
gridsearch,0.7,3,1,gs_tp07_bo3_1779276275,NA,269,ok
gridsearch,0.9,1,1,gs_tp09_bo1_1779276553,NA,34,failed
gridsearch,0.9,3,1,gs_tp09_bo3_1779276588,NA,275,ok
gridsearch,1.0,1,1,gs_tp10_bo1_1779276867,NA,193,failed
gridsearch,1.0,3,1,gs_tp10_bo3_1779277061,NA,279,ok
best_of_n_1,default,default,default,bon1_1779281728,NA,156,ok
best_of_n_3,default,default,default,bon3_1779281894_1,NA,70,ok
best_of_n_1,default,default,default,bon1_1779282204,NA,121,ok
best_of_n_3,default,default,default,bon3_1779282332_1,NA,250,ok
best_of_n_3,default,default,default,bon3_1779282586_2,NA,30,ok
best_of_n_3,default,default,default,bon3_1779282620_3,NA,48,ok
gridsearch,0.7,1,1,gs_tp07_bo1_1779282674,NA,52,ok
gridsearch,0.7,3,1,gs_tp07_bo3_1779282732,NA,120,ok
gridsearch,0.9,1,1,gs_tp09_bo1_1779282861,NA,230,ok
gridsearch,0.9,3,1,gs_tp09_bo3_1779283095,NA,51,ok
gridsearch,1.0,1,1,gs_tp10_bo1_1779283153,NA,53,ok
gridsearch,1.0,3,1,gs_tp10_bo3_1779283212,NA,127,ok
```

Then:
```bash```

## vLLM logs
The vLLM logs are quite flooded with
```bash
(APIServer pid=3650) WARNING 05-20 13:20:05 [protocol.py:116] The following fields were present in the request but ignored: {'best_of'}
(APIServer pid=3650) ERROR 05-20 13:20:05 [serving_chat.py:255] Error in preprocessing prompt inputs
(APIServer pid=3650) ERROR 05-20 13:20:05 [serving_chat.py:255] Traceback (most recent call last):
(APIServer pid=3650) ERROR 05-20 13:20:05 [serving_chat.py:255]   File "/home/user/micromamba/envs/vllm_srv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py", line 237, in create_chat_completion
(APIServer pid=3650) ERROR 05-20 13:20:05 [serving_chat.py:255]     conversation, engine_prompts = await self._preprocess_chat(
(APIServer pid=3650) ERROR 05-20 13:20:05 [serving_chat.py:255]   File "/home/user/micromamba/envs/vllm_srv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1191, in _preprocess_chat
(APIServer pid=3650) ERROR 05-20 13:20:05 [serving_chat.py:255]     prompt_inputs = await self._tokenize_prompt_input_async(
(APIServer pid=3650) ERROR 05-20 13:20:05 [serving_chat.py:255]   File "/home/user/micromamba/envs/vllm_srv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1048, in _tokenize_prompt_input_async
(APIServer pid=3650) ERROR 05-20 13:20:05 [serving_chat.py:255]     async for result in self._tokenize_prompt_inputs_async(
(APIServer pid=3650) ERROR 05-20 13:20:05 [serving_chat.py:255]   File "/home/user/micromamba/envs/vllm_srv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1069, in _tokenize_prompt_inputs_async
(APIServer pid=3650) ERROR 05-20 13:20:05 [serving_chat.py:255]     yield await self._normalize_prompt_text_to_input(
(APIServer pid=3650) ERROR 05-20 13:20:05 [serving_chat.py:255]   File "/home/user/micromamba/envs/vllm_srv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 939, in _normalize_prompt_text_to_input
(APIServer pid=3650) ERROR 05-20 13:20:05 [serving_chat.py:255]     return self._validate_input(request, input_ids, input_text)
(APIServer pid=3650) ERROR 05-20 13:20:05 [serving_chat.py:255]   File "/home/user/micromamba/envs/vllm_srv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_engine.py", line 1020, in _validate_input
(APIServer pid=3650) ERROR 05-20 13:20:05 [serving_chat.py:255]     raise ValueError(
(APIServer pid=3650) ERROR 05-20 13:20:05 [serving_chat.py:255] ValueError: This model's maximum context length is 4096 tokens. However, your request has 4516 input tokens. Please reduce the length of the input messages.
```