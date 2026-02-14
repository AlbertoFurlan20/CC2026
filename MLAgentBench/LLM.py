""" This file contains the code for calling all LLM APIs. """

import os
import openai
from functools import partial
import tiktoken
from .schema import TooLongPromptError, LLMError

enc = tiktoken.get_encoding("cl100k_base")

# ===================== CRFM (HELM) =====================
try:
    from helm.common.authentication import Authentication
    from helm.common.request import Request, RequestResult
    from helm.proxy.accounts import Account
    from helm.proxy.services.remote_service import RemoteService

    # setup CRFM API
    auth = Authentication(api_key=open("crfm_api_key.txt").read().strip())
    service = RemoteService("https://crfm-models.stanford.edu")
    account: Account = service.get_account(auth)
except Exception as e:
    print(e)
    print("Could not load CRFM API key crfm_api_key.txt.")

# ===================== Anthropic (Claude) =====================
anthropic = None
anthropic_client = None
try:
    import anthropic as _anthropic_mod
    anthropic = _anthropic_mod
    try:
        anthropic_client = anthropic.Anthropic(
            api_key=open("claude_api_key.txt").read().strip()
        )
    except Exception as e:
        print(e)
        print("Could not load anthropic API key claude_api_key.txt.")
except Exception as e:
    print(e)
    print("anthropic package not available; Claude models disabled.")

# ===================== VertexAI / Gemini =====================
try:
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel, Part
    from google.cloud.aiplatform_v1beta1.types import SafetySetting, HarmCategory

    vertexai.init(project=PROJECT_ID, location="us-central1")  # type: ignore[name-defined]
except Exception as e:
    print(e)
    print("Could not load VertexAI API.")

# ===================== HuggingFace local models =====================
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

loaded_hf_models = {}

class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, stop_sequence):
        super().__init__()
        self.stop_sequence = stop_sequence

    def __call__(self, input_ids, scores, **kwargs):
        stop_sequence_tensor = torch.tensor(
            self.stop_sequence,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        current_sequence = input_ids[:, -len(self.stop_sequence) :]
        return bool(torch.all(current_sequence == stop_sequence_tensor).item())

def log_to_file(log_file, prompt, completion, model, max_tokens_to_sample):
    """Log the prompt and completion to a file (sin depender de anthropic)."""
    with open(log_file, "a") as f:
        f.write("\n===================prompt=====================\n")
        f.write(prompt)
        num_prompt_tokens = len(enc.encode(prompt))

        f.write(
            f"\n==================={model} response ({max_tokens_to_sample})=====================\n"
        )
        f.write(completion)
        num_sample_tokens = len(enc.encode(completion))

        f.write("\n===================tokens=====================\n")
        f.write(f"Number of prompt tokens: {num_prompt_tokens}\n")
        f.write(f"Number of sampled tokens: {num_sample_tokens}\n")
        f.write("\n\n")

def complete_text_hf(
    prompt,
    stop_sequences=None,
    model="huggingface/codellama/CodeLlama-7b-hf",
    max_tokens_to_sample=2000,
    temperature=0.5,
    log_file=None,
    **kwargs,
):
    """
    Simple version of local HF (not using at the moment due to the vLLM, but kept here for potential future use).
    """
    if stop_sequences is None:
        stop_sequences = []

    model_id = model.split("/", 1)[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_id in loaded_hf_models:
        hf_model, tokenizer = loaded_hf_models[model_id]
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        loaded_hf_models[model_id] = (hf_model, tokenizer)

    encoded_input = tokenizer(
        prompt,
        return_tensors="pt",
        return_token_type_ids=False,
    ).to(device)

    stop_sequence_ids = tokenizer(
        stop_sequences,
        return_token_type_ids=False,
        add_special_tokens=False,
    )
    stopping_criteria = StoppingCriteriaList()
    for stop_sequence_input_ids in stop_sequence_ids.input_ids:
        stopping_criteria.append(
            StopAtSpecificTokenCriteria(stop_sequence=stop_sequence_input_ids)
        )

    output = hf_model.generate(
        **encoded_input,
        temperature=temperature,
        max_new_tokens=max_tokens_to_sample,
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria=stopping_criteria,
        **kwargs,
    )
    sequences = output.sequences
    sequences = [seq[len(encoded_input.input_ids[0]) :] for seq in sequences]
    all_decoded_text = tokenizer.batch_decode(sequences)
    completion = all_decoded_text[0]

    if log_file is not None:
        log_to_file(log_file, prompt, completion, model_id, max_tokens_to_sample)
    return completion

def complete_text_gemini(
    prompt,
    stop_sequences=None,
    model="gemini-pro",
    max_tokens_to_sample=2000,
    temperature=0.5,
    log_file=None,
    **kwargs,
):
    """Call the Gemini API to complete a prompt."""
    if stop_sequences is None:
        stop_sequences = []

    gen_model = GenerativeModel("gemini-pro")
    parameters = {
        "temperature": temperature,
        "max_output_tokens": max_tokens_to_sample,
        "stop_sequences": stop_sequences,
        **kwargs,
    }
    safety_settings = {
        # Unlock all to avoid weird blocks in experiments; rely on the user to not ask for harmful content, and we want to allow the model to say it if it "thinks" it's necessary for the task. Adjust as needed.
        harm_category: SafetySetting.HarmBlockThreshold(
            SafetySetting.HarmBlockThreshold.BLOCK_NONE
        )
        for harm_category in iter(HarmCategory)
    }

    response = gen_model.generate_content(
        [prompt],
        generation_config=parameters,
        safety_settings=safety_settings,
    )
    completion = response.text
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion

def complete_text_claude(
    prompt,
    stop_sequences=None,
    model="claude-v1",
    max_tokens_to_sample=2000,
    temperature=0.5,
    log_file=None,
    messages=None,
    **kwargs,
):
    """Call the Claude API to complete a prompt."""
    if anthropic is None or anthropic_client is None:
        raise LLMError("Anthropic / Claude client not configured in this environment.")

    if stop_sequences is None:
        stop_sequences = [anthropic.HUMAN_PROMPT]

    ai_prompt = anthropic.AI_PROMPT
    if "ai_prompt" in kwargs and kwargs["ai_prompt"] is not None:
        ai_prompt = kwargs["ai_prompt"]

    try:
        if model == "claude-3-opus-20240229":
            # new endpoint of mensages
            while True:
                try:
                    message = anthropic_client.messages.create(
                        messages=(
                            [{"role": "user", "content": prompt}]
                            if messages is None
                            else messages
                        ),
                        model=model,
                        stop_sequences=stop_sequences,
                        temperature=temperature,
                        max_tokens=max_tokens_to_sample,
                        **kwargs,
                    )
                except anthropic.InternalServerError:
                    # Simple retry
                    continue
                try:
                    completion = message.content[0].text
                    break
                except Exception:
                    print("end_turn???")
                    continue
        else:
            rsp = anthropic_client.completions.create(
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {ai_prompt}",
                stop_sequences=stop_sequences,
                model=model,
                temperature=temperature,
                max_tokens_to_sample=max_tokens_to_sample,
                **kwargs,
            )
            completion = rsp.completion

    except anthropic.APIStatusError as e:
        print(e)
        raise TooLongPromptError()
    except Exception as e:
        raise LLMError(e)

    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion


def get_embedding_crfm(text, model="openai/gpt-4-0314"):
    request = Request(
        model="openai/text-embedding-ada-002",
        prompt=text,
        embedding=True,
    )
    request_result: RequestResult = service.make_request(auth, request)
    return request_result.embedding

def complete_text_crfm(
    prompt="",
    stop_sequences=None,
    model="openai/gpt-4-0314",
    max_tokens_to_sample=2000,
    temperature=0.5,
    log_file=None,
    messages=None,
    **kwargs,
):
    if stop_sequences is None:
        stop_sequences = []

    random = log_file
    if messages:
        request = Request(
            prompt=prompt,
            messages=messages,
            model=model,
            stop_sequences=stop_sequences,
            temperature=temperature,
            max_tokens=max_tokens_to_sample,
            random=random,
        )
    else:
     
        request = Request(
            prompt=prompt,
            model=model,
            stop_sequences=stop_sequences,
            temperature=temperature,
            max_tokens=max_tokens_to_sample,
            random=random,
        )

    try:
        request_result: RequestResult = service.make_request(auth, request)
    except Exception as e:
 
        print(e)
        raise TooLongPromptError()

    if request_result.success is False:
        print(request_result.error)
        raise LLMError(request_result.error)
    completion = request_result.completions[0].text
    if log_file is not None:
        log_to_file(
            log_file,
            prompt if not messages else str(messages),
            completion,
            model,
            max_tokens_to_sample,
        )
    return completion

# ===================== OpenAI (local vLLM backend) =====================

OPENAI_BASE_URL = (
    os.getenv("OPENAI_API_BASE")
    or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8002/v1")
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "test")

# Config para openai==0.28.x
openai.api_base = OPENAI_BASE_URL
openai.api_key = OPENAI_API_KEY

def _map_logical_to_vllm_model(model: str) -> str:
    """
    Maps 'nice' names (the ones you pass from --llm-name) to the exact name that vLLM uses in --served-model-name.
    Adjust here if you change how you launch the server.
    """
    if not isinstance(model, str):
        return model

    m = model.strip()
    lower = m.lower()

    # Llama 3.1 8B
    if lower in {
        "llama-3.1-8b-instruct",
        "llama-3.1-8b",
        "llama3.1-8b-instruct",
        "llama-3.1-8b-instruct-hf",
    }:
        return "llama-3.1-8B-Instruct"

    # Qwen 2.5 7B
    if lower in {
        "qwen2.5-7b-instruct",
        "qwen-2.5-7b-instruct",
        "qwen-2_5-7b-instruct",
    }:
        return "qwen2.5-7b-instruct"

    # Here can add more local models in the future
    # if lower in {"other-modelo", "alias-modelo"}: return "server-name"

    return m

def complete_text_openai(
    prompt,
    stop_sequences=None,
    model="gpt-3.5-turbo",
    max_tokens_to_sample=300, #Change it dependending on the model context length 
    temperature=0.2,          #if we use the complete prompt (300) or just thought action action input (500)
    log_file=None,
    **kwargs,
):
    """
    Call the OpenAI-compatible API to complete a prompt (in this case vLLM).

    ALWAYS uses the ChatCompletion endpoint, independent of the model name
    because vLLM exposes both Llama and Qwen as chat models.
    Also, we map the logical name (--llm-name) to the actual vLLM name.
    """
    if stop_sequences is None:
        stop_sequences = []

    # Logical mapping -> real name in vLLM
    mapped_model = _map_logical_to_vllm_model(model)

    raw_request = {
        "model": mapped_model,
        "temperature": temperature,
        "max_tokens": max_tokens_to_sample,
        "stop": stop_sequences or None,  # API dont wants an empty list
        **kwargs,
    }

    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(messages=messages, **raw_request)
    completion = response["choices"][0]["message"]["content"]

    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion

# ===================== Dispatcher =====================

def complete_text(prompt, log_file, model, **kwargs):
    """Complete text using the specified model with appropriate API."""
    if model.startswith("claude"):
        # Anthropic / Claude
        stop_sequences = ["Observation:"]
        if anthropic is not None:
            stop_sequences.insert(0, anthropic.HUMAN_PROMPT)
        completion = complete_text_claude(
            prompt,
            stop_sequences=stop_sequences,
            log_file=log_file,
            model=model,
            **kwargs,
        )
    elif model.startswith("gemini"):
        completion = complete_text_gemini(
            prompt,
            stop_sequences=["Observation:"],
            log_file=log_file,
            model=model,
            **kwargs,
        )
    elif model.startswith("huggingface"):
        completion = complete_text_hf(
            prompt,
            stop_sequences=["Observation:"],
            log_file=log_file,
            model=model,
            **kwargs,
        )
    elif "/" in model:
        # CRFM API (model con formato "org/model")
        completion = complete_text_crfm(
            prompt,
            stop_sequences=["Observation:"],
            log_file=log_file,
            model=model,
            **kwargs,
        )
    else:
        # OpenAI / vLLM
        completion = complete_text_openai(
            prompt,
            stop_sequences=["Observation:"],
            log_file=log_file,
            model=model,
            **kwargs,
        )
    return completion

# specify fast models for summarization etc
# Limits based on llama 3.1 and qwen 2.5
CONTEXT_LIMIT_FAST = 4096  # if someday uses a bigger model 8k/32k, update this
MAX_OUTPUT_TOKENS_FAST = 128
MAX_INPUT_TOKENS_FAST = CONTEXT_LIMIT_FAST - MAX_OUTPUT_TOKENS_FAST - 64  # Security margin

def complete_text_fast(prompt, **kwargs):
    # 1) Cut the prompt if its to long
    try:
        tokens = enc.encode(prompt)
    except Exception:
        tokens = []
    if len(tokens) > MAX_INPUT_TOKENS_FAST:
        
        tokens = tokens[-MAX_INPUT_TOKENS_FAST:]
        prompt = enc.decode(tokens)

    # 2) Force a small max output tokens (for summaries, etc)
    max_tokens = kwargs.pop("max_tokens_to_sample", MAX_OUTPUT_TOKENS_FAST)

    return complete_text(
        prompt=prompt,
        model=FAST_MODEL,
        temperature=0.01,
        max_tokens_to_sample=max_tokens,
        log_file=kwargs.pop("log_file", None),
        **kwargs,
    )
