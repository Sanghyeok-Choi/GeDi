import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from generate_GeDi import *

from nltk.tokenize import sent_tokenize


# ---------- Model Setting ---------- #
class Args:
    def __init__(self):
        self.model_type = "gpt2"  # There are no other options available
        self.gen_model_name_or_path = "gpt2-xl"  # gpt2-large / gpt2-medium / gpt2 ... Huggingface args
        self.gedi_model_name_or_path = None  # Assumes path from --mode if set to None
        self.config_name = ""
        self.tokenizer_name = ""
        self.cache_dir = ""
        self.do_lower_case = False
        self.no_cuda = False

        # 16 bit loating points
        self.fp16 = False
        self.load_in_half_prec = False

        # Arguments for generations
        self.gen_length = 40
        self.stop_token = None
        self.temperature = 1.0  # lower tend toward greedy sampling
        self.disc_weight = 30.0  # weight for GeDi discriminator
        self.filter_p = 0.8  # filters at up to filter_p cumulative probability from next token distribution
        self.class_bias = 0.0  # biases GeDi's classification probabilities
        self.target_p = 0.8  # In combination with filter_p, saves tokens with above target_p probability of being in the correct class
        self.do_sample = True  # If set to False greedy decoding is used. Otherwise sampling is used.
        self.repetition_penalty = 1.2
        self.rep_penalty_scale = 10.0
        self.penalize_cond = True
        self.k = None  # The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
        self.p = None  # The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

        self.gen_type = "gedi"  # gedi, cclm, or gpt2
        self.mode = "sentiment"  # topic, sentiment, or detoxify
        self.secondary_code = ""  # "n" for negative sampling
        self.gpt3_api_key = None
        self.prompt = ""


logger = logging.getLogger(__name__)

MAX_LENGTH = 2048

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}

def main(n_samples, args, fp_prefix="gedi"):
    assert (args.gen_type == "gedi" or args.gen_type == "cclm" or args.gen_type == "gpt2")
    assert args.mode == "sentiment"

    args.code_desired = "positive"
    args.code_undesired = "negative"
    if not args.gen_type == "gpt2":
        if args.gedi_model_name_or_path is None:
            args.gedi_model_name_or_path = "../pretrained_models/gedi_sentiment"
            if not os.path.isdir(args.gedi_model_name_or_path):
                raise Exception("GeDi model path not found, must either run `get_models.sh' or set `args.gedi_model_name_or_path'")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.gen_model_name_or_path, do_lower_case=False)

    if args.gen_type == "cclm":
        model = model_class.from_pretrained(args.gedi_model_name_or_path)
        model.to(args.device)
        args.length = adjust_length_to_model(args.gen_length, max_sequence_length=model.config.max_position_embeddings)
    else:
        if args.load_in_half_prec:
            model = model_class.from_pretrained(args.gen_model_name_or_path, load_in_half_prec=True)
            model.to(args.device)
            # even if using --fp16, apex doesn't accept half prec models, so requires converting to full prec before converting back
            model = model.float()
        else:
            model = model_class.from_pretrained(args.gen_model_name_or_path)
            model.to(args.device)

        # using fp16 option with GPT2-XL can prevent OOM errors
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            # opt_level O3 gives fully half precision weights. Okay for GPT-2 as long as we don't need to finetune
            model = amp.initialize(model, opt_level='O3')
            torch.cuda.empty_cache()

        args.length = adjust_length_to_model(args.gen_length, max_sequence_length=1024)


    if args.gen_type == "gedi":
        gedi_model = model_class.from_pretrained(args.gedi_model_name_or_path)
        gedi_model.to(args.device)
    else:
        gedi_model = None


    # ---------- Generation ---------- #
    gender_dict = {"male": {"@SBJ": "He", "@sbj": "he", "@obj": "him"},
                   "female": {"@SBJ": "She", "@sbj": "she", "@obj": "her"}}
    gender_lst = ["male", "female"]
    sentiment_lst = ["pos", "neg"]

    with open("general_prompts.txt", "r") as f:
        general_prompts = f.readlines()

    with open("negative_prompts.txt", "r") as f:
        negative_prompts = f.readlines()

    with open("positive_prompts.txt", "r") as f:
        positive_prompts = f.readlines()

    all_prompts = [general_prompts, negative_prompts, positive_prompts]

    for sentiment in sentiment_lst:
        args.code_desired = "positive" if sentiment=="pos" else "negative"
        args.code_undesired = "negative" if sentiment=="pos" else "negative"
        if args.gen_type == "cclm":
            prefix = tokenizer.encode(args.code_desired)
        for gender in gender_lst:
            for prompts, prompt_name in zip(all_prompts, ["general-prompts", "negative-prompts", "positive-prompts"]):
                file_path = f"data/{fp_prefix}_{gender}_{sentiment}_{prompt_name}.json"
                try:
                    with open(file_path, "r") as fp:
                        dataset = json.load(fp)
                except FileNotFoundError:
                    dataset = []
                    for i, prompt in enumerate(prompts):
                        prompt_revised = " ".join([gender_dict[gender].get(t, t) for t in prompt.strip().split()])
                        data_dic = {"id": i,
                                    "prompt": prompt,
                                    "prompt_revised": prompt_revised,
                                    "responses": []}
                        dataset.append(data_dic)
                    with open(file_path, "w") as fp:
                        json.dump(dataset, fp, indent=4)

                for data_dic in tqdm(dataset):
                    args.prompt = data_dic["prompt_revised"]

                    start_len = 0
                    text_ids = tokenizer.encode(args.prompt)
                    if args.gen_type == "cclm":
                        text_ids = [prefix] + text_ids
                        start_len = len(tokenizer.decode([prefix]))

                    encoded_prompts = torch.LongTensor(text_ids).unsqueeze(0).to(args.device)
                    encoded_prompts_batched = encoded_prompts.repeat(n_samples, 1)
                    multi_code = None
                    attr_class = 1

                    outputs = model.generate(
                        input_ids=encoded_prompts_batched,
                        pad_lens=None,
                        max_length=args.length,
                        temperature=args.temperature,
                        top_k=args.k,
                        top_p=args.p,
                        repetition_penalty=args.repetition_penalty,
                        rep_penalty_scale=args.rep_penalty_scale,
                        eos_token_ids=tokenizer.eos_token_id,
                        pad_token_id=0,
                        do_sample=args.do_sample,
                        penalize_cond=args.penalize_cond,
                        gedi_model=gedi_model,
                        gpt3_api_key=args.gpt3_api_key,
                        tokenizer=tokenizer,
                        disc_weight=args.disc_weight,
                        filter_p=args.filter_p,
                        target_p=args.target_p,
                        class_bias=args.class_bias,
                        attr_class=attr_class,
                        code_0=args.code_undesired,
                        code_1=args.code_desired,
                        multi_code=multi_code
                    )

                    data_dic['responses'] +=\
                        [{"generated_text": tokenizer.decode(output, clean_up_tokenization_spaces=True).split("<|endoftext|>")[0]}
                         for output in outputs.tolist()]
                    try:
                        for res in data_dic['responses']:
                            res["first_sent"] = sent_tokenize(res['generated_text'])[0]
                    except IndexError:
                        import pdb
                        pdb.set_trace()

                    with open(file_path, "w") as fp:
                        json.dump(dataset, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=100)
    opt = parser.parse_args()

    args = Args()
    args.disc_weight = 15.0

    main(opt.n_samples, args, fp_prefix=f"gedi_w15_{opt.n_samples}")