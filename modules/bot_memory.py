from transformers import AutoTokenizer, BartForConditionalGeneration
import unicodedata
import torch

model = None
tokenizer = None


def init(model_name):
    global model, tokenizer
    model = BartForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)


settings = {
    "min_long_buffer_length": 1,
    "max_long_buffer_length": 512,
    "min_short_buffer_length": 1,
    "max_short_buffer_length": 1024,
    "long_buffer_length": 128,
    "short_buffer_length": 256,
    "repetition_penalty": 1.0,
    "temperature": 1.0
}

state = {"short_buffer": [["", ""]], "long_buffer": ["", ""]}


def reset_state():
    state["short_buffer"] = [["", ""]]
    state["long_buffer"] = ["", ""]


def get_memory(name1, name2):
    return "".join(
        [
            "\n",
            f"Memory of {name1}: ",
            "[",
            state["long_buffer"][0].strip(),
            "]",
            "\n",
            f"Memory of {name2}: ",
            "[",
            state["long_buffer"][1].strip(),
            "]",
            "\n",
            "<START>",
            "\n",
        ]
    )


# def format_entry(name1, name2, text, reply):
#    return f'{name1} says: "{text}"\n{name2} says: "{reply}"\n'


def update_state(model_encoder):
    print("Memory settings: ", settings)
    print("Memory state: ", state)
    name1memory = get_short_memory_string(0)
    name2memory = get_short_memory_string(1)
    summarize(model_encoder, name1memory, 0)
    summarize(model_encoder, name2memory, 1)
    state["short_buffer"] = [
        x for x in state["short_buffer"] if x[0] != "" or x[1] != ""
    ]


def get_short_memory_string(index):
    return (
        "\n".join([x[index] for x in state["short_buffer"] if x[index] != ""])
        .strip()
        .replace('"', "")
    )


def summarize(model_encoder, value, index):
    token_count = len(model_encoder(value)[0])

    # Summarize if short threshold is achieved
    if token_count < settings["short_buffer_length"]:
        return

    # Generate summary
    inputs = tokenizer(value, max_length=token_count, return_tensors="pt")
    bad_words_ids = [tokenizer(bad_word, add_special_tokens=False).input_ids for bad_word in ['\n', '"', '*']]
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=2,
        min_length=int(settings["long_buffer_length"] * 0.8),
        max_length=int(settings["long_buffer_length"]),
        repetition_penalty=float(settings["repetition_penalty"]),
        temperature=float(settings["temperature"]),
        bad_words_ids=bad_words_ids
    )
    summary = tokenizer.batch_decode(
        summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    # Normalize string
    summary = ' '.join(unicodedata.normalize('NFKC', summary).strip().split())
    # Swap buffers
    state["long_buffer"][index] = summary
    for entry in state["short_buffer"]:
        entry[index] = ""
    buffer_entry = ["", ""]
    buffer_entry[index] = summary
    state["short_buffer"].insert(0, buffer_entry)
