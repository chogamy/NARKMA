def sejong(examples, tokenizers):
    """
    TODO: mode
    if mode == train~
    else:
        only src
    """

    inputs = {}

    encoder = tokenizers["src"](examples["src"], padding="max_length", max_length=512)
    """
    return here
    """
    decoder0 = tokenizers["morph"](
        examples["morph_tgt"], padding="max_length", max_length=512
    )
    decoder1 = tokenizers["tag"](
        examples["tag_tgt"], padding="max_length", max_length=512
    )

    inputs["enc_input_ids"] = encoder["input_ids"]
    inputs["enc_attention_mask"] = encoder["attention_mask"]

    inputs["dec0_tgt"] = decoder0["input_ids"]
    inputs["dec_attention_mask"] = decoder0["attention_mask"]

    inputs["dec1_tgt"] = decoder1["input_ids"]

    inputs["enc_tgt"] = tokenizers["length"].encode(
        examples["src"], examples["morph_tgt"], max_length=512
    )

    inputs["dec_input_ids"] = tokenizers["length"].decode(
        inputs["enc_tgt"], examples["morph_tgt"], max_length=512, mode="max"
    )

    return inputs


PREPROCESS = {"sejong": sejong}
