from transformers import AutoConfig

MODEL_NAME = "eftt/VideoMae-ffc23-deepfake-detector"


def main():
    print("=" * 60)
    print("VIDEO TEMPORAL CONFIGURATION")
    print("=" * 60)

    config = AutoConfig.from_pretrained(MODEL_NAME)

    attributes = [
        "num_frames",
        "tubelet_size",
        "image_size",
        "patch_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
    ]

    print("\nKey Architectural Attributes:")
    for attribute in attributes:
        print(f"{attribute}: {getattr(config, attribute, None)}")

    print("\nFull Configuration Object:")
    print(config)
    print("=" * 60)


if __name__ == "__main__":
    main()