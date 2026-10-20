from transformers import AutoConfig, AutoImageProcessor

MODEL_NAME = "eftt/VideoMae-ffc23-deepfake-detector"


def main():
    print("=" * 60)
    print("VIDEO DETECTOR INSPECTION")
    print("=" * 60)
    print(f"Model ID: {MODEL_NAME}\n")

    config = AutoConfig.from_pretrained(MODEL_NAME)
    print("Architecture:")
    print(config.architectures)

    print("\nModel type:")
    print(config.model_type)

    print("\nNumber of labels:")
    print(config.num_labels)

    print("\nID2LABEL:")
    print(config.id2label)

    print("\nLABEL2ID:")
    print(config.label2id)

    print("\nImage processor:")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    print(processor)

    if hasattr(processor, "size"):
        print("\nProcessor size:")
        print(processor.size)


if __name__ == "__main__":
    main()