from transformers import VideoMAEForVideoClassification

MODEL_NAME = "eftt/VideoMae-ffc23-deepfake-detector"


def main():
    print("=" * 60)
    print("VIDEO CHECKPOINT COMPATIBILITY DIAGNOSTIC")
    print("=" * 60)

    model, loading_info = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME,
        output_loading_info=True,
    )

    print(f"Model ID: {MODEL_NAME}")
    print(f"Architecture: {model.config.architectures}")
    
    # Convert sets to lists to ensure compatibility across transformer versions
    missing_keys = list(loading_info["missing_keys"])
    unexpected_keys = list(loading_info["unexpected_keys"])

    print(f"Missing keys count: {len(missing_keys)}")
    print(f"Unexpected keys count: {len(unexpected_keys)}")

    if missing_keys:
        print("\nMissing key examples (first 10):")
        for key in missing_keys[:10]:
            print(f"  - {key}")

    if unexpected_keys:
        print("\nUnexpected key examples (first 10):")
        for key in unexpected_keys[:10]:
            print(f"  - {key}")

    print("=" * 60)


if __name__ == "__main__":
    main()