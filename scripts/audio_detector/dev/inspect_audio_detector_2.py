from transformers import AutoConfig

MODEL_ID = "Mahmoud59/wav2vec2-fake-audio-detector"

def main():
    print("=" * 60)
    print("INSPECTING AUDIO DETECTOR 2")
    print("=" * 60)
    try:
        config = AutoConfig.from_pretrained(MODEL_ID)
        print(f"Model ID       : {MODEL_ID}")
        print(f"Architecture   : {getattr(config, 'architectures', 'N/A')}")
        print(f"Model type     : {getattr(config, 'model_type', 'N/A')}")
        print(f"Num labels     : {getattr(config, 'num_labels', 'N/A')}")
        print(f"ID2LABEL       : {getattr(config, 'id2label', 'N/A')}")
        print(f"LABEL2ID       : {getattr(config, 'label2id', 'N/A')}")
        print(f"Sampling rate  : {getattr(config, 'sampling_rate', 'not specified')}")
    except Exception as e:
        print(f"Error inspecting model {MODEL_ID}: {e}")
    print("=" * 60)

if __name__ == "__main__":
    main()