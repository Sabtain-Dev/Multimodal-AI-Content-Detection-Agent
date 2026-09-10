import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.detectors.text.ensemble import TextDetectorEnsemble

def run_agent_test():
    print("Initializing 3-Model Text Detector Agent (TMR + Multilingual + Gradient)...")
    agent = TextDetectorEnsemble(
        tmr_threshold=0.70,
        multilingual_threshold=0.50,
        gradient_threshold=0.50
    )

    test_samples = [
        {
            "description": "Clear Human Sample",
            "text": "Now i can say confidently that i know 50+% of the use of git and github. But the rest of 50% will be explore when i can work with more than 5 peoples on one same project."
        },
        {
            "description": "Clear AI Sample",
            "text": "Certainly! Here is a comprehensive, structured overview of machine learning algorithms utilized in modern clinical decision support systems."
        },
        {
            "description": "Ambiguous / Edge Case Sample",
            "text": "Docker is not working properly on my local setup. I tried running docker-compose up but it fails with exit code 1."
        },
        {
            "description": "Short Text Sample",
            "text": "Fix bug in main.py"
        }
    ]

    print("\n================ THREE-MODEL TEXT DETECTOR AGENT EXECUTION ================")
    for idx, sample in enumerate(test_samples, 1):
        print(f"\n--- Test {idx}: {sample['description']} ---")
        print(f"Snippet: \"{sample['text'][:70]}...\"")
        
        result = agent.detect(sample["text"])
        
        print("\nIndividual Model Outputs:")
        print(f"  TMR (RoBERTa)         -> Score: {result.tmr.ai_score:.4f} | Pred: {result.tmr.prediction}")
        print(f"  Multilingual (XLM-R)  -> Score: {result.multilingual.ai_score:.4f} | Pred: {result.multilingual.prediction}")
        print(f"  Gradient (DeBERTa-v3) -> Score: {result.gradient.ai_score:.4f} | Pred: {result.gradient.prediction}")
        print("-------------------------------------------------------------------------")
        print(f"  Final Prediction   : {result.final_prediction.upper()}")
        print(f"  Vote Count         : AI: {result.votes_ai} | Human: {result.votes_human}")
        print(f"  Agreement Strength : {result.agreement_strength}")
        if result.warning:
            print(f"  Warning            : {result.warning}")
    print("\n==========================================================================")

if __name__ == "__main__":
    run_agent_test()