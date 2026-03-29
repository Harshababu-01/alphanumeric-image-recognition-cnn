import argparse
import sys
from src.train import train_model
from src.evaluate import evaluate_model
from app.app import main as app_main # Failsafe legacy support pattern

def main():
    parser = argparse.ArgumentParser(description="AI Pipeline Interface")
    parser.add_argument("step", choices=["train", "evaluate", "web"], help="Execution Logic Block")
    parser.add_argument("--mode", choices=["digit", "character"], default="character", help="Define Model Space")
    
    args, _ = parser.parse_known_args()
    
    if args.step == "train":
        train_model(args.mode)
    elif args.step == "evaluate":
        evaluate_model(args.mode)
    elif args.step == "web":
        import subprocess
        print("Launching Flask Web Infrastructure...")
        subprocess.run(["python", "app.py"])

if __name__ == "__main__":
    main()
