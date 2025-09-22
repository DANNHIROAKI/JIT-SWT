import argparse
import exp1
import exp2
import exp3

def main():
    parser = argparse.ArgumentParser(description="运行指定的实验任务。")
    parser.add_argument(
        '--exp', 
        type=int, 
        required=True, 
        choices=[1, 2, 3],
        help="要运行的实验编号 (1, 2, 或 3)。"
    )
    args = parser.parse_args()

    if args.exp == 1:
        exp1.run()
    elif args.exp == 2:
        exp2.run()
    elif args.exp == 3:
        exp3.run()

if __name__ == "__main__":
    main()

