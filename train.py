from tictactoe_rl import build_arg_parser, evaluate, train


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    policy, train_stats = train(episodes=args.episodes, lr=args.lr, seed=args.seed)
    eval_stats = evaluate(policy)
    policy.save(args.output, args.episodes, {"train": train_stats, "eval_vs_random": eval_stats})

    print(f"saved model to {args.output}")
    print(f"training stats: {train_stats}")
    print(f"evaluation vs random: {eval_stats}")


if __name__ == "__main__":
    main()

