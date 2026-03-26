from RL_Model.trainer import train_sac


def main():
    result = train_sac(
    service="voip",
    max_episodes=5,
    max_steps_per_episode=50,
    batch_size=16,
    warmup_steps=10,
    evaluation_interval=2,
    save_interval=2,
    replay_capacity=5000,
    checkpoint_dir="RL_Model/checkpoints_test",
    seed=42,
    verbose=True,
)

    print("\n=== TRAINING TEST SUMMARY ===")
    print("Total steps:", result["total_steps"])
    print("Saved checkpoints:")
    for ckpt in result["saved_checkpoints"]:
        print("  ", ckpt)

    history = result["history"]

    print("\nEpisode rewards:", history["episode_rewards"])
    print("Actor losses:", history["actor_losses"])
    print("Critic losses:", history["critic_losses"])
    print("Alpha values:", history["alpha_values"])
    print("Evaluation rewards:", history["evaluation_rewards"])

    assert len(history["episode_rewards"]) == 5, "Expected 5 episode rewards"
    assert len(history["actor_losses"]) == 5, "Expected 5 actor loss entries"
    assert len(history["critic_losses"]) == 5, "Expected 5 critic loss entries"
    assert len(history["alpha_values"]) == 5, "Expected 5 alpha entries"
    assert len(result["saved_checkpoints"]) >= 1, "Expected at least one checkpoint"

    print("\n✅ trainer.py test passed successfully.")


if __name__ == "__main__":
    main()