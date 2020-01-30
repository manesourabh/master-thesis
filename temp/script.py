"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.
Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.
"""

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.models.dqn.gym_ff_model import GymFfModel
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
# from rlpyt.models.mlp import MlpModel


# class GymMlpModel(MlpModel):
#     def __init__(self, **kwargs):
#         super().__init__(hidden_sizes=[128, 128], **kwargs)
#
#     def forward(self, observation, prev_action, prev_reward):
#         return self.model(observation)


class GymDqnAgent(DqnAgent):
    def __init__(self, ModelCls=GymFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(observation_shape=env_spaces.observation.shape,
                    action_size=env_spaces.action.n)


def build_and_train(env_id="CartPole-v1", run_ID=0, cuda_idx=0):
    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id),
        batch_T=1,  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0
    )
    algo = DQN(
        replay_size=int(5e4),
        learning_rate=5e-4,
        min_steps_learn=1000,
        target_update_interval=312,
        double_dqn=True,
        eps_steps=20000,
        clip_grad_norm=10
    )
    agent = GymDqnAgent(eps_final=0.02)
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=200000,
        log_interval_steps=1000,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict(env_id=env_id)
    name = "dqn_" + env_id
    log_dir = "scripts/dqn"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='CartPole-v1')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    args = parser.parse_args()
    build_and_train(
        env_id=args.env_id,
        run_ID=args.run_ID
    )
