
from model_manager import ModelManager, create_new_model
from random_network import RandomNetwork, PytorchAgentWrapper
from rl_agent import NewRLAgent
from noise import DirichletNoiseConfig
from train_network import train_network_steps
from self_play import self_play


def test():
    model_manager = ModelManager(model_class=NewRLAgent, save_dir='./shared_models/', device='cuda')
    weights = model_manager.load_model_weights('agent_v169.pth')
    model = PytorchAgentWrapper(weights, device='cuda')
    self_play(network=model, delay_between_turns=1.0, temperature=1.0, mcts_iterations=200, komi=0.0)


if __name__ == "__main__":
    test()

