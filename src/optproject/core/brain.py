from abc import ABC, abstractmethod

import numpy as np


class BrainModel(ABC):
    """
    Interface for any type of Brain

    Useful in case I decide later to implement the agents' controllers with
    other libraries, like pytorch
    """

    @abstractmethod
    def predict(self, inputs) -> int:
        """
        Args:
            inputs (np.ndarray): inputs from the environment

        Returns:
            int: choosen action
        """
        raise NotImplementedError

    @abstractmethod
    def clone_and_mutate(self, mutation_prob, mutation_amp):
        """
        Args:
            mutation_prob (float): probability of a gene mutating
            mutation_amp (float): amplitude of the mutation

        Returns:
            BrainModel: Brain of the offspring
        """
        raise NotImplementedError


class Brain(BrainModel):
    """
    Simple FFN implemented with numpy

    Genes of the agent are the weights of this net
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # TODO: provare a cercare se e meglio uniform o normal/gaussian
        self.W1 = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.b1 = np.random.uniform(-1, 1, (self.hidden_size,))
        self.W2 = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        self.b2 = np.random.uniform(-1, 1, (self.output_size,))

    def predict(self, inputs):
        """
        Performs a forward pass of the net and decides action

        Args:
            inputs (np.ndarray): inputs from the environment

        Returns:
            int: choosen action
        """

        # commentato perche spero non sbagliro a passare gli input
        # if not isinstance(inputs, np.ndarray):
        # inputs = np.array(inputs)
        x_in = np.asarray(inputs, dtype=np.float32).ravel()

        if x_in.shape[0] < self.input_size:
            pad = np.zeros(self.input_size - x_in.shape[0], dtype=np.float32)
            x_in = np.concatenate([x_in, pad])
        elif x_in.shape[0] > self.input_size:
            x_in = x_in[: self.input_size]

        # forward pass
        x = np.dot(x_in, self.W1) + self.b1
        x = np.maximum(0, x)  # ReLU
        x = np.dot(x, self.W2) + self.b2

        # TODO: argmax basta o devo fare softmax e poi scegliere il massimo?
        return int(np.argmax(x))

    def clone_and_mutate(self, mutation_prob, mutation_amp):
        """
        Simply creates a deepcopy of this brain and apply mutation

        Args:
            mutation_prob (float): probability of a gene mutating
            mutation_amp (float): amplitude of the mutation

        Returns:
            Brain: Brain of the offspring
        """

        child = Brain(self.input_size, self.hidden_size, self.output_size)

        child.W1 = self.W1.copy()
        child.W2 = self.W2.copy()
        child.b1 = self.b1.copy()
        child.b2 = self.b2.copy()

        for genome_part in [child.W1, child.b1, child.W2, child.b2]:

            # We choose which genes to mutate, if any
            mask = np.random.rand(*genome_part.shape) < mutation_prob
            # Then we mutate the genome in those indices
            mutation = np.random.normal(0, mutation_amp, genome_part.shape)
            genome_part[mask] += mutation[mask]

        return child


def run_brain_self_tests():
    print("#" * 50)
    print("Testing the ANN Brain from scratch implementation")

    input_neurons = 10
    hidden_neurons = 5
    output_neurons = 4  # These represent the number of actions

    # 1. Test Initialization
    parent_brain = Brain(input_neurons, hidden_neurons, output_neurons)
    print(f"Brain created with W1 shape: {parent_brain.W1.shape}")
    assert parent_brain.W1.shape == (input_neurons, hidden_neurons)

    # 2. Test Prediction
    dummy_input = np.random.rand(input_neurons)
    action = parent_brain.predict(dummy_input)
    print(f"Dummy input generated. Predicted action index: {action}")
    assert 0 <= action < output_neurons

    # 3. Test Cloning and Mutation
    child_brain = parent_brain.clone_and_mutate(mutation_prob=0.5, mutation_amp=0.1)
    print("Child brain created through cloning and mutation.")

    # Verify that the parent brain is unchanged
    parent_prediction_after = parent_brain.predict(dummy_input)
    print(f"Parent's prediction is still: {parent_prediction_after}")
    assert parent_prediction_after == action

    # Verify that the child brain is different
    # Due to mutation, they are extremely unlikely to be identical
    assert not np.array_equal(parent_brain.W1, child_brain.W1)
    print("Verified that child's genes (weights) are different from parent's.")

    print("\nNumpyBrain tests passed successfully!")


if __name__ == "__main__":
    run_brain_self_tests()
