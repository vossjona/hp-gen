import torch
import torch.nn.functional as F

from pathlib import Path

def build_dataset(context_length: int, words: list[str], ctoi: dict, itoc: dict) -> tuple[torch.tensor, torch.tensor]:
    inputs, labels = [], []
    for word in words:
        context = [0] * context_length
        
        for character in word:
            index = ctoi[character]
            inputs.append(context)
            labels.append(index)
            context = context[1:] + [index]
    inputs = torch.tensor(inputs)  # torch.tensor infers dtype; torch.Tensor is float by default
    labels = torch.tensor(labels)
    return inputs, labels


def create_embeddings(
    inputs: torch.tensor,
    embedding_table: torch.tensor
    ) -> torch.tensor:
    embedding = embedding_table[inputs]
    return embedding


def hidden_layer(
    embeddings: torch.tensor,
    hidden_layer_weights: torch.tensor,
    hidden_layer_biases: torch.tensor    
    ) -> torch.tensor:
    hidden_state = torch.tanh(embeddings.view(embeddings.shape[0], -1) @ hidden_layer_weights + hidden_layer_biases)  # views dont create new tensors that occupy memory
    return hidden_state


def calculate_logits(
    hidden_state: torch.tensor,
    logits_weigths: torch.tensor,
    logits_biases: torch.tensor
    ) -> torch.tensor:
    logits = hidden_state @ logits_weigths + logits_biases
    return logits


def train_model(
    inputs: torch.tensor,
    labels: torch.tensor,
    starting_parameters: dict,
    iterations: int,
    learning_rate: float
) -> None:
    for parameter in starting_parameters.values():
        parameter.requires_grad = True
    for iteration in range(iterations):
        # Forward pass
        embeddings = create_embeddings(
            inputs=inputs,
            embedding_table=starting_parameters['embedding_table'])
        
        hidden_state = hidden_layer(
            embeddings=embeddings,
            hidden_layer_weights=starting_parameters['hidden_layer_weights'],
            hidden_layer_biases=starting_parameters['hidden_layer_biases']
            )
        
        logits = calculate_logits(
            hidden_state=hidden_state,
            logits_weigths=starting_parameters['logits_weights'],
            logits_biases=starting_parameters['logits_biases']
            )
        
        nll_loss = F.cross_entropy(logits, labels)
        print(f'{iteration}: {nll_loss.item()}')
        
        # Backward pass
        for parameter in starting_parameters.values():
            parameter.grad = None
        nll_loss.backward()
        for parameter in starting_parameters.values():
            parameter.data -= learning_rate * parameter.grad


if __name__ == "__main__":
    data_set_path = Path('hp_style_words.txt')
    words = data_set_path.read_text().split('\n')
    
    characters = sorted(list(set(''.join(words) + '.')))
    ctoi = {c:i for i, c in enumerate(characters)}
    # {'.': 0, 'a': 1, 'b': 2, ... , 'z': 26}
    itoc = {i:c for c, i in ctoi.items()}
    # {0: '.', 1: 'a', 2: 'b', ... , 26: 'z'}
    
    context_length = 5 # how many characters influence the next character
    inputs, labels = build_dataset(context_length=context_length, words=words, ctoi=ctoi, itoc=itoc)
    
    embedding_dim = 4
    hidden_layer_neurons = 128
    
    starting_parameters = {
        'embedding_table': torch.randn(len(characters), embedding_dim),
        'hidden_layer_weights': torch.randn((inputs.shape[1] * embedding_dim, hidden_layer_neurons)),
        'hidden_layer_biases': torch.randn(hidden_layer_neurons),
        'logits_weights': torch.randn((hidden_layer_neurons, len(characters))),
        'logits_biases': torch.randn(len(characters))
    }
    
    train_model(inputs=inputs, labels=labels, starting_parameters=starting_parameters, iterations=1000, learning_rate=0.01)