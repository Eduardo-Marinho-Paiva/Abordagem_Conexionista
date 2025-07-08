# Abordagem Conexionista - Treinamento de Rede Neural com MNIST

Este código implementa um modelo de rede neural convolucional (CNN) para o treinamento e avaliação em um dataset de imagens de dígitos manuscritos (MNIST). O código utiliza PyTorch e Torchvision para carregar e treinar o modelo, e visualiza as previsões do modelo com a ajuda do Matplotlib.

## Colaboradores
1. Eduardo Marinho de Paiva
3. João Guilherme Bezerra Santos
4. Luiz Henrique Alves Ferreira
5. Paulo Sérgio da Silva Medeiros
6. Vinicius Eduardo Freitas de Sales

## Como Rodar o Código

### Rodar na Sua Máquina

Para rodar o código localmente em sua máquina, siga os passos abaixo:

1. **Instale o Python** (caso ainda não tenha): Você pode baixar o Python [aqui](https://www.python.org/downloads/).
2. **Instale as dependências**: Abra o terminal (ou prompt de comando) e execute:

   ```bash
   pip install torch torchvision matplotlib
   ```
3. **Baixe o código**: Copie o código completo para um arquivo Python, por exemplo, `modelo_mnist.py`.
4. **Execute o código**: No terminal, navegue até a pasta onde o arquivo está localizado e execute:

   ```bash
   python modelo_mnist.py
   ```

### Rodar no Google Colab

Você também pode rodar o código diretamente no Google Colab. Para isso:

1. **Acesse o Google Colab**: Vá para o [Google Colab](https://colab.research.google.com/).
2. **Carregue o código**: No Google Colab, clique em **File > Upload notebook** e carregue o código salvo.
3. **Execute o código**: Basta executar as células do notebook e o treinamento ocorrerá diretamente no ambiente do Colab.

Aqui está o link direto para o código no Google Colab:
[Executar no Google Colab](https://colab.research.google.com/drive/1vZtEn1vW3CBuKx1SNjE5VJeuBJAKfLfP)

Nesse caso, apenas copie o "Repositório" do Google Colab com o Código
## Explicação do Código:

### Passo 0: Instalar as dependências
   Você precisará instalar as bibliotecas necessárias para rodar o código:

   ```bash
   pip install torch torchvision matplotlib
   ```

### Passo 1: Carregar o Dataset

O código usa o dataset MNIST para treinamento e teste. O **MNIST** contém imagens de dígitos manuscritos (0-9) e é um dos conjuntos de dados mais usados para problemas de classificação de imagens.

```python
from torchvision import datasets
from torchvision.transforms import ToTensor

dados_treinamento = datasets.MNIST(
    root='data',           # Caminho onde o dataset será armazenado
    train=True,            # Conjunto de dados para treinamento
    transform=ToTensor(),  # Converte as imagens em tensores
    download=True          # Baixa o dataset caso não esteja presente
)

dados_teste = datasets.MNIST(
    root='data',
    train=False,           # Conjunto de dados para teste
    transform=ToTensor(),
    download=True
)
```

### Passo 2: Configurar os Loaders

Para carregar os dados em lotes, é utilizado o `DataLoader` do PyTorch. Ele permite carregar os dados em mini-lotes (batch) e realiza o embaralhamento das amostras.

```python
from torch.utils.data import DataLoader

loaders = {
    'treinamento': DataLoader(dados_treinamento, batch_size=100, shuffle=True, num_workers=1),
    'teste': DataLoader(dados_teste, batch_size=100, shuffle=True, num_workers=1),
}
```

### Passo 3: Definir o Modelo de Rede Neural

A rede neural é definida como uma classe que herda de `nn.Module` do PyTorch. A arquitetura consiste em duas camadas convolucionais seguidas por camadas totalmente conectadas (fully connected), e a aplicação de regularização por dropout.

```python
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.conv_layer1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv_layer2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_layer_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv_layer1(x), 2))
        x = F.relu(F.max_pool2d(self.conv_layer_drop(self.conv_layer2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x)
```

### Passo 4: Configuração do Dispositivo (CPU/GPU)

Se uma GPU estiver disponível, o código usará a aceleração via CUDA. Caso contrário, o modelo será executado na CPU.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Modelo().to(device)
```

### Passo 5: Definir o Otimizador e Função de Perda

O código usa o otimizador Adam para atualizar os pesos do modelo durante o treinamento. A função de perda usada é a `CrossEntropyLoss`, apropriada para problemas de classificação.

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()
```

### Passo 6: Funções de Treinamento e Avaliação

As funções de treinamento e avaliação controlam o processo de aprendizado e a avaliação do modelo. Durante o treinamento, o modelo aprende com os dados e ajusta os pesos. Após cada época, a função de teste avalia o modelo no conjunto de dados de teste.

```python
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders['treinamento']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f"Epoch: {epoch} [{batch_idx * len(data)} / {len(loaders['treinamento'].dataset)}] "
                  f"({100. * batch_idx / len(loaders['treinamento']):.0f}%) \t Loss: {loss.item():.6f}")

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loaders['teste']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
    
    test_loss /= len(loaders['teste'].dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders['teste'].dataset)} "
          f"({100. * correct / len(loaders['teste'].dataset):.0f}%)")
```

### Passo 7: Executando o Treinamento

O modelo é treinado por 10 épocas. A cada época, a função `train()` é chamada para treinar o modelo e a função `test()` é chamada para avaliar o desempenho.

```python
for epoch in range(1, 11):
    train(epoch)
    test()
```

### Passo 8: Visualizar a Predição

Após o treinamento, o modelo é testado com um único exemplo do conjunto de dados de teste. O código exibe a imagem correspondente ao exemplo, junto com a predição do modelo.

```python
import matplotlib.pyplot as plt
model.eval()

data, target = dados_teste[34]
data = data.unsqueeze(0).to(device)

output = model(data)
prediction = output.argmax(dim=1, keepdim=True).item()

print(f"Prediction: {prediction}")

image = data.squeeze(0).squeeze(0).cpu().numpy()
plt.imshow(image, cmap='gray')
plt.show()
```

## Estrutura do Modelo

* **Camadas Convolucionais (Conv2d):** Extraem características das imagens, operando em pequenos "blocos" (ou janelas) da imagem.
* **Camada de Dropout (Dropout2d):** Ajuda a evitar overfitting ao desligar aleatoriamente algumas conexões durante o treinamento.
* **Camadas Totalmente Conectadas (Linear):** Após as camadas convolucionais, são usadas para classificar as imagens com base nas características extraídas.

## Resultado Esperado

O modelo treina e testa a acurácia do modelo de classificação de dígitos do MNIST. Durante o treinamento, o código imprimirá a perda de cada lote, e após o treinamento, o modelo mostrará a acurácia final no conjunto de testes. Além disso, o código exibirá uma imagem do dataset com sua predição.
