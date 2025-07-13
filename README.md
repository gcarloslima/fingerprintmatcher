# 🔍 Reconhecimento de Digitais com IA

Este projeto é uma aplicação simples feita com [Streamlit](https://streamlit.io/) para comparar duas imagens de digitais e calcular a **similaridade** entre elas usando **ResNet50** da biblioteca **TorchVision**.

## ✨ Funcionalidades

- Upload de duas imagens de digitais
- Pré-processamento das imagens (tons de cinza, redimensionamento, equalização de histograma)
- Extração de embeddings usando a camada intermediária da ResNet50
- Cálculo da similaridade entre as digitais com **Cosine Similarity**
- Interface simples, direta e interativa

## 📁 Estrutura do Projeto

```
├── app.py             # Código principal da aplicação
├── requirements.txt   # Dependências da aplicação
```

## ✅ Pré-requisitos

- Python 3.8 ou superior
- Git (opcional, para clonar o repositório)

## 🚀 Como Rodar Localmente

### 1. Clone o repositório (ou baixe os arquivos)

```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio
```

> Caso tenha apenas os arquivos `app.py` e `requirements.txt`, basta colocá-los em uma pasta e seguir os próximos passos.

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv venv
```

### 3. Ative o ambiente virtual

- **Windows:**

```bash
venv\Scripts\activate
```

- **Linux/Mac:**

```bash
source venv/bin/activate
```

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

### 5. Rode a aplicação

```bash
streamlit run app.py
```

A aplicação abrirá automaticamente no navegador, geralmente em: `http://localhost:8501`

---

## 🧪 Exemplo de Uso

1. Faça upload de duas imagens de impressões digitais (formatos suportados: JPG, PNG, TIF).
2. Veja a pré-visualização das imagens.
3. O modelo exibirá uma **porcentagem de similaridade**.
4. A aplicação mostrará uma análise do resultado:
   - ✅ Muito semelhantes
   - ⚠️ Alguma semelhança
   - ❌ Diferentes

---

## 📦 Exemplo de `requirements.txt`

Se ainda não tiver, aqui vai uma sugestão:

```
streamlit
torch
torchvision
numpy
scikit-learn
Pillow
opencv-python
```

---

## 📸 Captura de Tela

> 💡 Adicione uma imagem ou GIF da interface aqui, se desejar!

---

## 🧠 Baseado em

- [PyTorch](https://pytorch.org/)
- [Torchvision ResNet50](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
- [Streamlit](https://streamlit.io/)

---

## 📄 Licença

Este projeto é livre para uso educacional e pessoal. Fique à vontade para adaptar!

---

