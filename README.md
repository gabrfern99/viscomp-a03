[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/EtNqLmRf)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11520116&assignment_repo_type=AssignmentRepo)
# INF0417 - Visão Computacional

## Resumo
### Na introdução do relatório, apresentamos o problema significativo e crescente das doenças das plantas e sua detecção precoce. Demonstramos a necessidade crítica da detecção rápida e precisa das doenças das plantas para prevenir perdas agrícolas significativas e garantir a segurança alimentar. Introduzimos a ideia inovadora de usar técnicas de visão computacional para identificar doenças das plantas, destacando a promessa dessa abordagem em melhorar a eficiência e precisão da detecção de doenças em comparação com os métodos tradicionais.

### Na seção de fundamentação teórica, fizemos uma revisão aprofundada da literatura e dos conceitos fundamentais necessários para entender o nosso estudo. Primeiro, apresentamos uma visão geral das principais doenças das plantas, sua prevalência e seu impacto. Depois, discutimos o campo da visão computacional e sua aplicação na detecção de doenças das plantas. Explicamos como as imagens de plantas infectadas são processadas e analisadas por algoritmos de visão computacional para identificar sinais de doença. Também revisamos várias pesquisas anteriores que aplicaram técnicas de visão computacional para a detecção de doenças das plantas, enfatizando os métodos utilizados e os resultados alcançados.

## Running the code

### Install required libraries
```
pip install -r requirements.txt
```
### Cloning the repository
```
git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection
```
### Running the inference
```
python plant_disease_inference.py --image path_to_your_image.jpg
```
#### Example:
```
python plant_disease_inference.py --image sick_plant.jpg
```

### Results:
#### Acertos
```
python plant_disease_inference.py --image 'Café (Coffee) - BichoMineiro (Leaf Miner) - Cropped/bmin002.jpg' 2>/dev/null
1/1 [==============================] - 0s 393ms/step
Predicted class:  Café (Coffee)_BichoMineiro (Leaf Miner)
```
```
python plant_disease_inference.py --image 'Algodão (Cotton) - Mancha de Mirotecio (Myrothecium leaf spot) - 1/DSC_0103.jpg' 2>/dev/null
1/1 [==============================] - 0s 324ms/step
Predicted class:  Algodão (Cotton)_Mancha de Mirotecio (Myrothecium leaf spot)
```
#### Erros
```
python plant_disease_inference.py --image 'Mandioca (Cassava) - Mancha Parda (Brown leaf spot) - 1/DSC_0502.jpg' 2>/dev/null
1/1 [==============================] - 0s 327ms/step
Predicted class:  Café (Coffee)_Mancha Aureolada (Bacterial Blight)
```
```
python plant_disease_inference.py --image 'Coqueiro (Coconut Tree) - Fitotoxidez (Phytotoxicity) - 1/DSC08024 (2).jpg' 2>/dev/null
1/1 [==============================] - 0s 385ms/step
Predicted class:  Coqueiro (Coconut Tree)_Queima Folhas (Coconut leaf blight)
```

Equipe:

201802678 - GABRIEL FERNANDO FARIA DE OLIVEIRA

202105850 - ISADORA STEFANY R R MESQUITA

202005501 - PEDRO MARTINS BITTENCOURT

202105845 - GUILHERME HENRIQUE DOS REIS

202105847 - GUSTAVO DOS REIS OLIVEIRA
