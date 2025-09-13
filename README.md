# Dissertation

# Steganography Detection Using Machine Learning

This repository contains the codebase for my MSc dissertation project:  
**"Steganography Detection Using Machine Learning Techniques"**.  

The project explores different machine learning and deep learning approaches to detect hidden data (stego images) within digital images. A dataset of **10,000 cover images** was generated, with stego images created using **LSB embedding** of random messages.

---

## 📂 Repository Structure

├── main.py # Dataset creation pipeline (downloads, embeds, splits)
├── Train_1.py # Algorithm 1: Simple Fully Connected Neural Network (baseline)
├── Train_2.py # Algorithm 2: CNN with High-Pass Filter (forensic-inspired)
├── Train_3.py # Algorithm 3: Deeper CNN (unstable, experimental)
├── Train_4.py # Algorithm 4: Transfer Learning with ResNet18
├── dataset/ # Generated dataset (train/test, cover/stego)
└── results/ # Output metrics, plots, and saved models


---

## ⚙️ Requirements

- Python 3.9+
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- aiohttp
- tqdm
- Pillow

##📊 Algorithms
Algorithm 1: Fully Connected Neural Network (Baseline)

Flattens images into 1D vectors.

Dense layers with ReLU activation.

Serves as a baseline — performance close to random guessing (~50%).

Algorithm 2: CNN with High-Pass Filter (Best Candidate)

Incorporates a fixed high-pass filter (SRM-inspired).

Convolutional layers with batch normalisation, dropout, and L2 regularisation.

Outputs probability of cover vs stego.

Showed some learning capability, though still near-chance.

Algorithm 3: Deeper CNN (Experimental)

A deeper CNN architecture with more layers.

Attempted to capture complex residuals.

Crashed after long runtimes (~13 hours) → excluded from final results.

Algorithm 4: Transfer Learning (ResNet18)

Uses pre-trained ResNet18 backbone (ImageNet).

Fine-tuned for binary classification (cover vs stego).

Collapsed to trivial predictions (always stego).

▶️ Running the Code
Step 1: Generate Dataset
python main.py

Downloads 10,000 cover images.

Embeds random 20-character messages into stego images (LSB).

Splits into train/ and test/ folders with cover/ and stego/ subfolders.

Step 2: Train an Algorithm
python Train_2.py


Replace with Train_1.py, Train_2.py, or Train_4.py depending on which algorithm to run.

Or

Run Directly in VS Code/Pycharm etc.

📁 Outputs

Each training script produces:

Saved model (.keras)

Classification report (classification_report.txt)

Plots (loss, accuracy, ROC AUC, confusion matrix)

📌 Notes

Dataset size and hardware limits restricted performance.

Results show challenges of applying generic ML/DL to steganalysis.

Future work: larger datasets, domain-specific CNNs (e.g., YeNet, SRNet), explainable AI, and GAN-based adversarial approaches.

👨‍🎓 Author
Filip Kasterski (21331206)
MSc Cyber Security Dissertation
Supervisor: Dr. Alex Akinbi



